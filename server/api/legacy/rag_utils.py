import os
import logging
import json
import time
import warnings
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Suppress MPS pin_memory warnings on Apple Silicon
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

# Core packages
from sentence_transformers import SentenceTransformer

# HTTP requests for RunPod API
import requests
import re

# NLTK for simple tokenization
import nltk
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# LangChain for text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# BM25 for keyword search
from rank_bm25 import BM25Okapi

# Removed Pinecone - using in-memory processing
import numpy as np

# Evaluation
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("All libraries imported (embedding model + RunPod vLLM API)!")


# Configuration
CONFIG = {
    "embedding_model": "BAAI/bge-large-en",  # Upgraded to large model (1024 dim)
    
    # RunPod vLLM Serverless API configuration
    "runpod_api_url": "https://api.runpod.ai/v2/itrscoi6yr4h5f/run",
    "runpod_api_key": None,  # Will be set from environment variable
    
    # LangChain RecursiveCharacterTextSplitter parameters
    "chunk_size": 300,       # characters per chunk
    "chunk_overlap": 50,     # overlapping characters (20% overlap)
    "separators": ["\n\n", "\n", " ", ""],  # Hierarchy of separators
    "retrieval_k": 30        # Increased from 10 to 30 for better context
}

# Set RunPod API key from environment variable
CONFIG["runpod_api_key"] = os.getenv('RUNPOD_API_KEY')
if not CONFIG["runpod_api_key"]:
    logger.warning("RUNPOD_API_KEY environment variable not set. RunPod API calls will fail.")

logger.info(f"Models selected:")
logger.info(f"  Embedding: {CONFIG['embedding_model']} (1024-dim, high-quality)")
logger.info(f"  Generation: RunPod vLLM API ({CONFIG['runpod_api_url']})")
logger.info(f"  Retrieval: Top-{CONFIG['retrieval_k']} chunks with intelligent context selection")



# Global variables for lazy loading
embedding_model = None
embedding_dimension = None
text_splitter = None

def parse_json_response(response_text: str) -> str:
    """
    Robust JSON parser that extracts and validates JSON from AI responses.
    Handles mixed content where JSON is followed by additional text.
    
    Args:
        response_text: The raw response from the AI model
        
    Returns:
        str: JSON string with all key-value pairs, or original text if no valid JSON found
    """
    try:
        # First, try to parse the entire response as JSON
        json.loads(response_text)
        return response_text
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from mixed content
    json_patterns = [
        # Look for JSON objects starting with { and ending with }
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        # Look for JSON that might span multiple lines
        r'\{[\s\S]*?\}',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text)
        for match in matches:
            try:
                # Try to parse each potential JSON match
                parsed = json.loads(match)
                if isinstance(parsed, dict) and parsed:  # Must be a non-empty dict
                    # Return the JSON as a formatted string
                    return json.dumps(parsed, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                continue
    
    # If no valid JSON found, try to extract structured data manually
    # Look for key-value patterns in the text
    extracted_data = {}
    
    # Pattern to match "key": "value" or key: value
    kv_patterns = [
        r'"([^"]+)":\s*"([^"]*)"',  # "key": "value"
        r'"([^"]+)":\s*([^,}\s]+)',  # "key": value (no quotes on value)
        r'([^":,{\s]+):\s*"([^"]*)"',  # key: "value" (no quotes on key)
        r'([^":,{\s]+):\s*([^,}\n]+)',  # key: value (no quotes)
    ]
    
    for pattern in kv_patterns:
        matches = re.findall(pattern, response_text)
        for key, value in matches:
            key = key.strip().strip('"').strip()
            value = value.strip().strip('"').strip().rstrip(',')
            
            # Clean up the value
            if value.lower() == 'null':
                value = None
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            else:
                # Try to convert to number if possible
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Keep as string
                    pass
            
            if key and key not in extracted_data:
                extracted_data[key] = value
    
    # If we extracted some data, return it as JSON
    if extracted_data:
        return json.dumps(extracted_data, indent=2, ensure_ascii=False)
    
    # Last resort: return original text
    return response_text


def extract_analysis_data(json_response: str) -> tuple[float, dict]:
    """
    Extract accuracy score and extracted fields from JSON response.
    
    Args:
        json_response: JSON string from AI response
        
    Returns:
        tuple: (accuracy_score, extracted_fields_dict)
    """
    accuracy_score = 0.0
    extracted_fields = {}
    
    try:
        # Try to parse the JSON response
        data = json.loads(json_response)
        
        if isinstance(data, dict):
            # Extract all fields as extracted_fields
            extracted_fields = dict(data)
            
            # Calculate a simple accuracy score based on completeness
            # Score based on how many fields have non-null values
            total_fields = len(data)
            completed_fields = sum(1 for value in data.values() if value is not None and str(value).strip())
            
            if total_fields > 0:
                accuracy_score = (completed_fields / total_fields) * 100
            else:
                accuracy_score = 0.0
                
            # If there's a specific confidence or score field, use that instead
            score_fields = ['confidence', 'score', 'accuracy', 'confidence_score']
            for field in score_fields:
                if field in data and isinstance(data[field], (int, float)):
                    accuracy_score = float(data[field])
                    if accuracy_score <= 1.0:  # Convert 0-1 scale to 0-100
                        accuracy_score *= 100
                    break
                    
    except json.JSONDecodeError:
        # If not valid JSON, try to extract basic info
        extracted_fields = {"raw_response": json_response[:500]}  # Store first 500 chars
        accuracy_score = 50.0  # Default score for non-JSON responses
        
    return accuracy_score, extracted_fields


def initialize_models():
    """Lazy load models only when needed"""
    global embedding_model, embedding_dimension, text_splitter
    
    if embedding_model is None:
        logger.info(f"Loading embedding model: {CONFIG['embedding_model']}")
        embedding_model = SentenceTransformer(CONFIG['embedding_model'])
        embedding_dimension = embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded! Dimension: {embedding_dimension}")
    
    if text_splitter is None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG['chunk_size'],
            chunk_overlap=CONFIG['chunk_overlap'],
            separators=CONFIG['separators'],
            length_function=len,
            is_separator_regex=False
        )
        logger.info(f"LangChain text splitter initialized")
    
    # Generation handled by RunPod vLLM API - no local model needed

logger.info("RAG utilities module loaded (models will be initialized on first use)")


def chunk_documents(documents: List[str], doc_ids: Optional[List[str]] = None) -> List[Dict]:
    """
    Chunk documents using LangChain RecursiveCharacterTextSplitter
    """
    # Initialize models if needed
    initialize_models()
    
    if doc_ids and len(doc_ids) != len(documents):
        raise ValueError("doc_ids length must match documents length")
    
    all_chunks = []
    
    for i, doc_text in enumerate(documents):
        doc_id = doc_ids[i] if doc_ids else f"doc_{i}"
        
        # Create LangChain Document object
        doc = Document(
            page_content=doc_text,
            metadata={"doc_id": doc_id, "original_length": len(doc_text)}
        )
        
        # Split using LangChain text splitter
        split_docs = text_splitter.split_documents([doc])
        
        # Convert to our format
        for chunk_idx, split_doc in enumerate(split_docs):
            chunk_text = split_doc.page_content
            word_count = len(chunk_text.split())
            char_count = len(chunk_text)
            
            all_chunks.append({
                "id": f"{doc_id}_chunk_{chunk_idx}",
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(split_docs),
                    "word_count": word_count,
                    "char_count": char_count,
                    "original_length": len(doc_text)
                }
            })
    
    # Calculate statistics
    total_chunks = len(all_chunks)
    avg_words = sum(c['metadata']['word_count'] for c in all_chunks) / total_chunks if total_chunks else 0
    avg_chars = sum(c['metadata']['char_count'] for c in all_chunks) / total_chunks if total_chunks else 0
    
    logger.info(f"Created {total_chunks} LangChain chunks from {len(documents)} documents")
    logger.info(f"Average per chunk: {avg_words:.1f} words, {avg_chars:.0f} characters")
    
    return all_chunks

def vectorize_chunks(chunks: List[Dict]) -> List[Dict]:
    """Generate embeddings for chunks using BAAI/bge-small-en"""
    # Initialize models if needed
    initialize_models()
    
    if not chunks:
        return []
    
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    start_time = time.time()
    
    # Extract texts
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings in batch
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Important for cosine similarity
    )
    
    # Add embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
    
    total_time = time.time() - start_time
    logger.info(f" Embeddings generated successfully in {total_time:.2f} seconds")
    logger.info(f"   Average time per chunk: {total_time/len(chunks):.3f}s")
    
    return chunks




def semantic_search_in_memory(query: str, chunks: List[Dict], top_k: int = 10) -> List[Dict]:
    """In-memory semantic search using cosine similarity"""
    # Initialize models if needed
    initialize_models()
    
    if not chunks:
        return []
    
    try:
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]
        
        # Extract chunk embeddings
        chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks])
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results
        semantic_results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                semantic_results.append({
                    "id": chunks[idx]["id"],
                    "score": float(similarities[idx]),
                    "text": chunks[idx]["text"],
                    "doc_id": chunks[idx]["metadata"]["doc_id"],
                    "chunk_idx": chunks[idx]["metadata"]["chunk_idx"],
                    "source": "semantic"
                })
        
        search_time = time.time() - start_time
        logger.info(f"In-memory semantic search completed in {search_time:.3f}s")
        
        return semantic_results
        
    except Exception as e:
        logger.error(f"In-memory semantic search failed: {e}")
        return []




def keyword_search_in_memory(query: str, chunks: List[Dict], top_k: int = 10) -> List[Dict]:
    """In-memory keyword search using BM25"""
    if not chunks:
        return []
    
    try:
        start_time = time.time()
        
        # Extract text from chunks
        documents = [chunk["text"] for chunk in chunks]
        
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Build BM25 index
        bm25 = BM25Okapi(tokenized_docs)
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = bm25.get_scores(query_tokens)
        
        # Get top-k documents
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Format results
        keyword_results = []
        for idx in top_indices:
            if scores[idx] > 0.1:  # Minimum BM25 score threshold
                keyword_results.append({
                    "id": chunks[idx]["id"],
                    "score": float(scores[idx]),
                    "text": chunks[idx]["text"],
                    "doc_id": chunks[idx]["metadata"]["doc_id"],
                    "chunk_idx": chunks[idx]["metadata"]["chunk_idx"],
                    "source": "keyword"
                })
        
        search_time = time.time() - start_time
        logger.info(f"In-memory keyword search completed in {search_time:.3f}s")
        
        return keyword_results
        
    except Exception as e:
        logger.error(f"In-memory keyword search failed: {e}")
        return []


# Removed old Pinecone-based semantic_search and keyword_search functions
# Now using semantic_search_in_memory and keyword_search_in_memory


def process_document_with_query(document_text: str, query: str, method: str = "hybrid", top_k: int = 30) -> Dict:
    """
    Process a single document with a query in-memory (no persistent storage)
    
    Args:
        document_text: The text content of the document
        query: The user's question/query
        method: "semantic", "keyword", or "hybrid"
        top_k: Number of chunks to retrieve
        
    Returns:
        Dictionary containing the answer and metadata
    """
    try:
        start_time = time.time()
        logger.info(f"Processing document with {method.upper()} retrieval: {query}")
        
        # 1. Chunk the document
        chunks = chunk_documents([document_text], doc_ids=["uploaded_doc"])
        if not chunks:
            return {
                "answer": "No content could be extracted from the document.",
                "error": "Document chunking failed",
                "num_docs_retrieved": 0,
                "doc_sources": [],
                "evaluation": {"overall_score": 0.0}
            }
        
        # 2. Generate embeddings for chunks
        chunks_with_embeddings = vectorize_chunks(chunks)
        
        # 3. Perform retrieval based on method
        if method == "semantic":
            relevant_chunks = semantic_search_in_memory(query, chunks_with_embeddings, top_k)
        elif method == "keyword":
            relevant_chunks = keyword_search_in_memory(query, chunks_with_embeddings, top_k)
        elif method == "hybrid":
            # Use more aggressive retrieval for better coverage
            semantic_results = semantic_search_in_memory(query, chunks_with_embeddings, int(top_k * 0.7))  # 70% semantic
            keyword_results = keyword_search_in_memory(query, chunks_with_embeddings, int(top_k * 0.5))   # 50% keyword
            relevant_chunks = reciprocal_rank_fusion(semantic_results, keyword_results)
        else:
            return {
                "answer": "Invalid retrieval method. Use 'semantic', 'keyword', or 'hybrid'.",
                "error": "Invalid method",
                "num_docs_retrieved": 0,
                "doc_sources": [],
                "evaluation": {"overall_score": 0.0}
            }
        
        retrieval_time = time.time() - start_time
        logger.info(f"Retrieval completed in {retrieval_time:.2f}s")
        
        if not relevant_chunks:
            return {
                "answer": "No relevant information found in the document for your query.",
                "error": "No relevant chunks found",
                "num_docs_retrieved": 0,
                "doc_sources": [],
                "evaluation": {"overall_score": 0.0}
            }
        
        # 4. Generate response using RunPod vLLM API
        answer = generate_response_runpod(query, relevant_chunks[:top_k])
        
        # 5. Evaluate response
        evaluation = evaluate_response_simple(query, answer, relevant_chunks[:top_k])
        
        overall_time = time.time() - start_time
        logger.info(f"Total processing time: {overall_time:.2f}s")
        
        # 6. Return structured response
        doc_sources = list(set([chunk.get("doc_id", "uploaded_doc") for chunk in relevant_chunks[:top_k]]))
        
        return {
            "answer": answer,
            "num_docs_retrieved": len(relevant_chunks),
            "doc_sources": doc_sources,
            "evaluation": evaluation,
            "processing_time": overall_time,
            "retrieval_method": method,
            "chunks_processed": len(chunks),
            "relevant_chunks": len(relevant_chunks)
        }
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        return {
            "answer": "An error occurred while processing your query.",
            "error": str(e),
            "num_docs_retrieved": 0,
            "doc_sources": [],
            "evaluation": {"overall_score": 0.0}
        }


def reciprocal_rank_fusion(semantic_results: List[Dict], keyword_results: List[Dict], k: int = 60) -> List[Dict]:
    """Combine semantic and keyword results using Reciprocal Rank Fusion"""
    
    # Create score dictionaries
    rrf_scores = {}
    
    # Add semantic search scores
    for rank, result in enumerate(semantic_results):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    # Add keyword search scores
    for rank, result in enumerate(keyword_results):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    # Create combined results
    all_results = {}
    
    # Add all results to dictionary
    for result in semantic_results + keyword_results:
        doc_id = result["id"]
        if doc_id not in all_results:
            all_results[doc_id] = result.copy()
            all_results[doc_id]["rrf_score"] = rrf_scores[doc_id]
            all_results[doc_id]["sources"] = [result["source"]]
        else:
            if result["source"] not in all_results[doc_id]["sources"]:
                all_results[doc_id]["sources"].append(result["source"])
    
    # Sort by RRF score
    fused_results = list(all_results.values())
    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    return fused_results

def generate_response_runpod(query: str, context_docs: List[Dict], max_tokens: int = 800) -> str:
    """Generate response using RunPod vLLM serverless API"""
    
    if not context_docs:
        return "I couldn't find relevant information to answer your question."
    
    if not CONFIG["runpod_api_key"]:
        return "RunPod API key not configured. Please set RUNPOD_API_KEY environment variable."
    
    start_time = time.time()
    
    # Prepare context with intelligent selection (top-10 from potentially 30 chunks)
    context_parts = []
    max_context_length = 4000  # Increased context window
    current_length = 0
    
    # Sort by score and take the best chunks that fit in context window
    sorted_docs = sorted(context_docs, key=lambda x: x.get('score', 0), reverse=True)
    
    for i, doc in enumerate(sorted_docs):
        chunk_text = doc['text']
        # Add chunk if it fits in the context window
        if current_length + len(chunk_text) + 50 < max_context_length:  # 50 chars for formatting
            context_parts.append(f"[Source {i+1}]: {chunk_text}")
            current_length += len(chunk_text) + 50
        if len(context_parts) >= 10:  # Max 10 chunks for LLM
            break
    
    context = "\n\n".join(context_parts)
    
    # Create prompt for the model with JSON format instruction
    prompt = f"""You are an AI document analysis assistant. Based on the provided context from the document, answer the user's question in a structured JSON format.

Context: {context}

Question: {query}

Instructions:
- Provide your response in valid JSON format only
- Include key-value pairs that are relevant to the question
- Extract specific facts, dates, numbers, names, and other important details
- Use clear, descriptive keys
- If information is not available in the context, indicate this clearly
- Be accurate and only include information that is explicitly stated or can be directly inferred

Response (JSON format only):"""
    
    try:
        # Prepare RunPod API request
        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CONFIG['runpod_api_key']}"
        }
        
        # Make request to RunPod API
        logger.info("Sending request to RunPod vLLM API...")
        response = requests.post(
            CONFIG["runpod_api_url"],
            json=payload,
            headers=headers,
            timeout=30  # 30 second timeout
        )
        
        if response.status_code != 200:
            logger.error(f"RunPod API error: {response.status_code} - {response.text}")
            return "Sorry, the text generation service is temporarily unavailable."
        
        result = response.json()
        
        # Handle RunPod serverless async response
        if "id" in result and result.get("status") in ["IN_QUEUE", "IN_PROGRESS"]:
            job_id = result["id"]
            logger.info(f"RunPod job queued: {job_id}. Polling for results...")
            
            # Poll for results (max 5 minutes)
            status_url = f"https://api.runpod.ai/v2/itrscoi6yr4h5f/status/{job_id}"
            max_polls = 30  # Max 30 attempts (30 × 10 seconds = 5 minutes)
            poll_count = 0
            
            while poll_count < max_polls:
                time.sleep(10)  # Wait 10 seconds between polls
                poll_count += 1
                
                try:
                    status_response = requests.get(status_url, headers=headers, timeout=10)
                    if status_response.status_code == 200:
                        status_result = status_response.json()
                        
                        if status_result.get("status") == "COMPLETED":
                            result = status_result
                            logger.info(f"RunPod job completed after {poll_count * 10} seconds")
                            break
                        elif status_result.get("status") == "FAILED":
                            logger.error(f"RunPod job failed: {status_result}")
                            return "Sorry, the text generation service failed to process your request."
                        
                        # Continue polling if still IN_PROGRESS or IN_QUEUE
                        logger.info(f"Polling {poll_count}/30 (every 10s): Status = {status_result.get('status')}")
                    else:
                        logger.warning(f"Status check failed: {status_response.status_code}")
                        
                except Exception as e:
                    logger.warning(f"Error checking status: {e}")
            
            if poll_count >= max_polls:
                logger.error("RunPod job timed out after 5 minutes")
                return "Sorry, the text generation request timed out after 5 minutes. Please try again."
        
        # Extract generated text from RunPod response
        if "output" in result and result["output"]:
            output = result["output"]
            
            # Handle different RunPod response formats
            if isinstance(output, list) and len(output) > 0:
                # New format: output is a list with choices
                first_output = output[0]
                if "choices" in first_output and len(first_output["choices"]) > 0:
                    choice = first_output["choices"][0]
                    if "tokens" in choice and len(choice["tokens"]) > 0:
                        generated_text = choice["tokens"][0]
                        logger.info(f"Extracted text from tokens: {len(generated_text)} characters")
                    elif "text" in choice:
                        generated_text = choice["text"]
                        logger.info(f"Extracted text from text field: {len(generated_text)} characters")
                    else:
                        generated_text = str(choice)
                        logger.info(f"Extracted text from choice object: {len(generated_text)} characters")
                else:
                    generated_text = str(first_output)
            elif isinstance(output, dict) and "text" in output:
                # Old format: output is a dict with text
                generated_text = output["text"]
            elif isinstance(output, str):
                # Simple string format
                generated_text = output
            else:
                logger.error(f"Unexpected RunPod response format: {result}")
                return "Sorry, I received an unexpected response from the generation service."
        else:
            logger.error(f"No output in RunPod response: {result}")
            return "Sorry, no response was generated."
        
        # Extract only the answer part (after "Answer:")
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            # If prompt is included in response, remove it
            answer = generated_text.replace(prompt, "").strip()
        
        # Clean up the answer
        if "You are an AI assistant" in answer:
            answer = answer.split("You are an AI assistant")[0].strip()
        if "Task:" in answer:
            answer = answer.split("Task:")[0].strip()
        
        # Remove any trailing incomplete sentences
        answer_lines = answer.split('\n')
        clean_lines = []
        for line in answer_lines:
            line = line.strip()
            if line and not line.startswith("Context"):
                clean_lines.append(line)
            if len(clean_lines) >= 10:  # Limit response length
                break
        
        answer = '\n'.join(clean_lines) if clean_lines else answer
        
        # Ensure minimum length
        if not answer or len(answer.strip()) < 10:
            return "I need more specific context to provide a comprehensive answer."
        
        generation_time = time.time() - start_time
        logger.info(f"RunPod vLLM response generated in {generation_time:.2f}s")
        
        # Parse JSON from the response
        parsed_response = parse_json_response(answer.strip())
        
        return parsed_response
        
    except requests.Timeout:
        logger.error("RunPod API request timed out")
        return "Sorry, the response generation timed out. Please try again."
    except requests.RequestException as e:
        logger.error(f"RunPod API request failed: {e}")
        return "Sorry, I encountered a network error while generating the response."
    except Exception as e:
        logger.error(f"RunPod generation failed: {e}")
        return "Sorry, I encountered an error while generating the response."


def evaluate_response_simple(query: str, response: str, context_docs: List[Dict]) -> Dict:
    """Simple evaluation without generating follow-up questions"""
    
    # Simple metrics
    response_length = len(response.split())
    context_used = len(context_docs)
    
    # Check if response contains key terms from query
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    word_overlap = len(query_words.intersection(response_words)) / len(query_words) if query_words else 0
    
    # Semantic similarity between query and response
    try:
        query_embedding = embedding_model.encode([query])
        response_embedding = embedding_model.encode([response])
        semantic_similarity = cosine_similarity(query_embedding, response_embedding)[0][0]
    except:
        semantic_similarity = 0.0
    
    # Context relevance (average similarity between query and context docs)
    context_relevance = 0.0
    if context_docs:
        try:
            context_texts = [doc['text'] for doc in context_docs[:3]]
            context_embeddings = embedding_model.encode(context_texts)
            query_embedding = embedding_model.encode([query])
            relevance_scores = cosine_similarity(query_embedding, context_embeddings)[0]
            context_relevance = np.mean(relevance_scores)
        except:
            context_relevance = 0.0
    
    return {
        "query": query,
        "response": response,
        "response_length": response_length,
        "context_docs_used": context_used,
        "word_overlap_score": float(word_overlap),
        "semantic_similarity": float(semantic_similarity),
        "context_relevance": float(context_relevance),
        "overall_score": float((word_overlap + semantic_similarity + context_relevance) / 3)
    }



def rag_chat_flexible(query: str, method: str = "hybrid", top_k: int = 10) -> Dict:
    """
    Flexible RAG pipeline - supports any retrieval method
    
    Args:
        query: Question to ask
        method: "semantic", "keyword", "hybrid" (default: "hybrid")
        top_k: Number of chunks to retrieve
    """
    logger.info(f" Processing query with {method.upper()} retrieval: {query}")
    overall_start = time.time()
    
    try:
        # Route to appropriate retrieval method
        retrieval_start = time.time()
        if method == "semantic":
            return {"answer": "This function is deprecated. Use process_document_with_query() instead.", "error": "Function deprecated"}
        elif method == "keyword":
            return {"answer": "This function is deprecated. Use process_document_with_query() instead.", "error": "Function deprecated"}
        elif method == "hybrid":
            return {"answer": "This function is deprecated. Use process_document_with_query() instead.", "error": "Function deprecated"}
            retrieved_docs = reciprocal_rank_fusion(sem_results, key_results)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'semantic', 'keyword', or 'hybrid'")
        
        retrieval_time = time.time() - retrieval_start
        logger.info(f" Retrieval completed in {retrieval_time:.2f}s")
        
        # Generate response
        generation_start = time.time()
        response = generate_response_runpod(query, retrieved_docs)
        generation_time = time.time() - generation_start
        
        evaluation = evaluate_response_simple(query, response, retrieved_docs)
        
        overall_time = time.time() - overall_start
        logger.info(f" Total RAG pipeline time: {overall_time:.2f}s")
        
        return {
            "query": query,
            "answer": response,
            "retrieval_method": method,
            "num_docs_retrieved": len(retrieved_docs),
            "retrieved_docs": retrieved_docs[:3],
            "doc_sources": list(set([doc["doc_id"] for doc in retrieved_docs])),
            "evaluation": evaluation,
            "timing": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": overall_time
            }
        }
        
    except Exception as e:
        logger.error(f" Chat pipeline failed: {e}")
        return {
            "query": query,
            "answer": f"Error: {e}",
            "retrieval_method": method,
            "num_docs_retrieved": 0,
            "retrieved_docs": [],
            "doc_sources": [],
            "evaluation": {},
            "timing": {}
        }


# DEPRECATED: No longer needed with single-request processing
def add_documents_to_system(documents: List[str], doc_ids: Optional[List[str]] = None) -> bool:
    """
    DEPRECATED: Complete pipeline to add documents using LangChain text splitter
    Use process_document_with_query() for single-request processing instead.
    """
    logger.warning("add_documents_to_system() is deprecated. Use process_document_with_query() instead.")
    return False

# Load your parsed text files
def load_extracted_text_files(directory_path: str = "extracted_texts") -> List[str]:
    """Load text files from your document processing pipeline"""
    from pathlib import Path
    
    texts_dir = Path(directory_path)
    if not texts_dir.exists():
        logger.error(f" Directory not found: {directory_path}")
        return []
    
    # Find all extracted text files
    text_files = list(texts_dir.glob("*_extracted_text.txt"))
    
    documents = []
    doc_ids = []
    
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content:  # Only add non-empty files
                # Extract document name from filename
                doc_name = file_path.stem.replace("_extracted_text", "")
                
                documents.append(content)
                doc_ids.append(doc_name)
                
        except Exception as e:
            logger.warning(f" Error reading {file_path}: {e}")
    
    logger.info(f" Loaded {len(documents)} text files from {directory_path}")
    if documents:
        total_words = sum(len(doc.split()) for doc in documents)
        avg_words = total_words / len(documents)
        logger.info(f" Average document length: {avg_words:.1f} words")
    
    return documents, doc_ids

# Example usage - load and add your documents
# Uncomment the lines below to test with your documents
"""
try:
    # Load your extracted text files
    my_documents, my_doc_ids = load_extracted_text_files("extracted_texts")
    
    if my_documents:
        # Add to RAG system with word-based chunking
        success = add_documents_to_system(my_documents, my_doc_ids)
        
        if success:
            logger.info(" Ready to chat with your documents!")
        else:
            logger.error(" Setup failed")
    else:
        logger.warning(" No documents found. Please check the 'extracted_texts' directory.")
        
except Exception as e:
    logger.error(f" Error: {e}")
"""


def rag_chat(query: str) -> Dict:
    """Main RAG chat function for Django integration"""
    try:
        # Use hybrid retrieval (best performing method)
        result = rag_chat_flexible(query)
        return result
    except Exception as e:
        logger.error(f" Error in rag_chat: {e}")
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "doc_sources": [],
            "evaluation": {"overall_score": 0},
            "num_docs_retrieved": 0
        }

def ingest_new_text_file(text_file_path: str, doc_id: str = None) -> str:
    """Ingest a single text file into the RAG system"""
    try:
        # Read the text file
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = Path(text_file_path).stem
        
        # Add to RAG system
        success = add_documents_to_system([text_content], [doc_id])
        
        if success:
            logger.info(f" Successfully ingested: {doc_id}")
            return doc_id
        else:
            logger.error(f" Failed to ingest: {doc_id}")
            return None
            
    except Exception as e:
        logger.error(f" Error ingesting file {text_file_path}: {e}")
        return None

def ask_question(question: str):
    """Simple interface to ask questions about your aviation documents"""
    logger.info(f"Question: {question}")
    logger.info("="*60)
    
    # Use hybrid retrieval (best performing method)
    result = rag_chat_flexible(question)
    
    # Display results
    logger.info(f"Answer:")
    logger.info(f"   {result['answer']}")
    logger.info(f"Source Documents: {', '.join(result['doc_sources'])}")
    logger.info(f"Confidence Score: {result['evaluation'].get('overall_score', 0):.3f}")
    logger.info(f"Retrieved {result['num_docs_retrieved']} relevant chunks")
    
    return None

# Test functions (uncomment to test)
# ask_question("What were the delays taken over two days coded?")
# ask_question("What GPS issues occurred?")
# ask_question("What maintenance was required?")