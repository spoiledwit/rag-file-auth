import os
import logging
import json
import time
import requests
import re
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for RunPod vLLM API only
CONFIG = {
    # RunPod vLLM Serverless API configuration
    "runpod_api_url": "https://api.runpod.ai/v2/itrscoi6yr4h5f/run",
    "runpod_api_key": None,  # Will be set from environment variable
}

# Set RunPod API key from environment variable
CONFIG["runpod_api_key"] = os.getenv('RUNPOD_API_KEY')
if not CONFIG["runpod_api_key"]:
    logger.warning("RUNPOD_API_KEY environment variable not set. RunPod API calls will fail.")

logger.info(f"Simplified RAG utilities loaded:")
logger.info(f"  Generation: RunPod vLLM API ({CONFIG['runpod_api_url']})")


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


def generate_response_runpod(query: str, context_docs: List[Dict], max_tokens: int = 800) -> str:
    """Generate response using RunPod vLLM serverless API"""
    
    if not context_docs:
        return "I couldn't find relevant information to answer your question."
    
    if not CONFIG["runpod_api_key"]:
        return "RunPod API key not configured. Please set RUNPOD_API_KEY environment variable."
    
    start_time = time.time()
    
    # Prepare context with intelligent selection (top-10 from potentially more chunks)
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
- Use ONLY flat key-value pairs - DO NOT create nested objects or nested structures
- All values should be strings, numbers, or null - NO nested objects or arrays
- Include key-value pairs that are relevant to the question
- Extract specific facts, dates, numbers, names, and other important details
- Use clear, descriptive keys (e.g., "Contact_Name", "Phone_Number", "Date", "Flight_Number")
- If information is not available in the context, indicate this clearly
- Be accurate and only include information that is explicitly stated or can be directly inferred

Response (JSON format only with flat key-value pairs):"""
    
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
    """Simple evaluation without embedding models"""
    
    # Simple metrics
    response_length = len(response.split())
    context_used = len(context_docs)
    
    # Check if response contains key terms from query
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    word_overlap = len(query_words.intersection(response_words)) / len(query_words) if query_words else 0
    
    # Calculate average context similarity score from Qdrant results
    context_similarity = 0.0
    if context_docs:
        scores = [doc.get('score', 0.0) for doc in context_docs if 'score' in doc]
        context_similarity = sum(scores) / len(scores) if scores else 0.0
    
    # Simple overall score calculation
    overall_score = (word_overlap + context_similarity) / 2
    
    return {
        "query": query,
        "response": response,
        "response_length": response_length,
        "context_docs_used": context_used,
        "word_overlap_score": float(word_overlap),
        "context_similarity": float(context_similarity),
        "overall_score": float(overall_score)
    }


logger.info("Simplified RAG utilities module loaded (RunPod generation only)")