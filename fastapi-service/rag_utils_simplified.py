import os
import logging
import time
import warnings
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from datetime import datetime
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress MPS pin_memory warnings on Apple Silicon
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

# Core packages
from sentence_transformers import SentenceTransformer

# Qdrant vector database
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models

# LangChain for text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "embedding_model": "BAAI/bge-large-en",  # 1024-dim embeddings
    
    # Qdrant configuration
    "qdrant_url": os.getenv("QDRANT_URL", "localhost").replace("https://", "http://"),  # Force HTTP
    "qdrant_port": int(os.getenv("QDRANT_PORT", 6333)),
    "qdrant_api_key": os.getenv("QDRANT_API_KEY"),  # Optional, for cloud Qdrant
    "qdrant_collection": "document_embeddings",
    
    # LangChain RecursiveCharacterTextSplitter parameters
    "chunk_size": 500,       # Increased for better context
    "chunk_overlap": 100,    # 20% overlap
    "separators": ["\n\n", "\n", " ", ""],  # Hierarchy of separators
}

logger.info(f"Configuration loaded:")
logger.info(f"  Embedding model: {CONFIG['embedding_model']}")
logger.info(f"  Qdrant: {CONFIG['qdrant_url']}:{CONFIG['qdrant_port']}")
logger.info(f"  Collection: {CONFIG['qdrant_collection']}")

# Global variables for lazy loading
embedding_model = None
text_splitter = None
qdrant_client = None


def initialize_models():
    """Lazy load models and connections only when needed"""
    global embedding_model, text_splitter, qdrant_client
    
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
        logger.info(f"Text splitter initialized")
    
    if qdrant_client is None:
        logger.info("Connecting to Qdrant...")
        try:
            # Check if URL is a full URL (with http/https)
            if CONFIG["qdrant_url"].startswith(('http://', 'https://')):
                # Cloud/Hosted Qdrant with full URL
                qdrant_client = QdrantClient(
                    url=CONFIG["qdrant_url"],
                    api_key=CONFIG["qdrant_api_key"],
                    prefer_grpc=False  # Use REST API for HTTPS
                )
            elif CONFIG["qdrant_api_key"]:
                # Cloud Qdrant with API key (backward compatibility)
                qdrant_client = QdrantClient(
                    url=CONFIG["qdrant_url"],
                    api_key=CONFIG["qdrant_api_key"],
                )
            else:
                # Local Qdrant
                qdrant_client = QdrantClient(
                    host=CONFIG["qdrant_url"],
                    port=CONFIG["qdrant_port"]
                )
            
            # Create collection if it doesn't exist
            create_collection_if_not_exists()
            logger.info("Qdrant client connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise


def create_collection_if_not_exists(collection_name: Optional[str] = None):
    """Create Qdrant collection if it doesn't exist"""
    global qdrant_client, embedding_model
    
    target_collection = collection_name or CONFIG["qdrant_collection"]
    
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if target_collection not in collection_names:
            # Get embedding dimension
            embedding_dimension = embedding_model.get_sentence_embedding_dimension()
            
            # Create collection
            qdrant_client.create_collection(
                collection_name=target_collection,
                vectors_config=VectorParams(
                    size=embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {target_collection}")
        else:
            logger.info(f"Qdrant collection already exists: {target_collection}")
            
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise


def chunk_text(text: str, doc_id: str = None, metadata: Dict = None) -> List[Dict]:
    """
    Chunk text using LangChain RecursiveCharacterTextSplitter
    
    Args:
        text: The text content to chunk
        doc_id: Optional document ID
        metadata: Optional metadata to attach to chunks
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    initialize_models()
    
    if not text:
        return []
    
    if doc_id is None:
        doc_id = str(uuid.uuid4())
    
    if metadata is None:
        metadata = {}
    
    # Create LangChain Document
    doc = Document(
        page_content=text,
        metadata={"doc_id": doc_id, "original_length": len(text), **metadata}
    )
    
    # Split using text splitter
    split_docs = text_splitter.split_documents([doc])
    
    # Convert to our format
    chunks = []
    for chunk_idx, split_doc in enumerate(split_docs):
        chunk_text = split_doc.page_content
        
        chunks.append({
            "id": f"{doc_id}_chunk_{chunk_idx}",
            "text": chunk_text,
            "metadata": {
                "doc_id": doc_id,
                "chunk_idx": chunk_idx,
                "total_chunks": len(split_docs),
                "char_count": len(chunk_text),
                "timestamp": datetime.now().isoformat(),
                **metadata
            }
        })
    
    logger.info(f"Created {len(chunks)} chunks from document {doc_id}")
    return chunks


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        Numpy array of embeddings
    """
    initialize_models()
    
    if not texts:
        return np.array([])
    
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    
    # Generate embeddings in batch
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Important for cosine similarity
    )
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    return embeddings


def store_in_qdrant(chunks: List[Dict], embeddings: np.ndarray, collection_name: Optional[str] = None) -> Dict:
    """
    Store chunks and their embeddings in Qdrant
    
    Args:
        chunks: List of chunk dictionaries
        embeddings: Numpy array of embeddings
        collection_name: Optional collection name (defaults to CONFIG collection)
        
    Returns:
        Dictionary with storage status and statistics
    """
    initialize_models()
    
    if not chunks or len(embeddings) == 0:
        return {"status": "error", "message": "No chunks or embeddings to store"}
    
    if len(chunks) != len(embeddings):
        return {"status": "error", "message": "Chunks and embeddings count mismatch"}
    
    # Use provided collection name or default
    target_collection = collection_name or CONFIG["qdrant_collection"]
    
    # Ensure target collection exists
    try:
        create_collection_if_not_exists(target_collection)
    except Exception as e:
        return {"status": "error", "message": f"Failed to create collection: {str(e)}"}
    
    try:
        # Prepare points for Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),  # Generate unique ID
                vector=embedding.tolist(),
                payload={
                    "text": chunk["text"],
                    "chunk_id": chunk["id"],
                    **chunk["metadata"]
                }
            )
            points.append(point)
        
        # Upload to Qdrant in batches
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=target_collection,
                points=batch
            )
            total_uploaded += len(batch)
            logger.info(f"Uploaded {total_uploaded}/{len(points)} points to {target_collection}")
        
        # Get collection info
        collection_info = qdrant_client.get_collection(target_collection)
        
        return {
            "status": "success",
            "message": f"Successfully stored {len(points)} embeddings",
            "statistics": {
                "chunks_stored": len(chunks),
                "collection_size": collection_info.points_count,
                "collection_name": target_collection
            }
        }
        
    except Exception as e:
        logger.error(f"Error storing in Qdrant: {e}")
        return {
            "status": "error",
            "message": f"Failed to store in Qdrant: {str(e)}"
        }


def process_and_store_document(
    text: str,
    doc_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
    collection_name: Optional[str] = None
) -> Dict:
    """
    Main function to process text and store in Qdrant
    
    Args:
        text: The extracted text from document
        doc_id: Optional document identifier
        metadata: Optional metadata for the document
        collection_name: Optional collection name (defaults to CONFIG collection)
        
    Returns:
        Dictionary with processing results
    """
    try:
        start_time = time.time()
        
        # 1. Chunk the text
        chunks = chunk_text(text, doc_id, metadata)
        if not chunks:
            return {
                "status": "error",
                "message": "No chunks created from document"
            }
        
        # 2. Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = generate_embeddings(texts)
        
        # 3. Store in Qdrant
        storage_result = store_in_qdrant(chunks, embeddings, collection_name)
        
        # 4. Prepare response
        processing_time = time.time() - start_time
        
        return {
            "status": storage_result["status"],
            "message": storage_result["message"],
            "doc_id": doc_id or chunks[0]["metadata"]["doc_id"],
            "processing_time": f"{processing_time:.2f}s",
            "chunks_created": len(chunks),
            "statistics": storage_result.get("statistics", {})
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return {
            "status": "error",
            "message": f"Failed to process document: {str(e)}"
        }


def search_similar(
    query: str,
    limit: int = 10,
    score_threshold: float = 0.7
) -> List[Dict]:
    """
    Search for similar chunks in Qdrant
    
    Args:
        query: Search query text
        limit: Maximum number of results
        score_threshold: Minimum similarity score
        
    Returns:
        List of similar chunks with scores
    """
    initialize_models()
    
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]
        
        # Search in Qdrant
        search_result = qdrant_client.search(
            collection_name=CONFIG["qdrant_collection"],
            query_vector=query_embedding.tolist(),
            limit=limit,
            score_threshold=score_threshold
        )
        
        # Format results
        results = []
        for hit in search_result:
            results.append({
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "chunk_id": hit.payload.get("chunk_id", ""),
                "doc_id": hit.payload.get("doc_id", ""),
                "metadata": {k: v for k, v in hit.payload.items() 
                           if k not in ["text", "chunk_id", "doc_id"]}
            })
        
        logger.info(f"Found {len(results)} similar chunks for query")
        return results
        
    except Exception as e:
        logger.error(f"Error searching in Qdrant: {e}")
        return []


def delete_document(doc_id: str) -> Dict:
    """
    Delete all chunks of a document from Qdrant
    
    Args:
        doc_id: Document ID to delete
        
    Returns:
        Dictionary with deletion status
    """
    initialize_models()
    
    try:
        # Delete points with matching doc_id
        qdrant_client.delete(
            collection_name=CONFIG["qdrant_collection"],
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            )
        )
        
        return {
            "status": "success",
            "message": f"Deleted document {doc_id} from Qdrant"
        }
        
    except Exception as e:
        logger.error(f"Error deleting from Qdrant: {e}")
        return {
            "status": "error",
            "message": f"Failed to delete document: {str(e)}"
        }


def get_collection_stats() -> Dict:
    """Get statistics about the Qdrant collection"""
    initialize_models()
    
    try:
        collection_info = qdrant_client.get_collection(CONFIG["qdrant_collection"])
        
        return {
            "status": "success",
            "collection_name": CONFIG["qdrant_collection"],
            "total_points": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance_metric": collection_info.config.params.vectors.distance
        }
        
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        return {
            "status": "error",
            "message": f"Failed to get stats: {str(e)}"
        }


# Example usage
if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    This is a sample document for testing the simplified RAG system.
    It contains multiple paragraphs to demonstrate text chunking.
    
    The system will chunk this text, generate embeddings, and store
    them in Qdrant for later retrieval.
    """
    
    # Process and store
    result = process_and_store_document(
        text=sample_text,
        doc_id="test_doc_001",
        metadata={"source": "test", "type": "sample"}
    )
    print(f"Storage result: {result}")
    
    # Search for similar content
    search_results = search_similar("sample document", limit=5)
    print(f"Search results: {search_results}")