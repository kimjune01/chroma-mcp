from typing import Dict, List, Optional
import chromadb
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
import argparse
from chromadb.config import Settings
import ssl
import datetime
import time

# Initialize FastMCP server
mcp = FastMCP("history")

# Global variables
_chroma_client = None

def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description='FastMCP server for Chroma DB')
    parser.add_argument('--client-type', 
                       choices=['http', 'cloud', 'persistent', 'ephemeral'],
                       default=os.getenv('CHROMA_CLIENT_TYPE', 'ephemeral'),
                       help='Type of Chroma client to use')
    parser.add_argument('--data-dir',
                       default=os.getenv('CHROMA_DATA_DIR'),
                       help='Directory for persistent client data (only used with persistent client)')
    parser.add_argument('--host', 
                       help='Chroma host (required for http client)', 
                       default=os.getenv('CHROMA_HOST'))
    parser.add_argument('--port', 
                       help='Chroma port (optional for http client)', 
                       default=os.getenv('CHROMA_PORT'))
    parser.add_argument('--custom-auth-credentials',
                       help='Custom auth credentials (optional for http client)', 
                       default=os.getenv('CHROMA_CUSTOM_AUTH_CREDENTIALS'))
    parser.add_argument('--tenant', 
                       help='Chroma tenant (optional for http client)', 
                       default=os.getenv('CHROMA_TENANT'))
    parser.add_argument('--database', 
                       help='Chroma database (required if tenant is provided)', 
                       default=os.getenv('CHROMA_DATABASE'))
    parser.add_argument('--api-key', 
                       help='Chroma API key (required if tenant is provided)', 
                       default=os.getenv('CHROMA_API_KEY'))
    parser.add_argument('--ssl', 
                       help='Use SSL (optional for http client)', 
                       type=lambda x: x.lower() in ['true', 'yes', '1', 't', 'y'],
                       default=os.getenv('CHROMA_SSL', 'true').lower() in ['true', 'yes', '1', 't', 'y'])
    parser.add_argument('--dotenv-path', 
                       help='Path to .env file', 
                       default=os.getenv('CHROMA_DOTENV_PATH', '.chroma_env'))
    return parser

def get_chroma_client(args=None):
    """Get or create the global Chroma client instance."""
    global _chroma_client
    if _chroma_client is None:
        if args is None:
            # Create parser and parse args if not provided
            parser = create_parser()
            args = parser.parse_args()
        # Load environment variables from .env file if it exists
        load_dotenv(dotenv_path=args.dotenv_path)
        if args.client_type == 'http':
            if not args.host:
                raise ValueError("Host must be provided via --host flag or CHROMA_HOST environment variable when using HTTP client")
            settings = Settings()
            if args.custom_auth_credentials:
                settings = Settings(
                    chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
                    chroma_client_auth_credentials=args.custom_auth_credentials
                )
            try:
                _chroma_client = chromadb.HttpClient(
                    host=args.host,
                    port=args.port if args.port else None,
                    ssl=args.ssl,
                    settings=settings
                )
            except ssl.SSLError as e:
                print(f"SSL connection failed: {str(e)}")
                raise
            except Exception as e:
                print(f"Error connecting to HTTP client: {str(e)}")
                raise
        elif args.client_type == 'cloud':
            if not args.tenant:
                raise ValueError("Tenant must be provided via --tenant flag or CHROMA_TENANT environment variable when using cloud client")
            if not args.database:
                raise ValueError("Database must be provided via --database flag or CHROMA_DATABASE environment variable when using cloud client")
            if not args.api_key:
                raise ValueError("API key must be provided via --api-key flag or CHROMA_API_KEY environment variable when using cloud client")
            try:
                _chroma_client = chromadb.HttpClient(
                    host="api.trychroma.com",
                    ssl=True,  # Always use SSL for cloud
                    tenant=args.tenant,
                    database=args.database,
                    headers={
                        'x-chroma-token': args.api_key
                    }
                )
            except ssl.SSLError as e:
                print(f"SSL connection failed: {str(e)}")
                raise
            except Exception as e:
                print(f"Error connecting to cloud client: {str(e)}")
                raise
        elif args.client_type == 'persistent':
            if not args.data_dir:
                raise ValueError("Data directory must be provided via --data-dir flag when using persistent client")
            _chroma_client = chromadb.PersistentClient(path=args.data_dir)
        else:  # ephemeral
            _chroma_client = chromadb.EphemeralClient()
    return _chroma_client

def _validate_non_empty_str(val, name):
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"{name} must be a non-empty string.")

def _validate_non_empty_list(val, name):
    if not isinstance(val, list) or not val or not all(isinstance(x, str) and x.strip() for x in val):
        raise ValueError(f"{name} must be a non-empty list of strings.")

def _validate_positive_int(val, name):
    if val is not None and (not isinstance(val, int) or val < 1):
        raise ValueError(f"{name} must be a positive integer if provided.")

def _validate_iso8601(val, name):
    try:
        datetime.datetime.fromisoformat(val.replace('Z', '+00:00'))
    except Exception:
        raise ValueError(f"{name} must be a valid ISO 8601 date string.")

##### Query and Listing Tools #####

@mcp.tool()
async def list_memories(
    limit: Optional[int] = 100,
    offset: Optional[int] = 0
) -> List[str]:
    """List all memory names in the database with pagination support. Memories are collections of documents. This is useful for discovering available memory stores for further queries."""
    client = get_chroma_client()
    try:
        # Always list all collections, then sort and apply limit/offset
        colls = client.list_collections()
        collection_objs = []
        for coll in colls:
            if hasattr(coll, 'metadata'):
                collection_objs.append(coll)
            else:
                try:
                    collection_objs.append(client.get_collection(coll))
                except Exception:
                    pass
        collection_objs = [c for c in collection_objs if not c.name.startswith("_")]
        def get_updated_at(c):
            meta = getattr(c, 'metadata', {}) or {}
            return int(meta.get('updated_at', 0))
        collection_objs.sort(key=get_updated_at, reverse=True)
        # Apply offset and limit after sorting
        if offset is None:
            offset = 0
        if limit is None:
            limit = len(collection_objs) - offset
        memory_names = [c.name for c in collection_objs[offset:offset+limit]]
        return memory_names
    except Exception as e:
        raise Exception(f"Failed to list memories: {str(e)}") from e

def _update_collection_metadata(collection):
    """Update the collection's metadata with the current timestamp, preserving other metadata."""
    meta = getattr(collection, 'metadata', {}) or {}
    meta['updated_at'] = int(time.time())
    collection.modify(metadata=meta)

@mcp.tool()
async def get_memory_info(memory_name: str) -> Dict:
    """Get information about a memory. Use this to inspect the size and sample contents of a specific memory before querying in detail."""
    _validate_non_empty_str(memory_name, "memory_name")
    client = get_chroma_client()
    try:
        collection = client.get_collection(memory_name)
        count = collection.count()
        peek_results = collection.peek(limit=3)
        # Remove embeddings from peek_results
        if isinstance(peek_results, list):
            cleaned_peek = [
                {k: v for k, v in doc.items() if k != "embeddings"} if isinstance(doc, dict) else doc
                for doc in peek_results
            ]
        elif isinstance(peek_results, dict):
            cleaned_peek = [{k: v for k, v in peek_results.items() if k != "embeddings"}]
        else:
            cleaned_peek = [peek_results]
        return {
            "name": memory_name,
            "count": count,
            "sample_documents": cleaned_peek
        }
    except Exception as e:
        raise Exception(f"Failed to get memory info for '{memory_name}': {str(e)}") from e

@mcp.tool()
async def query_memory(
    memory_name: str,
    query_texts: List[str],
    n_results: int = 10,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    include: List[str] = ["documents", "metadatas"],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict:
    """Query browser history from a memory with advanced filtering. This tool is ideal for retrieving relevant documents from a specific memory based on query text and optional filters. Querying by text performs a semantic search using vector embeddings.
    
    Args:
        memory_name: Name of the memory (collection) to query.
        query_texts: List of query strings to search for semantically relevant documents. Each string is embedded and compared to the stored documents to find the most similar ones. This enables natural language search, not just keyword matching.
            Examples:
                - ["What articles did I read about AI last week?"]
                - ["github.com", "python web scraping"]
                - ["meeting notes from yesterday"]
        n_results: Number of top results to return for each query text.
        where: (Optional) Metadata filter for advanced filtering (e.g., {"source": "wikipedia"}).
        where_document: (Optional) Document content filter for advanced filtering.
        include: List of fields to include in the result (e.g., ["documents", "metadatas"]).
        start_date: (Optional) Only include documents created after this ISO 8601 date string.
        end_date: (Optional) Only include documents created before this ISO 8601 date string.
    
    The query_texts parameter is used to specify the search intent in natural language or keywords. The system will return the most semantically similar documents from the memory for each query text provided.
    If start_date and/or end_date are provided, results will be filtered to only include documents within the specified date range (using the 'created_at' metadata field).
    """
    _validate_non_empty_str(memory_name, "memory_name")
    _validate_non_empty_list(query_texts, "query_texts")
    _validate_positive_int(n_results, "n_results")
    if start_date:
        _validate_iso8601(start_date, "start_date")
    if end_date:
        _validate_iso8601(end_date, "end_date")
    client = get_chroma_client()
    try:
        collection = client.get_collection(memory_name)
        _update_collection_metadata(collection)
        # Build date range filter if needed
        where_combined = where.copy() if where else {}
        if start_date or end_date:
            created_at_filter = {}
            if start_date:
                created_at_filter["$gte"] = datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00')).timestamp()
            if end_date:
                created_at_filter["$lte"] = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00')).timestamp()
            where_combined["created_at"] = created_at_filter
        # Only pass where if it is not empty, otherwise use None
        where_arg = where_combined if where_combined else None
        return collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where_arg,
            where_document=where_document,
            include=include
        )
    except Exception as e:
        raise Exception(f"Failed to query documents from memory '{memory_name}': {str(e)}") from e

@mcp.tool()
async def get_memory_documents(
    memory_name: str,
    ids: Optional[List[str]] = None,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    include: List[str] = ["documents", "metadatas"],
    limit: Optional[int] = 10,
    offset: Optional[int] = 0
) -> Dict:
    """Get browser history documents from a memory with optional filtering. Use this to retrieve specific documents or subsets of browser history from a memory by ID or filter criteria. Note: memory IDs are actually URLs of web pages."""
    _validate_non_empty_str(memory_name, "memory_name")
    if ids is not None:
        _validate_non_empty_list(ids, "ids")
    _validate_positive_int(limit, "limit")
    client = get_chroma_client()
    try:
        collection = client.get_collection(memory_name)
        _update_collection_metadata(collection)
        return collection.get(
            ids=ids,
            where=where,
            where_document=where_document,
            include=include,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        raise Exception(f"Failed to get documents from memory '{memory_name}': {str(e)}") from e

@mcp.tool()
async def query_memories_by_date_range(
    memory_names: List[str],
    start_date: str,
    end_date: str,
    limit: Optional[int] = 10,
    offset: Optional[int] = 0
) -> Dict[str, Dict]:
    """
    Query browser history from multiple memories within a date range. This is useful for retrieving time-bounded data across several memories, such as logs or events within a specific period.
    """
    _validate_non_empty_list(memory_names, "memory_names")
    _validate_iso8601(start_date, "start_date")
    _validate_iso8601(end_date, "end_date")
    _validate_positive_int(limit, "limit")
    client = get_chroma_client()
    results = {}
    # Convert ISO 8601 to epoch seconds
    def iso_to_epoch(s):
        dt = datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
        return dt.timestamp()
    start_epoch = iso_to_epoch(start_date)
    end_epoch = iso_to_epoch(end_date)
    where = {"$and": [
        {"created_at": {"$gte": start_epoch}},
        {"created_at": {"$lte": end_epoch}}
    ]}
    for name in memory_names:
        try:
            collection = client.get_collection(name)
            docs = collection.get(where=where, limit=limit, offset=offset)
            results[name] = docs
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

@mcp.tool()
async def query_memories_with_texts(
    memory_names: List[str],
    query_texts: List[str],
    n_results: int = 10,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    include: List[str] = ["documents", "metadatas"]
) -> Dict[str, Dict]:
    """
    Query browser history from multiple memories with the same query texts. Use this to perform parallel or comparative searches across several memories at once. Querying by text performs a semantic search using vector embeddings.
    """
    _validate_non_empty_list(memory_names, "memory_names")
    _validate_non_empty_list(query_texts, "query_texts")
    _validate_positive_int(n_results, "n_results")
    client = get_chroma_client()
    results = {}
    for name in memory_names:
        try:
            collection = client.get_collection(name)
            res = collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )
            if isinstance(res, dict) and "embeddings" in res:
                res = {k: v for k, v in res.items() if k != "embeddings"}
            results[name] = res
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

@mcp.tool()
async def health_check() -> dict:
    """Check if the server can import chromadb and connect to the database. Useful for monitoring and debugging deployments."""
    result = {"chromadb_imported": False, "db_connection": False}
    try:
        import chromadb
        result["chromadb_imported"] = True
    except Exception as e:
        result["error"] = f"chromadb import failed: {str(e)}"
        return result
    try:
        client = get_chroma_client()
        client.list_collections(limit=1)
        result["db_connection"] = True
    except Exception as e:
        result["error"] = f"DB connection failed: {str(e)}"
    return result

@mcp.tool()
async def remember_this_conversation(
    mcp_client_name: str,
    conversation_content: str,
) -> str:
    """Remember current conversation by storing it in the database so that it can be retrieved later.

    Args:
        mcp_client_name: Name of the MCP Client. e.g. "claude-desktop"
        conversation_content: Content of the conversation, either summarized or raw.
    
    Example queries that should trigger this tool:
        - "Memorize this conversation"
        - "Store our current chat"
        - "Remember this chat"
        - "Save our discussion"
        - "Log this conversation"
        - "Archive this exchange"
        - "Store this session for later"
    """
    client = get_chroma_client()
    try:
        created_at_epoch = datetime.datetime.now().timestamp()
        collection = client.get_or_create_collection(mcp_client_name)
        doc_id =f"{mcp_client_name}_{created_at_epoch}"
        # Store created_at as epoch
        collection.add(
            documents=[conversation_content],
            metadatas=[{"created_at": created_at_epoch}],
            ids=[doc_id]
        )
        return f"Successfully memorized 1 conversation in memory '{mcp_client_name}'"
    except Exception as e:
        raise Exception(f"Failed to memorize conversation in memory '{mcp_client_name}': {str(e)}") from e

@mcp.tool()
async def delete_all_large_entries() -> str:
    """Delete the oldest n_entries from the memory.
    """
    try:
        return remove_large_documents_from_all_collections(30*1024)
    except Exception as e:
        raise Exception(f"Failed to delete all large entries from all memories: {str(e)}") from e

def _sort_and_limit_docs(docs: dict, n_results: int) -> dict:
    if isinstance(docs, dict) and "metadatas" in docs and isinstance(docs["metadatas"], list):
        metadatas = docs["metadatas"]
        if all(isinstance(md, dict) and "created_at" in md for md in metadatas):
            sort_indices = sorted(range(len(metadatas)), key=lambda i: metadatas[i]["created_at"], reverse=True)
            for k in docs:
                if isinstance(docs[k], list) and len(docs[k]) == len(sort_indices):
                    docs[k] = [docs[k][i] for i in sort_indices]
            for k in docs:
                if isinstance(docs[k], list):
                    docs[k] = docs[k][:n_results]
    return docs

@mcp.tool()
async def recall_recent_conversations(
    mcp_client_name: str,
    n_results: int = 1,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> dict:
    """Recall recent browser history and chats by retrieving them from the database.
    
    Args:
        mcp_client_name: Name of the MCP Client. e.g. "claude-desktop"
        n_results: Number of recent conversations to recall.
        start_time: (Optional) Only recall conversations after this ISO 8601 time.
        end_time: (Optional) Only recall conversations before this ISO 8601 time.
    
    If no time range is specified, the tool will automatically expand the search window: it will first try since yesterday, then since a week ago, then a month ago, then a year ago, until results are found or all windows are exhausted.
    
    Returns:
        A dict of recalled browser history, sorted by most recent first (may be empty).
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(mcp_client_name)
        if not collection:
            return {"error": f"No memory found for '{mcp_client_name}'"}
        now = datetime.datetime.now(datetime.timezone.utc)
        # If time range is specified, use it
        if start_time or end_time:
            start_epoch = None
            end_epoch = None
            if start_time:
                _validate_iso8601(start_time, "start_time")
                start_epoch = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00')).timestamp()
            if end_time:
                _validate_iso8601(end_time, "end_time")
                end_epoch = datetime.datetime.fromisoformat(end_time.replace('Z', '+00:00')).timestamp()
            where = {"$and": [
                {"created_at": {"$gte": start_epoch}},
                {"created_at": {"$lte": end_epoch}}
            ]}
            docs = collection.get(where=where)
            return _sort_and_limit_docs(docs, n_results)
        # Exponential backoff: 1 day, 1 week, 1 month, 1 year
        windows = [datetime.timedelta(days=1), datetime.timedelta(weeks=1), datetime.timedelta(days=30), datetime.timedelta(days=365)]
        last_docs = None
        for window in windows:
            start_dt = now - window
            start_epoch = start_dt.timestamp()
            where = {"created_at": {"$gte": start_epoch}}
            docs = collection.get(where=where)
            last_docs = docs
            docs_sorted = _sort_and_limit_docs(docs, n_results)
            if docs_sorted.get("documents"):
                return docs_sorted
        # If all windows are empty, return the last (widest) result, limited
        return _sort_and_limit_docs(last_docs if last_docs is not None else {}, n_results)
    except Exception as e:
        return {"error": f"Failed to recall recent conversations from memory '{mcp_client_name}': {str(e)}"}

def remove_large_documents_from_all_collections(max_size_bytes: int = 100*1024) -> str:
    """
    Go through all collections and remove all documents larger than max_size_bytes.
    """
    client = get_chroma_client()
    collections = client.list_collections()
    count = 0
    for coll in collections:
        if hasattr(coll, 'name'):
            collection = client.get_collection(coll.name)
        else:
            collection = client.get_collection(coll)
        print(f"Checking collection: {collection.name}")
        total_deleted = 0
        offset = 0
        batch_size = 100
        while True:
            batch = collection.get(include=["documents"], limit=batch_size, offset=offset)
            docs = batch.get("documents", [])
            ids = batch.get("ids", [])
            if not docs or not ids:
                break
            to_delete = []
            for doc, doc_id in zip(docs, ids):
                if doc and isinstance(doc, str) and len(doc.encode("utf-8")) > max_size_bytes:
                    to_delete.append(doc_id)
            if to_delete:
                collection.delete(ids=to_delete)
                total_deleted += len(to_delete)
                count += total_deleted
            if len(docs) < batch_size:
                break
            offset += batch_size
    return f"Total deleted from all collections: {count}"

def main():
    """Entry point for the Chroma MCP server."""
    parser = create_parser()
    args = parser.parse_args()
    if args.dotenv_path:
        load_dotenv(dotenv_path=args.dotenv_path)
        parser = create_parser()
        args = parser.parse_args()
    # Validate required arguments based on client type
    if args.client_type == 'http':
        if not args.host:
            parser.error("Host must be provided via --host flag or CHROMA_HOST environment variable when using HTTP client")
    elif args.client_type == 'cloud':
        if not args.tenant:
            parser.error("Tenant must be provided via --tenant flag or CHROMA_TENANT environment variable when using cloud client")
        if not args.database:
            parser.error("Database must be provided via --database flag or CHROMA_DATABASE environment variable when using cloud client")
        if not args.api_key:
            parser.error("API key must be provided via --api-key flag or CHROMA_API_KEY environment variable when using cloud client")
    # Initialize client with parsed args
    try:
        get_chroma_client(args)
        print("Successfully initialized Chroma client")
    except Exception as e:
        print(f"Failed to initialize Chroma client: {str(e)}")
        raise
    # Initialize and run the server
    print("Starting MCP server")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
