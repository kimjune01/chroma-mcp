from typing import Dict, List, Optional
import chromadb
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
import argparse
from chromadb.config import Settings
import ssl
import datetime
import re

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
    limit: Optional[int] = 10,
    offset: Optional[int] = 0
) -> List[str]:
    """List all memory names in the database with pagination support. This is useful for discovering available memory stores for further queries."""
    _validate_positive_int(limit, "limit")
    _validate_positive_int(offset, "offset")
    client = get_chroma_client()
    try:
        colls = client.list_collections(limit=limit, offset=offset)
        return [coll.name for coll in colls]
    except Exception as e:
        raise Exception(f"Failed to list memories: {str(e)}") from e

@mcp.tool()
async def get_memory_info(memory_name: str) -> Dict:
    """Get information about a memory. Use this to inspect the size and sample contents of a specific memory before querying in detail."""
    _validate_non_empty_str(memory_name, "memory_name")
    client = get_chroma_client()
    try:
        collection = client.get_collection(memory_name)
        count = collection.count()
        peek_results = collection.peek(limit=3)
        return {
            "name": memory_name,
            "count": count,
            "sample_documents": peek_results
        }
    except Exception as e:
        raise Exception(f"Failed to get memory info for '{memory_name}': {str(e)}") from e

@mcp.tool()
async def query_memories(
    memory_name: str,
    query_texts: List[str],
    n_results: int = 10,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    include: List[str] = ["documents", "metadatas", "distances"]
) -> Dict:
    """Query documents from a memory with advanced filtering. This tool is ideal for retrieving relevant documents from a specific memory based on query text and optional filters. Querying by text performs a semantic search using vector embeddings."""
    _validate_non_empty_str(memory_name, "memory_name")
    _validate_non_empty_list(query_texts, "query_texts")
    _validate_positive_int(n_results, "n_results")
    client = get_chroma_client()
    try:
        collection = client.get_collection(memory_name)
        return collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
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
    """Get documents from a memory with optional filtering. Use this to retrieve specific documents or subsets of documents from a memory by ID or filter criteria."""
    _validate_non_empty_str(memory_name, "memory_name")
    if ids is not None:
        _validate_non_empty_list(ids, "ids")
    _validate_positive_int(limit, "limit")
    _validate_positive_int(offset, "offset")
    client = get_chroma_client()
    try:
        collection = client.get_collection(memory_name)
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
    Query documents from multiple memories within a date range. This is useful for retrieving time-bounded data across several memories, such as logs or events within a specific period.
    """
    _validate_non_empty_list(memory_names, "memory_names")
    _validate_iso8601(start_date, "start_date")
    _validate_iso8601(end_date, "end_date")
    _validate_positive_int(limit, "limit")
    _validate_positive_int(offset, "offset")
    client = get_chroma_client()
    results = {}
    # Convert ISO 8601 to epoch seconds
    def iso_to_epoch(s):
        dt = datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
        return dt.timestamp()
    start_epoch = iso_to_epoch(start_date)
    end_epoch = iso_to_epoch(end_date)
    where = {"created_at": {"$gte": start_epoch, "$lte": end_epoch}}
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
    include: List[str] = ["documents", "metadatas", "distances"]
) -> Dict[str, Dict]:
    """
    Query multiple memories with the same query texts. Use this to perform parallel or comparative searches across several memories at once. Querying by text performs a semantic search using vector embeddings.
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
    """Recall recent conversations and chats by retrieving them from the database.
    
    Args:
        mcp_client_name: Name of the MCP Client. e.g. "claude-desktop"
        n_results: Number of recent conversations to recall.
        start_time: (Optional) Only recall conversations after this ISO 8601 time.
        end_time: (Optional) Only recall conversations before this ISO 8601 time.
    
    If no time range is specified, the tool will automatically expand the search window: it will first try since yesterday, then since a week ago, then a month ago, then a year ago, until results are found or all windows are exhausted.
    
    Returns:
        A dict of recalled conversations, sorted by most recent first (may be empty).
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
            where = {}
            if start_epoch is not None and end_epoch is not None:
                where["created_at"] = {"$gte": start_epoch, "$lte": end_epoch}
            elif start_epoch is not None:
                where["created_at"] = {"$gte": start_epoch}
            elif end_epoch is not None:
                where["created_at"] = {"$lte": end_epoch}
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
