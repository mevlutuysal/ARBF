# In integration/vector_db_manager.py

import sys
import requests

# --- DEPENDENCY CHECK ---
try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("ERROR: ChromaDB library not found.")
    print("Please install it by running: pip install chromadb")
    sys.exit(1)

try:
    import sentence_transformers
except ImportError:
    print("ERROR: sentence-transformers library not found.")
    print("Please install it by running: pip install sentence-transformers")
    sys.exit(1)


# --- END DEPENDENCY CHECK ---


class VectorDBManager:
    """
    Manages a robust connection and interaction with a ChromaDB instance.
    Includes enhanced logging and error handling.
    """

    def __init__(self,
                 collection_name: str,
                 host: str = "localhost",
                 port: int = 8000,
                 embed_model: str = "all-mpnet-base-v2"):
        """
        Initializes the ChromaDB client and sets up the embedding function.
        Note: This no longer creates the collection, which must be done explicitly.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None
        self.collection = None # Collection will be set later
        self.embedding_function = None

        print("--- Initializing VectorDBManager ---")

        try:
            # 1. Define the embedding model.
            model_name = "all-mpnet-base-v2"
            print(f"Attempting to initialize embedding function with model: {model_name}...")
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
            print("Embedding function initialized successfully.")

            # 2. Connect to the ChromaDB service.
            print(f"Attempting to connect to ChromaDB at http://{self.host}:{self.port}...")
            self.client = chromadb.HttpClient(host=self.host, port=self.port)

            # 3. Verify the connection is alive.
            self.client.heartbeat()
            print("ChromaDB connection successful.")

        except requests.exceptions.ConnectionError as e:
            print("\nFATAL ERROR: Could not connect to ChromaDB.")
            print(
                f"Please ensure the ChromaDB Docker container is running and accessible at http://{self.host}:{self.port}")
            print(f"Details: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\nFATAL ERROR: An unexpected error occurred during VectorDBManager initialization: {e}")
            sys.exit(1)

        print("--- VectorDBManager client initialized successfully (collection not yet created) ---")


    def add_policy_document(self, text: str, doc_id: str, metadata: dict):
        """Adds a single document and its metadata to the collection."""
        if not self.collection:
            print("ERROR: Collection is not available. Cannot add document.")
            return
        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
        except Exception as e:
            print(f"ERROR: Could not add document with id '{doc_id}'. Details: {e}")

    def query_policies(self, query_text: str, n_results: int = 2):
        """Queries the collection for relevant policy documents."""
        if not self.collection:
            print("ERROR: Collection is not available. Cannot perform query.")
            return None

        print(f"\n[VectorDB] Performing query for: '{query_text}'")
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["metadatas", "documents", "distances"]
            )
            print(f"[VectorDB] Query successful. Found {len(results.get('ids', [[]])[0])} results.")
            # print(f"[VectorDB] Raw results: {results}") # Optional: uncomment for debugging
            return results
        except Exception as e:
            print(f"ERROR: An error occurred during the query. Details: {e}")
            return None