# integration/vector_db_manager.py
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions # For default embedding function
import uuid

class VectorDBManager:
    """
    Manages interactions with the ChromaDB vector database.
    Handles connection, collection creation, data addition, and querying.
    """
    DEFAULT_COLLECTION_NAME = "aigc_governance_policies"

    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = DEFAULT_COLLECTION_NAME):
        """
        Initializes the ChromaDB client and connects to the specified instance.

        Args:
            host: Hostname or IP address of the ChromaDB server.
            port: Port number of the ChromaDB server.
            collection_name: Name of the default collection to use/create.
        """
        try:
            # Using HttpClient to connect to a running Chroma server instance
            self.client = chromadb.HttpClient(host=host, port=port, settings=Settings(allow_reset=True))
            # Optional: Reset ChromaDB state (useful for development, remove for production)
            # self.client.reset()
            print(f"Connected to ChromaDB at {host}:{port}")

            # Use the default Sentence Transformer embedding function (downloads model on first use)
            # You can replace this with other embedding functions (OpenAI, Cohere, etc.) if needed
            # See: https://docs.trychroma.com/embeddings
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

            self.collection_name = collection_name
            # Get or create the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"} # Use cosine distance for similarity
            )
            print(f"Using ChromaDB collection: '{self.collection_name}'")

        except Exception as e:
            print(f"Error connecting to ChromaDB or getting collection: {e}")
            print("Ensure the ChromaDB Docker container is running and accessible.")
            self.client = None
            self.collection = None
            # Depending on requirements, you might want to raise the exception
            # raise

    def is_connected(self) -> bool:
        """Checks if the client is connected to ChromaDB."""
        if not self.client:
            return False
        try:
            # Heartbeat check
            self.client.heartbeat()
            return True
        except Exception:
            return False

    def add_policy_document(self, text: str, doc_id: str | None = None, metadata: dict | None = None):
        """
        Adds a policy document (text) to the ChromaDB collection.

        Args:
            text: The text content of the policy document.
            doc_id: Optional unique ID for the document. If None, a UUID is generated.
            metadata: Optional dictionary of metadata associated with the document.
        """
        if not self.collection:
            print("Error: ChromaDB collection not available.")
            return

        if not doc_id:
            doc_id = str(uuid.uuid4()) # Generate a unique ID if not provided

        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata] if metadata else None,
                ids=[doc_id]
            )
            print(f"Added document '{doc_id}' to collection '{self.collection_name}'.")
        except Exception as e:
            # Handle potential duplicate ID errors or other issues
            print(f"Error adding document '{doc_id}' to ChromaDB: {e}")

    def query_policies(self, query_text: str, n_results: int = 3, where_filter: dict | None = None) -> list | None:
        """
        Queries the collection for documents similar to the query text.

        Args:
            query_text: The text to search for similar documents.
            n_results: The maximum number of results to return.
            where_filter: Optional filter dictionary (e.g., {"license_type": "standard"}).
                          See ChromaDB docs for filter syntax.

        Returns:
            A list of query results, or None if an error occurs or not connected.
            Each result typically includes ids, distances, metadatas, documents.
        """
        if not self.collection:
            print("Error: ChromaDB collection not available.")
            return None

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter,
                include=['metadatas', 'documents', 'distances'] # Specify what data to include
            )
            print(f"ChromaDB query results for '{query_text}': {results}")
            return results
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return None

# Example usage (for testing this file directly)
if __name__ == '__main__':
    print("Testing VectorDBManager...")
    manager = VectorDBManager() # Connects to localhost:8000 by default

    if manager.is_connected():
        print("\n--- Adding Documents ---")
        manager.add_policy_document(
            doc_id="policy_cc_by",
            text="Creative Commons Attribution 4.0 International License (CC BY 4.0). Allows sharing and adaptation with attribution.",
            metadata={"license_type": "cc-by", "version": "4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}
        )
        manager.add_policy_document(
            doc_id="policy_cc_by_sa",
            text="Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0). Allows sharing and adaptation with attribution and requires adaptations to be shared under the same license.",
            metadata={"license_type": "cc-by-sa", "version": "4.0", "url": "https://creativecommons.org/licenses/by-sa/4.0/"}
        )
        manager.add_policy_document(
            doc_id="policy_private",
            text="Private Use License. Content is for personal use only and cannot be redistributed or modified.",
            metadata={"license_type": "private", "version": "1.0"}
        )

        print("\n--- Querying Documents ---")
        query = "What license allows modification but requires sharing adaptations?"
        results = manager.query_policies(query, n_results=2)

        if results:
             # Process results (example)
             # Results format is like: {'ids': [['id1', 'id2']], 'distances': [[d1, d2]], 'metadatas': [[m1, m2]], 'documents': [[doc1, doc2]]}
             if results.get('ids') and results['ids'][0]:
                 print("\nTop matching documents:")
                 for i, doc_id in enumerate(results['ids'][0]):
                     distance = results['distances'][0][i]
                     doc_text = results['documents'][0][i]
                     metadata = results['metadatas'][0][i]
                     print(f"  ID: {doc_id}, Distance: {distance:.4f}")
                     print(f"  Metadata: {metadata}")
                     print(f"  Text: {doc_text[:100]}...") # Print first 100 chars
             else:
                 print("Query returned no matching documents.")


        print("\n--- Querying with Filter ---")
        query_filter = "private use only"
        filter_dict = {"license_type": "private"}
        results_filtered = manager.query_policies(query_filter, n_results=1, where_filter=filter_dict)
        if results_filtered and results_filtered.get('ids') and results_filtered['ids'][0]:
             print(f"\nFiltered result for '{query_filter}' (license_type=private):")
             print(f"  ID: {results_filtered['ids'][0][0]}")
             print(f"  Document: {results_filtered['documents'][0][0]}")
        else:
             print("Filtered query returned no matching documents.")

    else:
        print("Could not connect to ChromaDB. Aborting tests.")

