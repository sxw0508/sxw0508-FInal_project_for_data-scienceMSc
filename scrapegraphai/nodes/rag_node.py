"""
RAGNode Module
"""
from typing import List, Optional
from .base_node import BaseNode
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

class RAGNode(BaseNode):
    """
    A node responsible for compressing the input tokens and storing the document
    in a vector database for retrieval. Relevant chunks are stored in the state.

    It allows scraping of big documents without exceeding the token limit of the language model.

    Attributes:
        llm_model: An instance of a language model client, configured for generating answers.
        verbose (bool): A flag indicating whether to show print statements during execution.

    Args:
        input (str): Boolean expression defining the input keys needed from the state.
        output (List[str]): List of output keys to be updated in the state.
        node_config (dict): Additional configuration for the node.
        node_name (str): The unique identifier name for the node, defaulting to "RAG".
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "RAG",
    ):
        super().__init__(node_name, "node", input, output, 2, node_config)

        self.llm_model = node_config["llm_model"]
        self.embedder_model = node_config.get("embedder_model", None)
        self.verbose = (
            False if node_config is None else node_config.get("verbose", False)
        )

    def execute(self, state: dict) -> dict:
        self.logger.info(f"--- Executing {self.node_name} Node ---")
        
        # Ensure the input state contains the expected key
        parsed_docs = state.get("parsed_doc")
        if not parsed_docs:
            raise ValueError(f"Missing 'parsed_doc' in state for {self.node_name}")
        
        if not parsed_docs:
            self.logger.warning(f"Parsed documents are empty in {self.node_name}")
            return state

        # Initialize Qdrant client
        if self.node_config.get("client_type") in ["memory", None]:
            client = QdrantClient(":memory:")
        elif self.node_config.get("client_type") == "local_db":
            client = QdrantClient(path="path/to/db")
        elif self.node_config.get("client_type") == "image":
            client = QdrantClient(url="http://localhost:6333")
        else:
            raise ValueError("client_type provided is not correct")

        # Prepare documents for vectorization
        docs = parsed_docs
        ids = [i for i in range(1, len(docs) + 1)]

        # Check if collection already exists before creating it
        collection_name = "vectorial_collection"
        try:
            # Get collection metadata to check vector dimension
            collection = client.get_collection(collection_name=collection_name)
            # Check if vector dimension matches the model's output size
            vector_size = collection["vectors"]["size"]
            if vector_size != 1536:  # assuming 1536 is your desired vector dimension
                self.logger.warning(
                    f"Expected vector size 1536, but found {vector_size}. Updating collection..."
                )
                # You might want to handle re-creating or adjusting your collection here
        except Exception:
            # Create the collection if it doesn't exist
            self.logger.info(f"Creating collection '{collection_name}'...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,  # Ensure embedding size matches the model's output
                    distance=Distance.COSINE,
                ),
            )

        # Compute actual embeddings using the embedder model
        if self.embedder_model:
            embeddings = self.embedder_model.embed(docs)
            embedding_size = len(embeddings[0])  # Assuming embedding is a list of lists
            if embedding_size != 1536:
                raise ValueError(
                    f"Embedding model produces vectors of size {embedding_size}, but collection expects size 1536."
                )
        else:
            # If no embedder model, use placeholder embeddings (this is just an example)
            embeddings = [[0.0] * 1536 for _ in docs]

        # Create and upsert points into the vector database
        points = [
            PointStruct(
                id=idx,
                vector=embedding,
                payload={"text": doc},
            )
            for idx, (doc, embedding) in enumerate(zip(docs, embeddings), start=1)
        ]
        client.upsert(collection_name=collection_name, points=points)

        # Update state with the vector database and relevant chunks
        state["vectorial_db"] = client
        state.update({"relevant_chunks": docs})

        self.logger.info(f"Upserted {len(docs)} documents into the vector database.")

        return state


