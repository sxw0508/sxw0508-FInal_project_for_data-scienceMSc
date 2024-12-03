"""
depth search graph Module
"""
from typing import Optional
import logging
from pydantic import BaseModel
from .base_graph import BaseGraph
from .abstract_graph import AbstractGraph
from ..nodes import (
    FetchNodeLevelK,
    ParseNodeDepthK,
    DescriptionNode,
    RAGNode,
    GenerateAnswerNodeKLevel
)

class DepthSearchGraph(AbstractGraph):
    def __init__(self, prompt: str, source: str, config: dict, schema: Optional[BaseModel] = None):
        super().__init__(prompt, config, source, schema)
        self.input_key = "url" if source.startswith("http") else "local_dir"

    def _create_graph(self) -> BaseGraph:
        """
        Creates the graph of nodes representing the workflow for web scraping.

        Returns:
            BaseGraph: A graph instance representing the web scraping workflow.
        """

        # Fetch Node: Downloads the web page or loads from local directory
        fetch_node_k = FetchNodeLevelK(
            input="url| local_dir",
            output=["docs"],
            node_config={
                "loader_kwargs": self.config.get("loader_kwargs", {}),
                "force": self.config.get("force", False),
                "cut": self.config.get("cut", True),
                "browser_base": self.config.get("browser_base"),
                "depth": self.config.get("depth", 1),
                "only_inside_links": self.config.get("only_inside_links", False),
                "llm_model": self.llm_model,
            }
        )

        # Parse Node: Processes the fetched HTML documents
        parse_node_k = ParseNodeDepthK(
            input="docs",
            output=["docs"],
            node_config={
                "verbose": self.config.get("verbose", False)
            }
        )

        # Description Node: Adds a summary to the fetched and parsed documents
        description_node = DescriptionNode(
            input="docs",
            output=["parsed_doc"],  # Ensure that it outputs 'parsed_doc'
            node_config={
                "llm_model": self.llm_model,
                "verbose": self.config.get("verbose", False),
                "cache_path": self.config.get("cache_path", False)
            }
        )

        # RAG Node: Uses the parsed documents to generate embeddings
        rag_node = RAGNode(
            input="parsed_doc",  # Make sure it gets 'parsed_doc' from the DescriptionNode
            output=["vectorial_db"],
            node_config={
                "llm_model": self.llm_model,
                "embedder_model": self.config.get("embedder_model", False),
                "verbose": self.config.get("verbose", False),
            }
        )

        # Answer Generation Node: Generates the final answer based on embeddings
        generate_answer_k = GenerateAnswerNodeKLevel(
            input="vectorial_db",
            output=["answer"],
            node_config={
                "llm_model": self.llm_model,
                "embedder_model": self.config.get("embedder_model", False),
                "verbose": self.config.get("verbose", False),
            }
        )

        # Return the graph structure with appropriate nodes and edges
        return BaseGraph(
            nodes=[
                fetch_node_k,
                parse_node_k,
                description_node,
                rag_node,
                generate_answer_k
            ],
            edges=[
                (fetch_node_k, parse_node_k),
                (parse_node_k, description_node),
                (description_node, rag_node),
                (rag_node, generate_answer_k)
            ],
            entry_point=fetch_node_k,
            graph_name=self.__class__.__name__
        )

    def run(self) -> str:
        """
        Executes the scraping process and returns the generated code.

        Returns:
            str: The generated code.
        """

        # Initialize inputs and run the graph
        inputs = {"user_prompt": self.prompt, self.input_key: self.source}
        self.final_state, self.execution_info = self.graph.execute(inputs)

        # Get the result
        docs = self.final_state.get("answer", "No answer")

        return docs

