# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:54:52 2024

@author: Xuewen Sun
"""

from typing import Optional
from pydantic import BaseModel
from .base_graph import BaseGraph
from .abstract_graph import AbstractGraph
from ..nodes import (
    FetchNodeLevelK,
    ParseNode,
    RAGNode,
    GenerateAnswerNode,
    ConditionalNode,
)
from ..prompts import REGEN_ADDITIONAL_INFO


class CustomGraph(AbstractGraph):
    """
    SmartScraperGraph is a scraping pipeline that includes FetchNodeLevelK, parsing,
    RAG-based chunk generation, and generating answers using a language model.

    Attributes:
        prompt (str): The prompt for the graph.
        source (str): The source of the graph.
        config (dict): Configuration parameters for the graph.
        schema (BaseModel): The schema for the graph output.
        llm_model: An instance of a language model client.
        embedder_model: An instance of an embedding model client.
        verbose (bool): Whether to show print statements during execution.
        headless (bool): Whether to run in headless mode.
    """

    def __init__(self, prompt: str, source: str, config: dict, schema: Optional[BaseModel] = None):
        """
        Initializes the SmartScraperGraph.

        Args:
            prompt (str): The user's prompt.
            source (str): The source URL or local directory.
            config (dict): Configuration dictionary.
            schema (Optional[BaseModel]): The schema for the output.
        """
        super().__init__(prompt, config, source, schema)
        self.input_key = "url" if source.startswith("http") else "local_dir"

    def _create_graph(self) -> BaseGraph:
        """
        Creates the scraping graph with FetchNodeLevelK, parse, RAG, and answer generation.

        Returns:
            BaseGraph: The constructed graph.
        """

        # Fetch Node Level K
        fetch_node = FetchNodeLevelK(
            input="url| local_dir",
            output=["doc"],
            node_config={
                "llm_model": self.llm_model,
                "depth": self.config.get("depth", 1),
                "only_inside_links": self.config.get("only_inside_links", False),
                "verbose": self.config.get("verbose", False),
                "headless": self.config.get("headless", True),
                "loader_kwargs": self.config.get("loader_kwargs", {}),
                "browser_base": self.config.get("browser_base"),
                "scrape_do": self.config.get("scrape_do"),
                "user_prompt":self.prompt
            },
        )

        # Parse Node
        parse_node = ParseNode(
            input="doc",
            output=["parsed_doc"],
            node_config={
                "llm_model": self.llm_model,
                "chunk_size": self.config.get("chunk_size", 5000),
                "verbose": self.config.get("verbose", False),
            },
        )

        # RAG Node
        rag_node = RAGNode(
            input="parsed_doc",
            output=["relevant_chunks"],
            node_config={
                "llm_model": self.llm_model,
                "embedder_model": self.config.get("embedder_model"),
                "client_type": self.config.get("vector_db", "memory"),
                "verbose": self.config.get("verbose", False),
            },
        )

        # Generate Answer Node
        generate_answer_node = GenerateAnswerNode(
            input="user_prompt & relevant_chunks",
            output=["answer"],
            node_config={
                "llm_model": self.llm_model,
                "schema": self.schema,
                "verbose": self.config.get("verbose", False),
            },
        )

        # Optional Conditional and Retry Nodes
        cond_node = None
        regen_node = None
        if self.config.get("reattempt") is True:
            cond_node = ConditionalNode(
                input="answer",
                output=["answer"],
                node_name="ConditionalNode",
                node_config={
                    "key_name": "answer",
                    "condition": 'not answer or answer=="NA"',
                },
            )
            regen_node = GenerateAnswerNode(
                input="user_prompt & answer",
                output=["answer"],
                node_config={
                    "llm_model": self.llm_model,
                    "additional_info": REGEN_ADDITIONAL_INFO,
                    "schema": self.schema,
                },
            )

        # Graph Variations
        graph_variation_config = {
            (False, False): {
                "nodes": [fetch_node, parse_node, rag_node, generate_answer_node],
                "edges": [
                    (fetch_node, parse_node),
                    (parse_node, rag_node),
                    (rag_node, generate_answer_node),
                ],
            },
            (False, True): {
                "nodes": [fetch_node, parse_node, rag_node, generate_answer_node, cond_node, regen_node],
                "edges": [
                    (fetch_node, parse_node),
                    (parse_node, rag_node),
                    (rag_node, generate_answer_node),
                    (generate_answer_node, cond_node),
                    (cond_node, regen_node),
                    (cond_node, None),
                ],
            },
        }

        reattempt = self.config.get("reattempt", False)
        config = graph_variation_config.get((False, reattempt))

        if config:
            return BaseGraph(
                nodes=config["nodes"],
                edges=config["edges"],
                entry_point=fetch_node,
                graph_name=self.__class__.__name__,
            )

        # Default graph
        return BaseGraph(
            nodes=[fetch_node, parse_node, rag_node, generate_answer_node],
            edges=[
                (fetch_node, parse_node),
                (parse_node, rag_node),
                (rag_node, generate_answer_node),
            ],
            entry_point=fetch_node,
            graph_name=self.__class__.__name__,
        )

    def run(self) -> str:
        """
        Executes the graph and returns the generated answer.

        Returns:
            str: The generated answer or a fallback message.
        """
        inputs = {"user_prompt": self.prompt, self.input_key: self.source}
        self.final_state, self.execution_info = self.graph.execute(inputs)
        return self.final_state.get("answer", "No answer found.")



