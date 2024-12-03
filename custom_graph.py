# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:54:52 2024

@author: Xuewen Sun
"""

"""
CustomGraph Module
"""
from typing import Optional
from pydantic import BaseModel
from .base_graph import BaseGraph
from .abstract_graph import AbstractGraph
from ..nodes import (
    FetchNodeLevelK,
    ParseNode,
    ReasoningNode,
    DescriptionNode,
    RAGNode,
    GenerateAnswerNode,
    ConditionalNode,
)
from pydantic import BaseModel

# 定义 schema
class DefaultSchema(BaseModel):
    example_field: str

class CustomGraph(AbstractGraph):
    """
    CustomGraph is a modular graph pipeline that processes tasks such as 
    fetching data, parsing content, summarization, reasoning, and generating answers.
    
    It adapts to different configurations to create a dynamic workflow.
    """

    def __init__(self, prompt: str, source: str, config: dict, schema: Optional[BaseModel] = None):
        super().__init__(prompt, config, source, schema)
        self.input_key = "url" if source.startswith("http") else "local_dir"

    def _create_graph(self) -> BaseGraph:
        """
        Creates the graph of nodes and defines the workflow dynamically 
        based on the configuration.

        Returns:
            BaseGraph: A graph instance representing the workflow.
        """
        # Nodes definition
        fetch_node = FetchNodeLevelK(
            input="url | local_dir",
            output=["doc"],
            node_config={
                "llm_model": self.llm_model,
                "depth": self.config.get("depth", 1),
                "llm": self.config.get("llm"),
            }
        )

        parse_node = ParseNode(
            input="doc",
            output=["parsed_doc"],
            node_config={
                "llm_model": self.llm_model,
                "chunk_size": self.model_token,
            }
        )
        reasoning_node = ReasoningNode(
            input="user_prompt & parsed_doc",
            output=["refined_prompt"],
            node_config={
                "llm_model": self.llm_model,
                "schema": DefaultSchema,  # 传递有效的 schema
                "additional_info": "Provide relevant context."
            }
        )
 

        conditional_node = ConditionalNode(
            input="refined_prompt",
            output=["is_sufficient"],
            node_config={
                "key_name": "is_sufficient",
                "condition": 'not refined_prompt or refined_prompt=="NA"',
            }
        )

        description_node = DescriptionNode(
            input="parsed_doc",
            output=["summaries"],
            node_config={
                "llm_model": self.llm_model,
                "verbose": self.config.get("verbose", False),
            }
        )

        rag_node = RAGNode(
            input="summaries",
            output=["relevant_chunks"],
            node_config={
                "llm_model": self.llm_model,
                "embedder_model": self.config.get("embedder_model"),
                "client_type": self.config.get("vector_db", "memory"),
            }
        )

        generate_answer_node = GenerateAnswerNode(
            input="refined_prompt & relevant_chunks",
            output=["answer"],
            node_config={
                "llm_model": self.llm_model,
                "schema": self.schema,
            }
        )

        # Graph variations
        graph_variation_config = {
            # Standard flow with reasoning and conditional processing
            "default": {
                "nodes": [fetch_node, parse_node, reasoning_node, conditional_node, description_node, rag_node, generate_answer_node],
                "edges": [
                    (fetch_node, parse_node),
                    (parse_node, reasoning_node),
                    (reasoning_node, conditional_node),
                    (conditional_node, description_node),  # If answer is not sufficient
                    (description_node, rag_node),
                    (rag_node, generate_answer_node),
                    (conditional_node, generate_answer_node),  # If answer is sufficient
                ]
            },
            # Simplified flow without reasoning
            "simple": {
                "nodes": [fetch_node, parse_node, generate_answer_node],
                "edges": [
                    (fetch_node, parse_node),
                    (parse_node, generate_answer_node),
                ]
            },
        }

        # Select graph variation based on configuration
        reasoning_enabled = self.config.get("reasoning", True)
        graph_config = graph_variation_config["default"] if reasoning_enabled else graph_variation_config["simple"]

        # Create the graph
        return BaseGraph(
            nodes=graph_config["nodes"],
            edges=graph_config["edges"],
            entry_point=fetch_node,
            graph_name=self.__class__.__name__
        )

    def run(self) -> str:
        """
        Executes the graph and retrieves the answer to the prompt.

        Returns:
            str: The final answer to the user's prompt.
        """
        inputs = {"user_prompt": self.prompt, self.input_key: self.source}
        self.final_state, self.execution_info = self.graph.execute(inputs)
        return self.final_state.get("answer", "No answer found.")

