"""
GenerateAnswerNodeKLevel Module
"""
from typing import List, Optional
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_aws import ChatBedrock
from ..utils.output_parser import get_structured_output_parser, get_pydantic_output_parser
from .base_node import BaseNode
from ..prompts import (
    TEMPLATE_CHUNKS, TEMPLATE_NO_CHUNKS, TEMPLATE_MERGE,
    TEMPLATE_CHUNKS_MD, TEMPLATE_NO_CHUNKS_MD, TEMPLATE_MERGE_MD
)

class GenerateAnswerNodeKLevel(BaseNode):
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
        node_name (str): The unique identifier name for the node, defaulting to "Parse".
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "GANLK",
    ):
        super().__init__(node_name, "node", input, output, 2, node_config)

        self.llm_model = node_config["llm_model"]
        self.embedder_model = node_config.get("embedder_model", None)
        self.verbose = node_config.get("verbose", False)
        self.force = node_config.get("force", False)
        self.script_creator = node_config.get("script_creator", False)
        self.is_md_scraper = node_config.get("is_md_scraper", False)
        self.additional_info = node_config.get("additional_info", "")

    def execute(self, state: dict) -> dict:
        """
        Main method to execute the node. Retrieves user prompt, queries the vectorial
        database for relevant documents, processes them, and generates a final answer.
        """

        self.logger.info(f"--- Executing {self.node_name} Node ---")

        # Retrieve user prompt from state
        user_prompt = state.get("user_prompt")
        if not user_prompt:
            raise ValueError("User prompt is missing in the state.")

        # Set up output parser and format instructions based on schema
        output_parser, format_instructions = self._prepare_output_parser(state)

        # Select appropriate template based on model and task type
        template_no_chunks_prompt, template_chunks_prompt, template_merge_prompt = self._select_templates()

        # Add additional information to prompts if available
        if self.additional_info:
            template_no_chunks_prompt = self.additional_info + template_no_chunks_prompt
            template_chunks_prompt = self.additional_info + template_chunks_prompt
            template_merge_prompt = self.additional_info + template_merge_prompt

        # Query vectorial database
        client = state["vectorial_db"]
        answer_db = self._query_vector_database(client, state, user_prompt)

        # Process relevant chunks and prepare chains for parallel execution
        chains_dict = self._process_chunks(answer_db, user_prompt, template_chunks_prompt)

        # Run the parallel execution of chains
        async_runner = RunnableParallel(**chains_dict)
        batch_results = async_runner.invoke({"format_instructions": user_prompt})

        # Merge results and generate final answer
        answer = self._merge_results(batch_results, user_prompt, template_merge_prompt, format_instructions, output_parser)

        # Store answer in the state
        state["answer"] = answer

        return state

    def _prepare_output_parser(self, state: dict):
        """
        Prepares the appropriate output parser and format instructions
        based on the model configuration and schema in the state.
        """
        if self.node_config.get("schema"):
            schema = self.node_config["schema"]
            if isinstance(self.llm_model, (ChatOpenAI, ChatMistralAI)):
                self.llm_model = self.llm_model.with_structured_output(schema=schema)
                output_parser = get_structured_output_parser(schema)
                format_instructions = "NA"
            else:
                output_parser = get_pydantic_output_parser(schema)
                format_instructions = output_parser.get_format_instructions()
        else:
            if isinstance(self.llm_model, ChatBedrock):
                output_parser = None
                format_instructions = ""
            else:
                output_parser = JsonOutputParser()
                format_instructions = output_parser.get_format_instructions()

        return output_parser, format_instructions

    def _select_templates(self):
        """
        Selects appropriate templates for chunk processing and merging based on the configuration.
        """
        if isinstance(self.llm_model, (ChatOpenAI, AzureChatOpenAI)) and not self.script_creator or self.force and not self.script_creator or self.is_md_scraper:
            template_no_chunks_prompt = TEMPLATE_NO_CHUNKS_MD
            template_chunks_prompt = TEMPLATE_CHUNKS_MD
            template_merge_prompt = TEMPLATE_MERGE_MD
        else:
            template_no_chunks_prompt = TEMPLATE_NO_CHUNKS
            template_chunks_prompt = TEMPLATE_CHUNKS
            template_merge_prompt = TEMPLATE_MERGE

        return template_no_chunks_prompt, template_chunks_prompt, template_merge_prompt

    def _query_vector_database(self, client, state, user_prompt):
        """
        Queries the vectorial database based on the user prompt and embeddings if available.
        """
        if state.get("embeddings"):
            import openai
            openai_client = openai.Client()

            # Search query using embeddings
            embedding_vector = openai_client.embeddings.create(
                input=[user_prompt],
                model=state.get("embeddings").get("model")
            ).data[0].embedding

            # Ensure embedding size is 1536
            if len(embedding_vector) != 1536:
                raise ValueError(f"Embedding vector has size {len(embedding_vector)}, but collection expects size 1536.")

            return client.search(
                collection_name="vectorial_collection",
                query_vector=embedding_vector,
            )
        else:
            # Text query without embeddings
            return client.query(
                collection_name="vectorial_collection",
                query_text=user_prompt
            )

    def _process_chunks(self, answer_db, user_prompt, template_chunks_prompt):
        """
        Processes the retrieved chunks, prepares the chains for parallel execution.
        """
        chains_dict = {}
        elems = [state.get("docs")[elem.id - 1] for elem in answer_db if elem.score > 0.5]

        for i, chunk in enumerate(tqdm(elems, desc="Processing chunks", disable=not self.verbose)):
            prompt = PromptTemplate(
                template=template_chunks_prompt,
                input_variables=["format_instructions"],
                partial_variables={"context": chunk.get("document"), "chunk_id": i + 1},
            )
            chain_name = f"chunk{i + 1}"
            chains_dict[chain_name] = prompt | self.llm_model

        return chains_dict

    def _merge_results(self, batch_results, user_prompt, template_merge_prompt, format_instructions, output_parser):
        """
        Merges the results from the chunked documents and generates the final answer.
        """
        merge_prompt = PromptTemplate(
            template=template_merge_prompt,
            input_variables=["context", "question"],
            partial_variables={"format_instructions": format_instructions},
        )

        merge_chain = merge_prompt | self.llm_model
        if output_parser:
            merge_chain = merge_chain | output_parser

        return merge_chain.invoke({"context": batch_results, "question": user_prompt})

