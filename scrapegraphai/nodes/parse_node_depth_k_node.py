from typing import List, Optional
from langchain_community.document_transformers import Html2TextTransformer
from .base_node import BaseNode
from langchain.schema import Document  # 导入 Document 类

class ParseNodeDepthK(BaseNode):
    """
    A node responsible for parsing HTML content from a series of documents.

    This node enhances the scraping workflow by allowing for targeted extraction of
    content, thereby optimizing the processing of large HTML documents.

    Attributes:
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
        node_name: str = "ParseNodeDepthK",
    ):
        super().__init__(node_name, "node", input, output, 1, node_config)

        self.verbose = (
            False if node_config is None else node_config.get("verbose", False)
        )

    def execute(self, state: dict) -> dict:
        """
        Executes the node's logic to parse the HTML documents content.

        Args:
            state (dict): The current state of the graph. The input keys will be used to fetch the
                            correct data from the state.

        Returns:
            dict: The updated state with the output key containing the parsed content chunks.

        Raises:
            KeyError: If the input keys are not found in the state, indicating that the
                        necessary information for parsing the content is missing.
        """

        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        input_data = [state[key] for key in input_keys]

        documents = input_data[0]

        # Ensure each document is a Document object with a 'page_content' attribute
        for i, doc in enumerate(documents):
            if isinstance(doc, str):  # If doc is a plain string, wrap it into a Document
                documents[i] = Document(page_content=doc, metadata={})

        for doc in documents:
            # Now doc is guaranteed to be a Document object with 'page_content'
            document_md = Html2TextTransformer(ignore_links=True).transform_documents([doc])
            doc.page_content = document_md[0].page_content

        state.update({self.output[0]: documents})
        self.logger.debug(f"State after Fetch: {state}")

        return state

