"""
fetch_node_level_k module
"""
from typing import List, Optional
from urllib.parse import urljoin
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from .base_node import BaseNode
from ..docloaders import ChromiumLoader
from langchain.schema import HumanMessage
from langchain.schema import Document


class FetchNodeLevelK(BaseNode):
    """
    A node for recursively fetching HTML content and sub-links up to a certain depth.
    Uses an LLM to filter links for relevance before processing.
    """

    def __init__(self,
                 input: str,
                 output: List[str],
                 node_config: Optional[dict] = None,
                 node_name: str = "FetchLevelK"):
        """
        Initializes the FetchNodeLevelK instance.
        """
        super().__init__(node_name, "node", input, output, 2, node_config)

        # Configuration parameters
        self.embedder_model = node_config.get("embedder_model", None)
        self.verbose = node_config.get("verbose", False) if node_config else False
        self.cache_path = node_config.get("cache_path", False)
        self.headless = node_config.get("headless", True) if node_config else True
        self.loader_kwargs = node_config.get("loader_kwargs", {}) if node_config else {}
        self.browser_base = node_config.get("browser_base", None)
        self.scrape_do = node_config.get("scrape_do", None)
        self.depth = node_config.get("depth", 1) if node_config else 1
        self.only_inside_links = node_config.get("only_inside_links", False) if node_config else False
        self.min_input_len = 1

        # Load LLM model
        self.llm_model = node_config.get("llm_model", None)
        if self.llm_model is None:
            raise ValueError("The node_config must include an 'llm' parameter for LLM-based link filtering.")

        # User-defined prompt for LLM
        self.user_prompt = node_config.get("user_prompt", "")

    def execute(self, state: dict) -> dict:
        """
        Executes the node's logic, recursively fetching HTML and links, and filtering using LLM.
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")
    
        input_keys = self.get_input_keys(state)
        input_data = [state[key] for key in input_keys]
        source = input_data[0]
    
        documents = [{"source": source}]
        loader_kwargs = self.node_config.get("loader_kwargs", {})
    
        for _ in range(self.depth):
            documents = self.obtain_content(documents, loader_kwargs, self.user_prompt, source)
    
        # 将字典转换为 Document 对象
        document_objects = [
            Document(
                page_content=doc["document"][0].page_content,
                metadata={"source": doc["source"]}
            )
            for doc in documents if "document" in doc
        ]
    
        # 更新 state，确保 ParseNode 可以接收正确的输入
        state.update({self.output[0]: document_objects})
        
        return state

    def fetch_content(self, source: str, loader_kwargs) -> Optional[List[Document]]:
        """
        Fetches the HTML content of a given URL.
        """
        self.logger.info(f"Fetching HTML content from: {source}")

        if self.browser_base:
            from ..docloaders.browser_base import browser_base_fetch
            data = browser_base_fetch(
                self.browser_base.get("api_key"),
                self.browser_base.get("project_id"),
                [source]
            )
            document = [Document(page_content=content, metadata={"source": source}) for content in data]
        elif self.scrape_do:
            from ..docloaders.scrape_do import scrape_do_fetch
            data = scrape_do_fetch(self.scrape_do.get("api_key"), source)
            document = [Document(page_content=data, metadata={"source": source})]
        else:
            loader = ChromiumLoader([source], headless=self.headless, **loader_kwargs)
            document = loader.load()

        return document

    def extract_links(self, html_content: str) -> List[str]:
        """
        Extracts all hyperlinks from the HTML content.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        links = [link['href'] for link in soup.find_all('a', href=True)]
        self.logger.info(f"Extracted {len(links)} links.")
        return links

    def get_full_links(self, base_url: str, links: List[str]) -> List[str]:
        """
        Converts relative URLs to full URLs based on the base URL.
        """
        full_links = []
        for link in links:
            if self.only_inside_links and link.startswith("http") and base_url not in link:
                continue
            full_link = link if link.startswith("http") else urljoin(base_url, link)
            full_links.append(full_link)
        return full_links

    def filter_links_with_llm(self, links: List[str], user_prompt: str, base_url: str) -> List[str]:
        """
        Filters hyperlinks using LLM based on a user-defined prompt.
        Converts relative links to full URLs before passing them to LLM for analysis.
        
        Args:
            links (List[str]): List of URLs (some may be relative).
            user_prompt (str): The prompt to be passed to the LLM for filtering links.
            base_url (str): The base URL to resolve relative links to full URLs.

        Returns:
            List[str]: A list of URLs that are approved by the LLM.
        """
        filtered_links = []
        self.logger.info(f"Analyzing {len(links)} links using LLM.")

        for link in links:
            # If the link is a relative URL (starts with '/'), resolve it to a full URL
            if link.startswith("/"):
                link = urljoin(base_url, link)  # Convert to full URL based on the base URL
            
            prompt = f"You are performing a web crawler task\n user prompt:{user_prompt}\n Base_url:{base_url} you have pulled this link during the task, URL: {link}\nAccording to the prompt,Should this link be visited? (yes/no):"

            try:
                # Convert prompt to a list of HumanMessage for LLM
                messages = [HumanMessage(content=prompt)]
                
                response = self.llm_model.invoke(messages)  # Pass a list of messages to the model
                
                # Ensure the response content is a string and strip whitespace
                result = str(response.content).strip().lower()
                
            
                if "yes" in result:
                    self.logger.info(f"LLM approved link: {link}")
                    filtered_links.append(link)
                else:
                    self.logger.info(f"LLM rejected link: {link}")
            except Exception as e:
                self.logger.error(f"LLM failed to analyze link: {link}. Error: {str(e)}")
            
        return filtered_links

    def obtain_content(self, documents: List[dict], loader_kwargs: dict, user_prompt: str, base_url: str) -> List[dict]:
        new_documents = []
        for doc in documents:
            source = doc['source']
            if 'document' not in doc:
                document = self.fetch_content(source, loader_kwargs)
    
                if not document or not document[0].page_content.strip():
                    self.logger.warning(f"Failed to fetch content for {source}")
                    continue
    
                self.logger.info(f"Fetched content for {source} at depth {doc.get('depth', 1)}")
                doc['document'] = document
                links = self.extract_links(doc['document'][0].page_content)
    
                # Apply link filtering or extraction based on depth
                filtered_links = self.filter_links_with_llm(links, user_prompt, base_url) if self.depth > 1 else links
    
                full_links = self.get_full_links(source, filtered_links)
    
                # Add the new documents
                for link in full_links:
                    if not any(d.get('source', '') == link for d in documents + new_documents) and doc.get('depth', 1) < self.depth:
                        self.logger.info(f"Adding link to new documents: {link}")
                        new_documents.append({"source": link, "depth": doc.get('depth', 1) + 1})
    
        documents.extend(new_documents)
        return documents

