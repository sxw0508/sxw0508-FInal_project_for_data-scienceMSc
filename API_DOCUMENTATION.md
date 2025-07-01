# ScrapeGraphAI API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Graphs](#graphs)
5. [Nodes](#nodes)
6. [Builders](#builders)
7. [Models](#models)
8. [Utilities](#utilities)
9. [Examples](#examples)
10. [Configuration](#configuration)
11. [Troubleshooting](#troubleshooting)

## Overview

ScrapeGraphAI is a Python library that uses Large Language Models (LLMs) and graph-based workflows to create intelligent web scraping pipelines. It provides a modular architecture where different tasks are represented as nodes in a graph, allowing for flexible and reusable scraping solutions.

### Key Features

- **LLM-powered scraping**: Uses natural language understanding to extract information
- **Graph-based workflows**: Modular, reusable node-based architecture  
- **Multiple data sources**: Support for URLs, local files, PDFs, CSVs, JSON, XML
- **Schema validation**: Structured output with Pydantic models
- **Multi-model support**: OpenAI, Google, Mistral, Ollama, and more
- **Async processing**: Parallel execution for better performance

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

```
langchain>=0.2.14
langchain-openai>=0.1.22
beautifulsoup4>=4.12.3
pandas>=2.2.2
playwright>=1.43.0
# ... see requirements.txt for full list
```

## Core Concepts

### Graph Architecture

ScrapeGraphAI uses a directed graph where:
- **Nodes** represent individual processing steps
- **Edges** define the flow of data between nodes
- **State** is passed between nodes containing the current data

### Basic Workflow

1. **Fetch**: Retrieve content from source (URL, file, etc.)
2. **Parse**: Process and clean the content
3. **Generate**: Use LLM to extract structured information
4. **Output**: Return formatted results

## Graphs

### AbstractGraph

Base class for all graph implementations.

```python
from scrapegraphai.graphs import AbstractGraph

class AbstractGraph(ABC):
    def __init__(self, prompt: str, config: dict, source: Optional[str] = None, schema: Optional[BaseModel] = None):
        """
        Initialize a graph.
        
        Args:
            prompt: Natural language description of what to extract
            config: Configuration dictionary with LLM and other settings
            source: URL, file path, or content to scrape
            schema: Pydantic model for structured output
        """
```

### SmartScraperGraph

The main graph for intelligent web scraping.

```python
from scrapegraphai.graphs import SmartScraperGraph

# Basic usage
graph = SmartScraperGraph(
    prompt="List me all the attractions in Chioggia.",
    source="https://en.wikipedia.org/wiki/Chioggia", 
    config={"llm": {"model": "openai/gpt-3.5-turbo", "api_key": "your-key"}}
)

result = graph.run()
print(result)
```

#### Configuration Options

- `html_mode` (bool): Process raw HTML without parsing (default: False)
- `reasoning` (bool): Enable reasoning step for complex extraction (default: False)
- `reattempt` (bool): Retry on failure with regeneration (default: False)
- `force` (bool): Force markdown conversion (default: False)
- `cut` (bool): Enable content truncation (default: True)
- `additional_info` (str): Additional context for the LLM

#### Example with Schema

```python
from pydantic import BaseModel
from typing import List

class Attraction(BaseModel):
    name: str
    description: str
    location: str

class AttractionsSchema(BaseModel):
    attractions: List[Attraction]

graph = SmartScraperGraph(
    prompt="Extract tourist attractions with names, descriptions and locations",
    source="https://example.com/attractions",
    config={"llm": {"model": "openai/gpt-4", "api_key": "your-key"}},
    schema=AttractionsSchema
)

result = graph.run()  # Returns structured AttractionsSchema object
```

### Other Graph Types

#### SmartScraperMultiGraph
Process multiple URLs with the same prompt.

```python
from scrapegraphai.graphs import SmartScraperMultiGraph

graph = SmartScraperMultiGraph(
    prompt="Extract product information",
    source=["https://example1.com", "https://example2.com"],
    config={"llm": {"model": "openai/gpt-3.5-turbo", "api_key": "your-key"}}
)
```

#### SearchGraph
Search the internet and scrape results.

```python
from scrapegraphai.graphs import SearchGraph

graph = SearchGraph(
    prompt="Find the latest news about AI",
    config={"llm": {"model": "openai/gpt-3.5-turbo", "api_key": "your-key"}}
)
```

#### JSONScraperGraph / CSVScraperGraph / XMLScraperGraph
Specialized graphs for different data formats.

```python
from scrapegraphai.graphs import JSONScraperGraph, CSVScraperGraph

# JSON scraping
json_graph = JSONScraperGraph(
    prompt="Extract user information",
    source="data.json",
    config={"llm": {"model": "openai/gpt-3.5-turbo", "api_key": "your-key"}}
)

# CSV scraping  
csv_graph = CSVScraperGraph(
    prompt="Analyze sales data and find trends",
    source="sales.csv",
    config={"llm": {"model": "openai/gpt-3.5-turbo", "api_key": "your-key"}}
)
```

#### OmniScraperGraph
Universal scraper that automatically detects content type.

```python
from scrapegraphai.graphs import OmniScraperGraph

graph = OmniScraperGraph(
    prompt="Extract main information",
    source="document.pdf",  # Automatically handles PDF, HTML, JSON, etc.
    config={"llm": {"model": "openai/gpt-3.5-turbo", "api_key": "your-key"}}
)
```

#### ScriptCreatorGraph
Generate scraping scripts instead of performing extraction.

```python
from scrapegraphai.graphs import ScriptCreatorGraph

graph = ScriptCreatorGraph(
    prompt="Create a script to scrape product prices",
    source="https://example-shop.com",
    config={"llm": {"model": "openai/gpt-3.5-turbo", "api_key": "your-key"}}
)

script = graph.run()  # Returns Python scraping script
```

### BaseGraph

Core graph execution engine.

```python
from scrapegraphai.graphs import BaseGraph
from scrapegraphai.nodes import FetchNode, ParseNode, GenerateAnswerNode

# Create custom graph
fetch_node = FetchNode(input="url", output=["doc"])
parse_node = ParseNode(input="doc", output=["parsed_doc"])  
answer_node = GenerateAnswerNode(input="user_prompt & parsed_doc", output=["answer"])

graph = BaseGraph(
    nodes=[fetch_node, parse_node, answer_node],
    edges=[(fetch_node, parse_node), (parse_node, answer_node)],
    entry_point=fetch_node
)

# Execute graph
initial_state = {"user_prompt": "Extract main content", "url": "https://example.com"}
final_state, execution_info = graph.execute(initial_state)
```

## Nodes

### BaseNode

Abstract base class for all nodes.

```python
from scrapegraphai.nodes import BaseNode

class BaseNode(ABC):
    def __init__(self, node_name: str, node_type: str, input: str, output: List[str], 
                 min_input_len: int = 1, node_config: Optional[dict] = None):
        """
        Initialize a node.
        
        Args:
            node_name: Unique identifier for the node
            node_type: Either 'node' or 'conditional_node' 
            input: Boolean expression defining required input keys
            output: List of keys this node will add to state
            min_input_len: Minimum number of input keys required
            node_config: Additional configuration parameters
        """
    
    @abstractmethod
    def execute(self, state: dict) -> dict:
        """Execute the node's logic and return updated state."""
```

### Core Nodes

#### FetchNode
Retrieves content from URLs, files, or local sources.

```python
from scrapegraphai.nodes import FetchNode

fetch_node = FetchNode(
    input="url",
    output=["doc"],
    node_config={
        "llm_model": llm_model,
        "headless": True,  # Run browser in headless mode
        "loader_kwargs": {"wait_until": "networkidle"},  # Playwright options
        "browser_base": {  # Optional: Use BrowserBase service
            "api_key": "your-key",
            "project_id": "your-project"
        }
    }
)
```

**Supported Input Types:**
- `url`: Web pages (http/https)
- `local_dir`: Local HTML content
- `pdf`: PDF files
- `csv`: CSV files
- `json`: JSON files
- `xml`: XML files
- `md`: Markdown files

#### ParseNode
Processes and chunks content for LLM consumption.

```python
from scrapegraphai.nodes import ParseNode

parse_node = ParseNode(
    input="doc",
    output=["parsed_doc"],
    node_config={
        "llm_model": llm_model,
        "chunk_size": 4000  # Token limit per chunk
    }
)
```

#### GenerateAnswerNode
Uses LLM to extract information based on prompt.

```python
from scrapegraphai.nodes import GenerateAnswerNode

answer_node = GenerateAnswerNode(
    input="user_prompt & parsed_doc",
    output=["answer"],
    node_config={
        "llm_model": llm_model,
        "schema": MySchema,  # Optional: Pydantic schema for structured output
        "additional_info": "Focus on extracting prices in USD"
    }
)
```

#### ConditionalNode
Routes execution based on conditions.

```python
from scrapegraphai.nodes import ConditionalNode

conditional_node = ConditionalNode(
    input="answer", 
    output=["answer"],
    node_config={
        "key_name": "answer",
        "condition": 'not answer or answer == "NA"'
    }
)
```

### Specialized Nodes

#### SearchInternetNode
Searches the web for information.

```python
from scrapegraphai.nodes import SearchInternetNode

search_node = SearchInternetNode(
    input="user_prompt",
    output=["search_results"],
    node_config={
        "llm_model": llm_model,
        "max_results": 10
    }
)
```

#### RAGNode
Performs retrieval-augmented generation.

```python
from scrapegraphai.nodes import RAGNode

rag_node = RAGNode(
    input="user_prompt & parsed_doc",
    output=["relevant_chunks"],
    node_config={
        "llm_model": llm_model,
        "embedder_model": embedder_model,
        "top_k": 5
    }
)
```

#### ImageToTextNode
Extracts text from images using OCR.

```python
from scrapegraphai.nodes import ImageToTextNode

image_node = ImageToTextNode(
    input="image_data",
    output=["extracted_text"],
    node_config={
        "llm_model": vision_model  # Vision-capable model
    }
)
```

#### TextToSpeechNode
Converts text to speech.

```python
from scrapegraphai.nodes import TextToSpeechNode

tts_node = TextToSpeechNode(
    input="answer",
    output=["audio_data"],
    node_config={
        "tts_model": tts_model,
        "voice": "alloy"
    }
)
```

### Node Input Expressions

Nodes use boolean expressions to define required inputs:

```python
# Single input
input="user_prompt"

# Multiple inputs with AND
input="user_prompt & parsed_doc"

# Alternative inputs with OR  
input="parsed_doc | doc"

# Complex expressions with parentheses
input="user_prompt & (parsed_doc | doc)"
```

## Builders

### GraphBuilder

Automatically generates graph configurations from natural language descriptions.

```python
from scrapegraphai.builders import GraphBuilder

# Initialize builder
builder = GraphBuilder(
    prompt="Create a scraper to extract product information from e-commerce sites",
    config={
        "llm": {
            "api_key": "your-openai-key",
            "model": "gpt-4",
            "temperature": 0.7
        }
    }
)

# Generate graph configuration
graph_config = builder.build_graph()
print(graph_config)

# Visualize graph (requires graphviz)
viz = GraphBuilder.convert_json_to_graphviz(graph_config)
viz.render('scraping_graph', format='png')
```

#### Available Nodes for GraphBuilder

The GraphBuilder can automatically select from these node types:

- `FetchNode`: Retrieve content from sources
- `ParseNode`: Process and chunk content
- `GenerateAnswerNode`: Extract information with LLM
- `SearchInternetNode`: Search the web
- `RAGNode`: Retrieval-augmented generation
- `ConditionalNode`: Conditional routing
- `MergeAnswersNode`: Combine multiple results
- `ImageToTextNode`: OCR and image analysis
- `TextToSpeechNode`: Text-to-speech conversion

## Models

### LLM Integration

ScrapeGraphAI supports multiple LLM providers:

#### OpenAI
```python
config = {
    "llm": {
        "model": "openai/gpt-4",
        "api_key": "your-openai-key",
        "temperature": 0.1,
        "max_tokens": 2000
    }
}
```

#### Google Gemini
```python
config = {
    "llm": {
        "model": "google_genai/gemini-pro", 
        "api_key": "your-google-key",
        "temperature": 0.1
    }
}
```

#### Anthropic Claude
```python
config = {
    "llm": {
        "model": "anthropic/claude-3-sonnet-20240229",
        "api_key": "your-anthropic-key",
        "temperature": 0.1
    }
}
```

#### Ollama (Local)
```python
config = {
    "llm": {
        "model": "ollama/llama2",
        "base_url": "http://localhost:11434",
        "temperature": 0.1
    }
}
```

#### Custom Model Instance
```python
from langchain_openai import ChatOpenAI

custom_llm = ChatOpenAI(model="gpt-4", api_key="your-key")

config = {
    "llm": {
        "model_instance": custom_llm,
        "model_tokens": 8192
    }
}
```

### Specialized Models

#### OpenAI Image-to-Text
```python
from scrapegraphai.models import OpenAIImageToText

itt_model = OpenAIImageToText(
    api_key="your-openai-key",
    model="gpt-4-vision-preview",
    max_tokens=500
)
```

#### OpenAI Text-to-Speech
```python
from scrapegraphai.models import OpenAITextToSpeech

tts_model = OpenAITextToSpeech(
    api_key="your-openai-key",
    model="tts-1",
    voice="alloy"
)
```

## Utilities

### HTML Processing

```python
from scrapegraphai.utils import cleanup_html, convert_to_md

# Clean HTML content
cleaned_html = cleanup_html(raw_html, base_url)

# Convert HTML to Markdown
markdown_content = convert_to_md(html_content)
```

### Text Processing

```python
from scrapegraphai.utils import split_text_into_chunks, num_tokens_calculus

# Split text into chunks
chunks = split_text_into_chunks(text, chunk_size=1000, overlap=200)

# Calculate token count
token_count = num_tokens_calculus(text, model="gpt-3.5-turbo")
```

### Proxy Management

```python
from scrapegraphai.utils import Proxy, search_proxy_servers

# Create proxy configuration
proxy = Proxy()
proxy_config = proxy.get_proxy()

# Search for proxy servers
proxies = search_proxy_servers(country="US", anonymity="elite")
```

### Data Export

```python
from scrapegraphai.utils import export_to_json, export_to_csv, export_to_xml

# Export results to different formats
export_to_json(data, "results.json")
export_to_csv(data, "results.csv") 
export_to_xml(data, "results.xml")
```

### Logging

```python
from scrapegraphai.utils import get_logger, set_verbosity_info, set_verbosity_warning

# Get logger instance
logger = get_logger()

# Set verbosity levels
set_verbosity_info()    # Detailed logging
set_verbosity_warning() # Minimal logging
```

## Examples

### Basic Web Scraping

```python
from scrapegraphai.graphs import SmartScraperGraph

# Simple extraction
graph = SmartScraperGraph(
    prompt="Extract the main article title and summary",
    source="https://example-news.com/article",
    config={
        "llm": {
            "model": "openai/gpt-3.5-turbo",
            "api_key": "your-openai-key"
        }
    }
)

result = graph.run()
print(result)
```

### E-commerce Product Scraping

```python
from pydantic import BaseModel
from typing import List, Optional

class Product(BaseModel):
    name: str
    price: float
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    availability: str

class ProductList(BaseModel):
    products: List[Product]

graph = SmartScraperGraph(
    prompt="Extract product information including name, price, rating, number of reviews, and availability",
    source="https://example-store.com/products",
    config={
        "llm": {
            "model": "openai/gpt-4",
            "api_key": "your-openai-key"
        }
    },
    schema=ProductList
)

products = graph.run()
for product in products.products:
    print(f"{product.name}: ${product.price}")
```

### Multi-page Scraping

```python
from scrapegraphai.graphs import SmartScraperMultiGraph

urls = [
    "https://news-site.com/tech/article1",
    "https://news-site.com/tech/article2", 
    "https://news-site.com/tech/article3"
]

graph = SmartScraperMultiGraph(
    prompt="Extract article title, author, publication date, and main points",
    source=urls,
    config={
        "llm": {
            "model": "openai/gpt-3.5-turbo",
            "api_key": "your-openai-key"
        }
    }
)

results = graph.run()  # Returns list of results for each URL
```

### Search and Scrape

```python
from scrapegraphai.graphs import SearchGraph

graph = SearchGraph(
    prompt="Find recent news about artificial intelligence breakthroughs in 2024",
    config={
        "llm": {
            "model": "openai/gpt-3.5-turbo", 
            "api_key": "your-openai-key"
        },
        "max_results": 5
    }
)

news_summary = graph.run()
```

### PDF Document Processing

```python
from scrapegraphai.graphs import DocumentScraperGraph

graph = DocumentScraperGraph(
    prompt="Extract key financial metrics and highlights from this annual report",
    source="annual_report_2023.pdf",
    config={
        "llm": {
            "model": "openai/gpt-4",
            "api_key": "your-openai-key"
        }
    }
)

financial_data = graph.run()
```

### Custom Graph Creation

```python
from scrapegraphai.graphs import BaseGraph
from scrapegraphai.nodes import (
    FetchNode, ParseNode, RAGNode, 
    GenerateAnswerNode, ConditionalNode
)

# Create nodes with configuration
fetch_node = FetchNode(
    input="url",
    output=["doc"],
    node_config={"llm_model": llm_model}
)

parse_node = ParseNode(
    input="doc", 
    output=["parsed_doc"],
    node_config={"llm_model": llm_model, "chunk_size": 4000}
)

rag_node = RAGNode(
    input="user_prompt & parsed_doc",
    output=["relevant_chunks"],
    node_config={
        "llm_model": llm_model,
        "embedder_model": embedder_model
    }
)

answer_node = GenerateAnswerNode(
    input="user_prompt & relevant_chunks",
    output=["answer"],
    node_config={"llm_model": llm_model}
)

# Create graph
custom_graph = BaseGraph(
    nodes=[fetch_node, parse_node, rag_node, answer_node],
    edges=[
        (fetch_node, parse_node),
        (parse_node, rag_node), 
        (rag_node, answer_node)
    ],
    entry_point=fetch_node
)

# Execute
state = {
    "user_prompt": "What are the main topics discussed?",
    "url": "https://example.com/article"
}

final_state, exec_info = custom_graph.execute(state)
result = final_state["answer"]
```

## Configuration

### LLM Configuration

```python
# Basic configuration
config = {
    "llm": {
        "model": "openai/gpt-3.5-turbo",
        "api_key": "your-api-key",
        "temperature": 0.1,
        "max_tokens": 2000
    }
}

# Advanced configuration with rate limiting
config = {
    "llm": {
        "model": "openai/gpt-4",
        "api_key": "your-api-key", 
        "temperature": 0.1,
        "rate_limit": {
            "requests_per_second": 1,
            "max_retries": 3
        }
    }
}
```

### Browser Configuration

```python
config = {
    "llm": {"model": "openai/gpt-3.5-turbo", "api_key": "key"},
    "headless": True,  # Run browser in headless mode
    "loader_kwargs": {
        "wait_until": "networkidle",  # Wait for network idle
        "timeout": 30000,  # 30 second timeout
        "user_agent": "Custom User Agent"
    }
}
```

### Proxy Configuration

```python
config = {
    "llm": {"model": "openai/gpt-3.5-turbo", "api_key": "key"},
    "loader_kwargs": {
        "proxy": {
            "server": "http://proxy-server:8080",
            "username": "user",
            "password": "pass"
        }
    }
}
```

### Global Settings

```python
config = {
    "llm": {"model": "openai/gpt-3.5-turbo", "api_key": "key"},
    "verbose": True,           # Enable detailed logging
    "headless": False,         # Show browser window
    "html_mode": False,        # Enable HTML parsing
    "reasoning": True,         # Enable reasoning step
    "reattempt": True,         # Retry on failure
    "force": False,            # Force markdown conversion
    "cut": True,               # Enable content truncation
    "cache_path": "./cache",   # Cache directory
    "additional_info": "Focus on extracting numerical data"
}
```

## Troubleshooting

### Common Issues

#### 1. API Key Errors
```python
# Ensure API key is set correctly
config = {
    "llm": {
        "model": "openai/gpt-3.5-turbo",
        "api_key": "sk-..."  # Must start with sk- for OpenAI
    }
}
```

#### 2. Content Not Loading
```python
# Try different browser settings
config = {
    "llm": {"model": "openai/gpt-3.5-turbo", "api_key": "key"},
    "loader_kwargs": {
        "wait_until": "domcontentloaded",  # Try different wait conditions
        "timeout": 60000,  # Increase timeout
        "headless": False  # Debug with visible browser
    }
}
```

#### 3. Rate Limiting
```python
# Add rate limiting
config = {
    "llm": {
        "model": "openai/gpt-3.5-turbo", 
        "api_key": "key",
        "rate_limit": {
            "requests_per_second": 0.5,  # Slow down requests
            "max_retries": 5
        }
    }
}
```

#### 4. Large Content Issues
```python
# Enable chunking and parsing
config = {
    "llm": {"model": "openai/gpt-3.5-turbo", "api_key": "key"},
    "html_mode": False,  # Enable parsing
    "cut": True,         # Enable content cutting
}
```

#### 5. Schema Validation Errors
```python
from pydantic import BaseModel, Field
from typing import Optional

class SafeSchema(BaseModel):
    title: Optional[str] = Field(None, description="Article title")
    content: Optional[str] = Field(None, description="Main content")
    # Use Optional fields to handle missing data gracefully
```

### Debug Mode

```python
# Enable verbose logging for debugging
config = {
    "llm": {"model": "openai/gpt-3.5-turbo", "api_key": "key"},
    "verbose": True,
    "headless": False  # See browser actions
}

graph = SmartScraperGraph(prompt, source, config)
result = graph.run()

# Check execution info
exec_info = graph.get_execution_info()
for info in exec_info:
    print(f"Node: {info['node_name']}")
    print(f"Tokens: {info['total_tokens']}")
    print(f"Cost: ${info['total_cost_USD']}")
    print(f"Time: {info['exec_time']}s")
```

### Performance Optimization

```python
# Optimize for speed and cost
config = {
    "llm": {
        "model": "openai/gpt-3.5-turbo",  # Faster, cheaper model
        "api_key": "key",
        "temperature": 0  # Deterministic results
    },
    "html_mode": True,    # Skip parsing for simple extractions
    "cut": True,          # Truncate large content
    "cache_path": "./cache"  # Cache results
}
```

### Error Handling

```python
try:
    graph = SmartScraperGraph(prompt, source, config)
    result = graph.run()
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Scraping failed: {e}")
    # Check if it's a network issue, API limit, etc.
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review configuration options
- Enable verbose logging for debugging
- Ensure all dependencies are installed correctly