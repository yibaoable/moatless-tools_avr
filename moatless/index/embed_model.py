import os
from typing import TYPE_CHECKING
from typing import Optional
from llama_index.core.embeddings import BaseEmbedding

def get_embed_model(model_name: str) -> Optional[BaseEmbedding]:
    if model_name.startswith("ep-"):  # 火山方舟模型
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError as e:
            raise ImportError("需要安装 llama-index-embeddings-openai") from e
        
        return OpenAIEmbedding(
            model_name=model_name,
            api_key=os.environ.get("ARK_API_KEY"),
            api_base="https://ark.cn-beijing.volces.com/api/v3"
        )
    elif model_name == "BAAI/bge-code-v1" or model_name == "Qwen/Qwen3-Embedding-8B":
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.core import Settings
        except ImportError as e:
            raise ImportError(
                "llama-index-embeddings-huggingface is not installed. Please install it using `pip install llama-index-embeddings-huggingface`"
            ) from e
        return HuggingFaceEmbedding(
            model_name=model_name
        )
    elif model_name.startswith("voyage"):
        try:
            from llama_index.embeddings.voyageai import VoyageEmbedding
        except ImportError as e:
            raise ImportError(
                "llama-index-embeddings-voyageai is not installed. Please install it using `pip install llama-index-embeddings-voyageai`"
            ) from e
        if "VOYAGE_API_KEY" not in os.environ:
            raise ValueError("VOYAGE_API_KEY environment variable is not set. Please set it to your Voyage API key.")
        print(f"Using Voyage model: {model_name}")
        return VoyageEmbedding(
            model_name=model_name,
            voyage_api_key=os.environ.get("VOYAGE_API_KEY"),
            truncation=True,
            embed_batch_size=60,
        )
    else:
        # Assumes OpenAI otherwise
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError as e:
            raise ImportError(
                "llama-index-embeddings-openai is not installed. Please install it using `pip install llama-index-embeddings-openai`"
            ) from e
        return OpenAIEmbedding(model_name=model_name)