from langchain_community.embeddings import LocalAIEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings

from config import llm_studio_embeddings_url


def get_embedding_function(use_llm_studio: bool = False, model: str = "llama3.1:8b"):
    if use_llm_studio:
        embeddings = LocalAIEmbeddings(
            openai_api_base=llm_studio_embeddings_url,
            model=model,
        )
    else:
        embeddings = OllamaEmbeddings(model=model)
    return embeddings
