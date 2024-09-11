import os

llm_studio_base_api_url = os.getenv("LLM_STUDIO_BASE_API_URL", "http://127.0.0.1:1234")
llm_studio_chat_completion_url = llm_studio_base_api_url + os.getenv("LLM_STUDIO_CHAT_COMPLETION_PATH", "/v1/chat-completion")
llm_studio_embeddings_url = llm_studio_base_api_url + os.getenv("LLM_STUDIO_EMBEDDINGS_PATH", "/v1/embeddings")
llm_studio_model_id = os.getenv("LLM_STUDIO_MODEL_ID", "meta-llama-3.1-8b-instruct-128k-f16")
ollama_model_id = os.getenv("OLLAMA_MODEL_ID", "llama3.1:8b")
ollama_base_api_url = os.getenv("OLLAMA_BASE_API_URL", "http://127.0.0.1:11434")
