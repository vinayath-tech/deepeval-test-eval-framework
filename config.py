import os
from dotenv import load_dotenv, find_dotenv
from deepeval.models import OllamaModel, OllamaEmbeddingModel

load_dotenv(find_dotenv())

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_OPENAI_BASE_URL = f"{OLLAMA_BASE_URL}/v1"

ORDER_AGENT_MODEL = os.getenv("ORDER_AGENT_MODEL", "ollama:qwen2.5:3b")
ORDER_AGENT_JUDGE_MODEL_LOCAL = OllamaModel(
    model = os.getenv("ORDER_AGENT_JUDGE_MODEL", "qwen2.5:3b"),
    base_url= OLLAMA_BASE_URL,
    temperature = 0
)
ORDER_AGENT_JUDGE_MODEL_OPENAI = os.getenv("ORDER_AGENT_JUDGE_MODEL_OPENAI", "gpt-4.1-mini")

SUMMARIZER_AGENT_MODEL = os.getenv("SUMMARIZER_AGENT_MODEL", "qwen2.5:3b")
SUMMARIZER_JUDGE_MODEL_LOCAL = OllamaModel(
    model = os.getenv("SUMMARIZER_JUDGE_MODEL", "qwen2.5:3b"),
    base_url= OLLAMA_BASE_URL,
    temperature=0
)
SUMMARIZER_JUDGE_MODEL_OPENAI = os.getenv("SUMMARIZER_JUDGE_MODEL_OPENAI", "gpt-4.1-mini")

RAG_AGENT_MODEL = os.getenv("RAG_AGENT_MODEL", "ollama:qwen2.5:3b")
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "ollama:nomic-embed-text-v2-moe")
RAG_AGENT_JUDGE_MODEL_LOCAL = OllamaModel(
    model = os.getenv("RAG_AGENT_JUDGE_MODEL", "qwen2.5:3b"),
    base_url = OLLAMA_BASE_URL,
    temperature = 0
)
RAG_SYNTH_EMBEDDER = OllamaEmbeddingModel(
    model = os.getenv("RAG_SYNTH_EMBEDDER", "nomic-embed-text-v2-moe"),
    base_url = OLLAMA_BASE_URL
)
RAG_AGENT_JUDGE_MODEL_OPENAI = os.getenv("RAG_AGENT_JUDGE_MODEL_OPENAI", "gpt-4.1-mini")

CHAT_AGENT_MODEL = os.getenv("CHAT_AGENT_MODEL", "qwen2.5:3b")
CHAT_JUDGE_MODEL_LOCAL = OllamaModel(
    model = os.getenv("CHAT_JUDGE_MODEL", "qwen2.5:3b"),
    base_url = OLLAMA_BASE_URL,
    temperature = 0
)
CHAT_JUDGE_MODEL_OPENAI = os.getenv("CHAT_JUDGE_MODEL_OPENAI", "gpt-4.1-mini")