import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

ORDER_AGENT_MODEL = os.getenv("ORDER_AGENT_MODEL", "openai:gpt-4.1-mini")
ORDER_AGENT_JUDGE_MODEL = os.getenv("ORDER_AGENT_JUDGE_MODEL", "gpt-4.1-mini")
SUMMARIZER_AGENT_MODEL = os.getenv("SUMMARIZER_AGENT_MODEL", "gpt-4.1-mini")
CHAT_AGENT_MODEL = os.getenv("CHAT_AGENT_MODEL", "gpt-4.1-mini")
RAG_AGENT_MODEL = os.getenv("RAG_AGENT_MODEL", "openai:gpt-4.1-mini")
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "openai:text-embedding-3-small")
RAG_AGENT_JUDGE_MODEL = os.getenv("RAG_AGENT_JUDGE_MODEL", "gpt-4.1-mini")