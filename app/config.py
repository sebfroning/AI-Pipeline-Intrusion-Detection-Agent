import os

from langchain_ollama import ChatOllama


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

model = ChatOllama(model=OLLAMA_MODEL)
