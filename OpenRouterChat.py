import json
import asyncio
import httpx
import re
import os
import chromadb
from chromadb.utils import embedding_functions
import httpx
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import hashlib

OPENROUTE_MODEL = "openrouter/free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- CONTEXT PLACEHOLDER SETUP ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Local embedding model
DOCS_DIRECTORY = "./documents"  # Directory containing your documents
CHROMA_DB_DIR = "./chroma_db"

class RAGSystem:
    def __init__(
        self,
        db_dir= CHROMA_DB_DIR,
        embedding_model= EMBEDDING_MODEL,
        collection_name="documents",
        batch_size=100,  # Number of chunks to insert into ChromaDB at once
    ):
        self.client = chromadb.PersistentClient(path=db_dir)
        self.embedding_func = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_func
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        self.batch_size = batch_size

    def _lazy_file_reader(self, file_path: Path, buffer_size=1024 * 1024):
        """Lazily reads a file in blocks to prevent memory spikes on large files."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                yield data

    def stream_chunks_from_directory(self, target_directory: str):
        """A generator dataloader that yields batches of chunks from disk."""
        dir_path = Path(target_directory)
        if not dir_path.exists():
            return

        batch_documents = []
        batch_ids = []
        batch_metadatas = []

        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                source_path = str(file_path.resolve())
                filename = file_path.name

                # Clear old chunks for this file before re-ingesting
                self.collection.delete(where={"source": source_path})

                path_hash = hashlib.md5(source_path.encode()).hexdigest()
                chunk_index = 0

                # Read file content efficiently (buffers large files)
                file_text = ""
                try:
                    for block in self._lazy_file_reader(file_path):
                        file_text += block
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue

                if not file_text.strip():
                    continue

                # Split the text into chunks
                chunks = self.text_splitter.split_text(file_text)

                for chunk in chunks:
                    chunk_id = f"{path_hash}_{chunk_index}"
                    metadata = {
                        "source": source_path,
                        "filename": filename,
                        "chunk_index": chunk_index,
                    }

                    batch_documents.append(chunk)
                    batch_ids.append(chunk_id)
                    batch_metadatas.append(metadata)
                    chunk_index += 1

                    # Yield a batch when it reaches the batch size limit
                    if len(batch_documents) >= self.batch_size:
                        yield batch_documents, batch_ids, batch_metadatas
                        # Reset batch lists
                        batch_documents, batch_ids, batch_metadatas = [], [], []

        # Yield any remaining items in the final partial batch
        if batch_documents:
            yield batch_documents, batch_ids, batch_metadatas

    def ingest(self, target_directory: str):
        print(f"Starting streaming ingestion from: {target_directory}")
        total_chunks_added = 0

        # Consume the dataloader generator batch by batch
        for documents, ids, metadatas in self.stream_chunks_from_directory(
            target_directory
        ):
            self.collection.add(
                documents=documents, ids=ids, metadatas=metadatas
            )
            total_chunks_added += len(documents)
            print(f"Flushed batch of {len(documents)} chunks to ChromaDB...")

        print(
            f"\nIngestion complete! Total chunks in DB:"
            f" {self.collection.count()}"
        )

    def retrieve_context(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant document chunks"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else None

def track_usage_metrics(chunk: dict):
    """Parses and logs final token usage parameters and API cost metrics."""
    usage = chunk.get("usage")
    if usage:
        print("\n\n--- Usage Statistics ---")
        print(f"Prompt Tokens:     {usage.get('prompt_tokens', 0)}")
        print(f"Completion Tokens: {usage.get('completion_tokens', 0)}")
        print(f"Total Tokens:      {usage.get('total_tokens', 0)}")
        print(f"Estimated Cost:    ${usage.get('cost', 0.0):.7f}")

async def stream_chat(messages, client, headers):
    payload = {
        "model": OPENROUTE_MODEL,
        "messages": messages,
        "stream": True,
    }
    
    full_reply = []
    try:
        async with client.stream("POST", OPENROUTER_URL, headers=headers, json=payload) as response:
            if response.status_code != 200:
                print(f"\nError: {response}")
                return ""

            async for line in response.aiter_lines():
                if not line:
                    continue
                clean_line = re.sub(r'^(data|Assistant):\s*', '', line).strip()
        
                if not clean_line or clean_line == "[DONE]":
                    continue
                    
                try:
                    # Parse line string to Python dictionary
                    chunk = json.loads(clean_line)

                    if "usage" in chunk:
                        track_usage_metrics(chunk=chunk)
                    
                    # Follow OpenAI/OpenRouter nested format schema: choices -> index 0 -> delta -> content
                    choices = chunk.get("choices", [])
                    if choices and isinstance(choices, list):
                        first_choice = choices[0]
                        delta = first_choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            print(content, end="", flush=True)
                            full_reply.append(content)
                        
                except json.JSONDecodeError:
                    # Skip any logs that are not valid JSON lines
                    continue
                    
    except Exception as e:
        print(f"\nConnection Error: {e}")
        
    complete_text = "".join(full_reply)
    return complete_text

def flush_context(conversation_history):
    print(conversation_history)
    conversation_history = [
        {"role": "system", "content": "You are a helpful and concise AI assistant."}
    ]


def parse_user_input_command(command_str=None):
    if command_str:
        if command_str == 'flush_context':
            flush_context(conversation_history=conversation_history)

# Maintains core conversation history (clean, un-duplicated history strings)
conversation_history = [
    {"role": "system", "content": "You are a helpful and concise AI assistant."}
]

async def main():
    if not API_KEY:
        print("API_KEY not set, exiting\n")
        return

    rag = RAGSystem()

    # First-time setup: ingest documents
    if rag.collection.count() == 0:
        print("Ingesting documents...")
        rag.ingest_documents()

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Python Context-Aware Client",
        "Content-Type": "application/json",
    }

    timeout = httpx.Timeout(60.0, connect=10.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        print("--- OpenRouter Chat Loop Started (Type 'quit' to exit) ---")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Exiting chat session. Goodbye!")
                break

            if user_input[0] == '/':
                parse_user_input_command(user_input[1:])
                continue
                
            if not user_input:
                continue

            context = rag.retrieve_context(user_input)
            if not context:
                print("No relevant context found.\n")
                continue
            context_text = "".join(context)
            contextual_content = f"Use the following context to answer the question: Context: {context_text}, Question: {user_input}, Answer:"

            # 1. Append raw query to historical log array so memory stays accurate
            conversation_history.append({"role": "user", "content": user_input})

            # 2. Build a localized copy of payload array containing contextual tags 
            # This injects the placeholder *only* for the current API execution turn
            ephemeral_messages = conversation_history.copy()
            
            # Formulate the contextual query structure sent upstream
            ephemeral_messages[-1] = {"role": "user", "content": contextual_content}

            # 3. Request inference stream using the contextual copy
            assistant_response = await stream_chat(ephemeral_messages, client, headers)
            
            # 4. Save response to main history array for continuous turn references
            if assistant_response:
                conversation_history.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession interrupted. Goodbye!")
