import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- ⚙️ CONFIGURATION ---
print("⚙️ Initializing local AI models (Ollama & BGE)...")
Settings.llm = Ollama(model="llama3", request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def run_isaac_final_system():
    # 1. Directory Validation
    if not os.path.exists("./documents") or not os.listdir("./documents"):
        print("❌ Error: 'documents' folder is empty!")
        return

    # 2. Smart Indexing (Persistence)
    PERSIST_DIR = "./storage"
    if not os.path.exists(PERSIST_DIR):
        print("📄 Reading documents and creating NEW index...")
        documents = SimpleDirectoryReader("./documents").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        print("🧠 Loading existing index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    # 3. Query Engine
    query_engine = index.as_query_engine(similarity_top_k=2)

    # 4. Interactive Question (No more hardcoding!)
    user_question = input("\n❓ Enter your question: ")
    if not user_question.strip():
        print("Empty question. Exiting.")
        return

    # 5. Generate Response
    response = query_engine.query(user_question)
    
    # 🛡️ THE GUARD: Checking if we even have any sources
    if not response.source_nodes:
        print("\n💡 [AI RESPONSE]:")
        print("I'm sorry, I couldn't find any relevant documents to answer this.")
        return

    # Score threshold check
    best_score = response.source_nodes[0].score if response.source_nodes[0].score else 0

    print("\n💡 [AI RESPONSE]:")
    if best_score < 0.70:
        print(f"Confidence is too low ({best_score:.2f}). Please provide more specific documents.")
    else:
        print(response)

    # 6. Clean Source Display
    print("\n📚 [SOURCE USED]:")
    seen_files = set()
    for node in response.source_nodes:
        file_name = node.metadata.get('file_name')
        if file_name not in seen_files:
            print(f"- File: {file_name} (Score: {node.score:.2f})")
            seen_files.add(file_name)

if __name__ == "__main__":
    run_isaac_final_system()
