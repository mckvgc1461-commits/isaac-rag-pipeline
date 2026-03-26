import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- ⚙️ CONFIGURATION (LOCAL & PROFESSIONAL) ---
print("⚙️ Initializing local AI models (Ollama & BGE)...")

# Using Llama3 as the reasoning engine
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# Using BGE-Small for high-efficiency local embeddings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def run_isaac_final_system():
    # 1. Directory & File Validation
    if not os.path.exists("./documents") or not os.listdir("./documents"):
        print("❌ Error: 'documents' folder is empty! Please add your research files (PDF/TXT).")
        return

    # 2. Smart Indexing (Persistence to prevent duplicates and increase speed)
    PERSIST_DIR = "./storage"
    
    if not os.path.exists(PERSIST_DIR):
        # First run: Read documents and save to ./storage
        print("📄 Reading research documents and creating NEW index...")
        documents = SimpleDirectoryReader("./documents").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # Subsequent runs: Load from local storage for speed and consistency
        print("🧠 Loading existing index from storage (Fast Mode)...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    # 3. Initialize Query Engine (Retrieve top 2 relevant chunks)
    query_engine = index.as_query_engine(similarity_top_k=2)

    # 4. Define the Question
    user_question = "What is the main goal of the Isaac project?"
    print(f"\n❓ Question: {user_question}")

    # 5. Generate Response with Hallucination Guard
    response = query_engine.query(user_question)
    
    # Check the confidence score of the top result
    best_score = response.source_nodes[0].score if response.source_nodes else 0

    print("\n💡 [AI RESPONSE]:")
    
    # 🛡️ THRESHOLD CHECK: If score is below 0.70, the AI won't guess/hallucinate.
    if best_score < 0.70:
        print("I'm sorry, I couldn't find enough reliable information in the documents to answer this accurately.")
    else:
        print(response)

    # 6. Professional Source Attribution (Unique sources only)
    print("\n📚 [SOURCE USED]:")
    seen_files = set()
    for node in response.source_nodes:
        file_name = node.metadata.get('file_name')
        if file_name not in seen_files:
            print(f"- From file: {file_name} (Match score: {node.score:.2f})")
            seen_files.add(file_name)

if __name__ == "__main__":
    run_isaac_final_system()
