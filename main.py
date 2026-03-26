import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- ⚙️ CONFIGURATION ---
print("⚙️ Initializing local AI models (Ollama & BGE)...")
Settings.llm = Ollama(model="llama3", request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def run_isaac_final_system():
    # 1. Klasör Kontrolü
    if not os.path.exists("./documents") or not os.listdir("./documents"):
        print("❌ Error: 'documents' folder is empty!")
        return

    # 2. Akıllı İndeksleme (Persistence)
    PERSIST_DIR = "./storage"
    
    # BURASI ÖNEMLİ: PDF'i güncellediysen eski hafızayı siliyoruz ki yenisini okusun
    if not os.path.exists(PERSIST_DIR):
        print("📄 Reading updated documents and creating index...")
        documents = SimpleDirectoryReader("./documents").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        print("🧠 Loading index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    # 3. Sorgu Motoru
    query_engine = index.as_query_engine(similarity_top_k=2)

    # 4. OTOMATİK SORU (Hiçbir şey yazmana gerek yok!)
    user_question = "What is the main goal of the Isaac project?"
    print(f"\n❓ Question: {user_question}")

    # 5. Cevap Üretme
    response = query_engine.query(user_question)
    
    if not response.source_nodes:
        print("\n💡 [AI RESPONSE]: No sources found.")
        return

    best_score = response.source_nodes[0].score if response.source_nodes[0].score else 0

    print("\n💡 [AI RESPONSE]:")
    # Not: PDF'i doldurduysan artık burası 0.70'i geçer
    if best_score < 0.60: 
        print(f"Confidence low ({best_score:.2f}). Please check if your PDF is full of data.")
    else:
        print(response)

    # 6. Kaynak Gösterimi
    print("\n📚 [SOURCE USED]:")
    seen_files = set()
    for node in response.source_nodes:
        file_name = node.metadata.get('file_name')
        if file_name not in seen_files:
            print(f"- File: {file_name} (Score: {node.score:.2f})")
            seen_files.add(file_name)

if __name__ == "__main__":
    run_isaac_final_system()
