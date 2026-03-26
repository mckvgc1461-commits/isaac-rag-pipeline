import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader # <-- Yeni PDF Okuyucu

# --- ⚙️ CONFIGURATION ---
print("⚙️ Initializing Isaac RAG Engine (with PyMuPDF)...")
Settings.llm = Ollama(model="llama3", request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def run_isaac_final_system():
    # 1. Dosya Kontrolü ve PDF Extractor Tanımlama
    if not os.path.exists("./documents") or not os.listdir("./documents"):
        print("❌ Error: 'documents' folder is empty!")
        return
    
    # PDF'leri daha iyi okumak için PyMuPDFReader kullanıyoruz
    file_extractor = {".pdf": PyMuPDFReader()}

    # 2. Akıllı İndeksleme
    PERSIST_DIR = "./storage"
    
    # PDF içeriği değiştiyse veya okunmadıysa storage'ı silmen gerekir
    if not os.path.exists(PERSIST_DIR):
        print("📄 Reading PDF content deeply and creating index...")
        # file_extractor'ı buraya ekledik!
        documents = SimpleDirectoryReader("./documents", file_extractor=file_extractor).load_data()
        
        # DEBUG: PDF gerçekten okundu mu kontrol et (Gerekirse silersin)
        for doc in documents:
            print(f"✅ Loaded: {doc.metadata.get('file_name')} ({len(doc.text)} characters read)")
        
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        print("🧠 Loading existing index (Fast Mode)...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    # 3. Sorgu Motoru
    query_engine = index.as_query_engine(similarity_top_k=2)

    user_question = "What is the main goal of the Isaac project?"
    print(f"\n❓ Question: {user_question}")

    # 4. Cevap Üretme
    response = query_engine.query(user_question)
    
    if not response.source_nodes:
        print("\n💡 [AI RESPONSE]: No document context found.")
        return

    best_score = response.source_nodes[0].score if response.source_nodes[0].score else 0

    print("\n💡 [AI RESPONSE]:")
    if best_score < 0.65:
        print(f"I couldn't find a strong match in the text (Score: {best_score:.2f}).")
    else:
        print(response)

if __name__ == "__main__":
    run_isaac_final_system()
