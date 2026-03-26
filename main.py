import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

# --- ⚙️ CONFIGURATION ---
print("⚙️ Initializing ISAAC RAG Engine (Professional Suite)...")

# Settings for Llama 3 and BGE Embeddings
Settings.llm = Ollama(model="llama3", request_timeout=600.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def run_isaac_system():
    # 1. Directory and Extractor Setup
    DOCS_DIR = "./documents"
    PERSIST_DIR = "./storage"

    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        print("❌ Error: 'documents' directory is missing or empty.")
        return
    
    # Using PyMuPDF for high-fidelity scientific document parsing
    file_extractor = {".pdf": PyMuPDFReader()}

    # 2. Intelligent Indexing & Persistence Layer
    if not os.path.exists(PERSIST_DIR):
        print("📄 Parsing research documents and generating vector index...")
        
        # Load data using specialized PDF extractor
        documents = SimpleDirectoryReader(DOCS_DIR, file_extractor=file_extractor).load_data()
        
        # Logging metadata for verification
        for doc in documents:
            file_name = doc.metadata.get('file_name', 'Unknown')
            char_count = len(doc.text)
            print(f"✅ Indexed: {file_name} | {char_count} characters processed.")
        
        # Create and persist index
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print("💾 Vector index successfully persisted to local storage.")
    else:
        print("🧠 Loading existing knowledge base from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    # 3. Query Engine Configuration
    # Optimized for top 2 most relevant chunks
    query_engine = index.as_query_engine(similarity_top_k=2)

    # Example Research Query
    user_query = "What is the primary methodology and technical goal of the Isaac project?"
    print(f"\n🔍 [QUERY]: {user_query}")

    # 4. Context-Aware Synthesis
    print("⏳ AI is synthesizing the response...")
    response = query_engine.query(user_query)
    
    if not response.source_nodes:
        print("\n💡 [ISAAC RESPONSE]: No relevant context found in the provided documents.")
        return

    # Confidence Threshold Analysis
    confidence_score = response.source_nodes[0].score if response.source_nodes[0].score else 0

    print("-" * 50)
    print("💡 [ISAAC RESPONSE]:")
    
    if confidence_score < 0.60:
        print(f"I found some information, but the confidence score is low ({confidence_score:.2f}).")
        print("Please verify the document content.")
    else:
        print(response)
    print("-" * 50)

if __name__ == "__main__":
    run_isaac_system()
