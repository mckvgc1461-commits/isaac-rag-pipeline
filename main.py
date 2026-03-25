import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- CONFIGURATION (FREE & LOCAL) ---
# We use Ollama for the "Brain" and HuggingFace for "Understanding"
print("⚙️ Initializing local AI models (Ollama)...")
Settings.llm = Ollama(model="llama3", request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def run_isaac_prototype():
    # 1. Load the text file you created in 'documents' folder
    if not os.path.exists("./documents") or not os.listdir("./documents"):
        print("❌ Error: 'documents' folder is empty! Put your .txt file there.")
        return

    print("📄 Reading research documents...")
    documents = SimpleDirectoryReader("./documents").load_data()
    
    # 2. Indexing (Making the data searchable)
    print("🧠 Creating knowledge index...")
    index = VectorStoreIndex.from_documents(documents)
    
    # 3. Setup the Query Engine
    query_engine = index.as_query_engine(similarity_top_k=2)
    
    # 4. Ask the Question
    user_question = "What is the main goal of the Isaac project?"
    print(f"\n❓ Question: {user_question}")
    
    response = query_engine.query(user_question)
    
    # 5. Show the Result
    print("\n💡 [AI RESPONSE]:")
    print(response)
    
    print("\n📚 [SOURCE USED]:")
    for node in response.source_nodes:
        print(f"- From file: {node.metadata.get('file_name')} (Match score: {node.score:.2f})")

if __name__ == "__main__":
    run_isaac_prototype()