"Developed with the assistance of AI tools to ensure high-quality local RAG implementation."

pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface

Put your data in ./documents folder.

Run python main.py


⚙️ Initializing local AI models (Ollama & BGE)...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████████████████████████████| 199/199 [00:00<00:00, 8653.97it/s]
BertModel LOAD REPORT from: BAAI/bge-small-en-v1.5
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
📄 Reading research documents and creating NEW index...

❓ Question: What is the main goal of the Isaac project?

💡 [AI RESPONSE]:
I'm sorry, I couldn't find enough reliable information in the documents to answer this accurately.

📚 [SOURCE USED]:
- From file: micobaba.pdf (Match score: 0.65)
PS C:\Users\kerem-mirac\Desktop\isaac-rag-pipeline> 
