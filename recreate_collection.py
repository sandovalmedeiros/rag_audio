from qdrant_client import QdrantClient, models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Nome da collection
collection_name = "chat_com_audios"

# Modelo de embedding multilíngue compatível com português
embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Instancia o modelo de embedding
embedder = HuggingFaceEmbedding(model_name=embed_model_name)

# Obtém automaticamente a dimensão do vetor
vector_dim = embedder.dim
print(f"📐 Dimensão detectada do modelo '{embed_model_name}': {vector_dim}")

# Conecta ao Qdrant
client = QdrantClient(url="http://localhost:6333")

# Deleta collection se já existir
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
    print(f"🗑️ Collection '{collection_name}' deletada.")

# Cria nova collection com dimensão correta
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=vector_dim,
        distance=models.Distance.COSINE
    )
)
print(f"✅ Collection '{collection_name}' recriada com dimensão {vector_dim}.")