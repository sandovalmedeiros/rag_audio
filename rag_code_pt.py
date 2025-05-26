from qdrant_client import models
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.sambanovasystems import SambaNovaCloud
import assemblyai as aai
import os
from typing import List, Dict
from llama_index.core.base.llms.types import ChatMessage, MessageRole

def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

class EmbedData:
    def __init__(self, embed_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", batch_size=32):
        self.embed_model_name = embed_model_name
        self.embed_model = self._load_embed_model()
        self.batch_size = batch_size
        self.embeddings = []

    def _load_embed_model(self):
        embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name, trust_remote_code=True, cache_folder='./hf_cache')
        return embed_model

    def generate_embedding(self, context):
        return self.embed_model.get_text_embedding_batch(context)

    def embed(self, contexts):
        self.contexts = contexts
        for batch_context in batch_iterate(contexts, self.batch_size):
            batch_embeddings = self.generate_embedding(batch_context)
            self.embeddings.extend(batch_embeddings)

class QdrantVDB_QB:
    def __init__(self, collection_name, embeddata: EmbedData, batch_size=512):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.embeddata = embeddata
        self.vector_dim = len(self.embeddata.embed_model.get_text_embedding("teste de dimensão"))

    def define_client(self):
        self.client = QdrantClient(url="http://localhost:6333", prefer_grpc=True)

    def create_collection(self):
        # Verifica se a coleção existe e a exclui para recriar
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_dim, 
                distance=models.Distance.DOT,  # Mudança para DOT como na versão original
                on_disk=True
            ),
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=5, 
                indexing_threshold=0
            ),
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True)
            ),
        )

    def ingest_data(self):
        for batch_context, batch_embeddings in zip(
            batch_iterate(self.embeddata.contexts, self.batch_size),
            batch_iterate(self.embeddata.embeddings, self.batch_size)
        ):
            self.client.upload_collection(
                collection_name=self.collection_name,
                vectors=batch_embeddings,
                payload=[{"context": context} for context in batch_context]
            )
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
        )

class Retriever:
    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query):
        query_embedding = self.embeddata.embed_model.get_query_embedding(query)
        result = self.vector_db.client.search(
            collection_name=self.vector_db.collection_name,
            query_vector=query_embedding,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False, 
                    rescore=True, 
                    oversampling=2.0
                )
            ),
            timeout=1000,
        )
        return result

class RAG:
    def __init__(self, retriever, llm_name=None):
        self.retriever = retriever
        # Modelo DeepSeek padrão na SambaNova Cloud
        self.llm_name = llm_name or os.getenv("LLM_MODEL_NAME", "DeepSeek-R1-Distill-Llama-70B")
        
        # Adicionado sistema de mensagens como na versão original
        system_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            content="Você é um assistente útil que responde perguntas sobre o documento do usuário.",
        )
        self.messages = [system_msg]
        
        self.llm = self._setup_llm()

        self.qa_prompt_tmpl_str = """As informações de contexto estão abaixo.
            ---------------------
            {context}
            ---------------------
            Com base nas informações acima, quero que você pense passo a passo para responder à pergunta de forma clara e objetiva. Caso não saiba a resposta, diga apenas 'Não sei'.
            Pergunta: {query}
            Resposta:"""

    def _setup_llm(self):
        # Lógica corrigida: SambaNova Cloud por padrão
        # Só usa Ollama se explicitamente configurado
        if self.llm_name.startswith("ollama:") or "ollama" in self.llm_name.lower():
            # Remove prefixo 'ollama:' se presente
            model_name = self.llm_name.replace("ollama:", "")
            return Ollama(
                model=model_name,
                temperature=0.7,
                context_window=8192,
                options={
                    "num_ctx": 8192,
                    "num_gpu": 0,
                    "low_vram": True,
                }
            )
        else:
            # Usa SambaNova Cloud para todos os outros modelos (incluindo DeepSeek)
            return SambaNovaCloud(
                model=self.llm_name,
                temperature=0.7,
                context_window=32000,
            )

    def generate_context(self, query):
        result = self.retriever.search(query)
        context = [dict(data) for data in result]
        combined_prompt = []
        for entry in context[:2]:
            context_text = entry["payload"]["context"]
            combined_prompt.append(context_text)
        return "\n\n---\n\n".join(combined_prompt)

    def query(self, query):
        context = self.generate_context(query=query)
        prompt = self.qa_prompt_tmpl_str.format(context=context, query=query)
        user_msg = ChatMessage(role=MessageRole.USER, content=prompt)
        streaming_response = self.llm.stream_complete(user_msg.content)
        return streaming_response

class Transcribe:
    def __init__(self, api_key: str):
        """Inicializa a classe Transcribe com a chave da API do AssemblyAI."""
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()

    def transcribe_audio(self, audio_path: str) -> List[Dict[str, str]]:
        """
        Transcreve um arquivo de áudio e retorna transcrições com identificação de locutor.
        
        Args:
            audio_path: Caminho para o arquivo de áudio
            
        Returns:
            Lista de dicionários contendo informações de locutor e texto
        """
        # Configura transcrição com identificação de locutor
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=2,  # Ajuste conforme necessário
            language_code="pt"   # Código de idioma para português
        )
        
        try:
            # Transcreve o áudio
            transcript = self.transcriber.transcribe(audio_path, config=config)
        except Exception as e:
            print(f"❌ Erro na transcrição: {e}")
            raise
        
        # Extrai as falas dos locutores
        speaker_transcripts = []
        for utterance in transcript.utterances:
            speaker_transcripts.append({
                "speaker": f"Locutor {utterance.speaker}",
                "text": utterance.text
            })
        
        return speaker_transcripts