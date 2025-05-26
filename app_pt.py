# Adaptado para uso dinâmico entre Ollama e SambaNova

import os
import gc
import uuid
import tempfile
import base64
from dotenv import load_dotenv
from rag_code_pt import Transcribe, EmbedData, QdrantVDB_QB, Retriever, RAG
import streamlit as st

# Carrega variáveis de ambiente
load_dotenv()

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
collection_name = "chat_com_audios"
batch_size = 32

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

with st.sidebar:
    st.header("Adicione seu arquivo de áudio!")

    uploaded_file = st.file_uploader("Escolha um arquivo de áudio", type=["mp3", "wav", "m4a"])

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Transcrevendo com AssemblyAI e armazenando no banco vetorial...")

                if file_key not in st.session_state.get('file_cache', {}):
                    # Inicializa transcriber
                    transcriber = Transcribe(api_key=os.getenv("ASSEMBLYAI_API_KEY"))
                    
                    # Obtém transcrições com identificação de locutor
                    transcripts = transcriber.transcribe_audio(file_path)
                    st.session_state.transcripts = transcripts

                    # Cada segmento de locutor torna-se um documento separado para embedding
                    documents = [f"Locutor {t['speaker']}: {t['text']}" for t in transcripts]

                    # embed data    
                    embeddata = EmbedData(
                        embed_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                        batch_size=batch_size
                    )
                    embeddata.embed(documents)

                    # configura banco de dados vetorial
                    qdrant_vdb = QdrantVDB_QB(
                        collection_name=collection_name,
                        embeddata=embeddata,
                        batch_size=batch_size
                    )
                    qdrant_vdb.define_client()
                    qdrant_vdb.create_collection()
                    qdrant_vdb.ingest_data()

                    # configura retriever
                    retriever = Retriever(vector_db=qdrant_vdb, embeddata=embeddata)

                    # configura rag
                    query_engine = RAG(
                        retriever=retriever, 
                        llm_name=os.getenv("LLM_MODEL_NAME", "DeepSeek-R1-Distill-Llama-70B")
                    )
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                # Informa ao usuário que o arquivo foi processado
                st.success("Pronto para conversar!")
                
                # Exibe player de áudio
                st.audio(uploaded_file)

                # Exibe transcrição com identificação de locutor
                st.subheader("Transcrição")
                with st.expander("Exibir transcrição completa", expanded=True):
                    for t in st.session_state.transcripts:
                        st.text(f"**{t['speaker']}**: {t['text']}")

        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
            st.stop()     

col1, col2 = st.columns([6, 1])

with col1:
    # Verifica se os arquivos de imagem existem antes de tentar carregá-los
    model_name = os.getenv("LLM_MODEL_NAME", "DeepSeek R1 Distill")
    
    # Tenta carregar imagens da pasta assets
    assemblyai_img = None
    deepseek_img = None
    
    # Busca imagem do AssemblyAI
    for img_file in ["Assemblyai.png", "assemblyai.png", "AssemblyAI.png", "assembly_ai.png"]:
        img_path = os.path.join("assets", img_file)
        if os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    assemblyai_img = base64.b64encode(f.read()).decode()
                break
            except Exception:
                continue
    
    # Busca imagem do DeepSeek/SambaNova/Llama
    for img_file in ["deepseek.png", "deep-seek.png", "DeepSeek.png", "sambanova.png", "samba-nova.png", "llama.png", "meta-llama.png", "Meta-Llama.png"]:
        img_path = os.path.join("assets", img_file)
        if os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    deepseek_img = base64.b64encode(f.read()).decode()
                break
            except Exception:
                continue
    
    # Renderiza o título baseado nas imagens disponíveis
    if assemblyai_img and deepseek_img:
        st.markdown(f"""
        # RAG sobre Áudio impulsionado por <img src="data:image/png;base64,{assemblyai_img}" width="200" style="vertical-align: -15px; padding-right: 10px;"> e <img src="data:image/png;base64,{deepseek_img}" width="200" style="vertical-align: -15px; padding-left: 10px;">
        """, unsafe_allow_html=True)
    elif assemblyai_img:
        st.markdown(f"""
        # RAG sobre Áudio impulsionado por <img src="data:image/png;base64,{assemblyai_img}" width="200" style="vertical-align: -15px; padding-right: 10px;"> e **{model_name}**
        """, unsafe_allow_html=True)
    elif deepseek_img:
        st.markdown(f"""
        # RAG sobre Áudio impulsionado por **AssemblyAI** e <img src="data:image/png;base64,{deepseek_img}" width="200" style="vertical-align: -15px; padding-left: 10px;">
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        # RAG sobre Áudio impulsionado por **AssemblyAI** e **{model_name}**
        """)

with col2:
    st.button("Limpar ↺", on_click=reset_chat)

# Inicializa histórico de chat
if "messages" not in st.session_state:
    reset_chat()

# Exibe mensagens do chat do histórico na re-execução do app
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Aceita entrada do usuário
if prompt := st.chat_input("Pergunte algo sobre o conteúdo do áudio..."):
    # Adiciona mensagem do usuário ao histórico do chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Exibe mensagem do usuário no container de mensagem do chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Exibe resposta do assistente no container de mensagem do chat
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Obtém resposta streaming
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response:
            try:
                new_text = chunk.raw["choices"][0]["delta"]["content"]
                full_response += new_text
                message_placeholder.markdown(full_response + "▌")
            except:
                pass

        message_placeholder.markdown(full_response)

    # Adiciona resposta do assistente ao histórico do chat
    st.session_state.messages.append({"role": "assistant", "content": full_response})