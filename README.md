# 🎧 RAG sobre Arquivos de Áudio com AssemblyAI, LLMs e Qdrant

Este projeto implementa um sistema RAG (Retrieval-Augmented Generation) que permite fazer perguntas
sobre o conteúdo de arquivos de áudio transcritos, combinando **transcrição com IA**,
**armazenamento vetorial** e **modelos de linguagem natural**.

---

## 🔍 Tecnologias Utilizadas

- **AssemblyAI** – Transcrição automática de áudio com API de ponta
- **LlamaIndex** – Estrutura RAG e interface com LLMs
- **Qdrant VectorDB** – Banco vetorial de alta performance
- **Streamlit** – Interface web interativa
- **SambaNova / Ollama** – Suporte dinâmico a LLMs locais e em nuvem

---

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/sandovalmedeiros/rag_audio.git
cd rag_audio
```

### 2. Crie e ative o ambiente virtual

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure o arquivo `.env`

Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

Edite com suas chaves reais:

Configurar o AssemblyAI:

Obtenha uma chave de API do AssemblyAI (http://bit.ly/4bGBdux) e defina-a no arquivo .env da seguinte forma:

ASSEMBLYAI_API_KEY=<SUA_CHAVE_API>

Configurar o SambaNova:

Obtenha uma chave de API do SambaNova (https://sambanova.ai/) e defina-a no arquivo .env da seguinte forma:

SAMBANOVA_API_KEY=<SUA_CHAVE_API_SAMBANOVA>
Observação: em vez do SambaNova, você também pode usar o Ollama.

```env
ASSEMBLYAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
SAMBANOVA_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL_NAME=DeepSeek-R1-Distill-Llama-70B  # ou outro modelo como mistral ou tinyllama via Ollama
```

---

## ⚙️ Execução simplificada (Windows)

### ▶️ Iniciar o Qdrant (banco vetorial)

```bash
run_docker.bat
```

> Esse comando inicia o Qdrant via Docker. Necessário apenas uma vez por sessão.

### ▶️ Iniciar a aplicação web

```bash
run_app.bat
```

> Esse comando ativa o ambiente virtual e inicia a interface web no navegador.

---

## 🧠 Modelos Suportados

Você pode alternar dinamicamente entre os seguintes backends no `.env`:

| Provedor     | Exemplo de valor em `LLM_MODEL_NAME`          |
|--------------|-----------------------------------------------|
| SambaNova    | `DeepSeek-R1-Distill-Llama-70B`               |
| Ollama local | `mistral`, `llama2`, `tinyllama`, `llama3.1:8b`|

---

## 🐳 Qdrant via Docker (alternativa manual)

```bash
docker-compose up -d
```

> Certifique-se de que as portas `6333` e `6334` estão liberadas.

---

## 🚀 Executando a aplicação (modo técnico)

```bash
streamlit run app.py
```

---

## 📁 Estrutura dos Arquivos

- `app.py` – Interface com Streamlit
- `rag_code.py` – Lógica de vetorização, RAG e consulta ao LLM
- `recreate_collection.py` – Script para resetar a coleção do Qdrant
- `.env` – Chaves da API e escolha do modelo
- `assets/` – Logos utilizados na interface
- `run_app.bat` – Script auxiliar para execução no Windows
- `run_docker.bat` – Script auxiliar para iniciar o banco vetorial

---

## 🛠️ Requisitos

- Python 3.11+
- Docker (para Qdrant)
- RAM recomendada: ≥8 GB para modelos locais

---

## 💡 Dica

Se você tiver pouca memória RAM, use modelos como:

```env
LLM_MODEL_NAME=tinyllama
```

E evite modelos como `llama3.1:8b` localmente.

---

## 🤝 Créditos

Adaptado de: https://github.com/patchy631/ai-engineering-hub/tree/main/chat-with-audios?ref=dailydoseofds.com
Com ❤️ por Sandova.
Inspirado por soluções modernas de RAG com áudio e LLMs.
