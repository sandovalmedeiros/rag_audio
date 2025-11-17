# CLAUDE.md - AI Assistant Guide for RAG Audio Project

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) system for audio files** that enables users to ask questions about transcribed audio content. The system combines:

- **Audio Transcription**: AssemblyAI for automatic transcription with speaker identification
- **Vector Storage**: Qdrant for high-performance vector database
- **LLM Integration**: Dynamic support for SambaNova Cloud (DeepSeek models) and Ollama (local models)
- **Web Interface**: Streamlit for interactive chat interface
- **Embeddings**: HuggingFace sentence transformers for multilingual support (Portuguese-focused)

### Key Features

- Upload audio files (MP3, WAV, M4A)
- Automatic transcription with speaker diarization (identifies different speakers)
- Vector-based semantic search over transcribed content
- Chat interface for querying audio content
- Streaming responses from LLMs
- Session-based file caching for performance

---

## Architecture

### Data Flow

```
Audio File → AssemblyAI (Transcription) → Speaker Segments →
→ HuggingFace Embeddings → Qdrant Vector DB →
→ RAG Query → LLM (SambaNova/Ollama) → Streaming Response
```

### Core Components

1. **Transcribe** (`rag_code_pt.py:166-204`): Handles audio transcription via AssemblyAI
2. **EmbedData** (`rag_code_pt.py:16-34`): Generates embeddings using HuggingFace models
3. **QdrantVDB_QB** (`rag_code_pt.py:36-81`): Manages Qdrant vector database operations
4. **Retriever** (`rag_code_pt.py:82-101`): Performs semantic search over vector database
5. **RAG** (`rag_code_pt.py:103-164`): Orchestrates retrieval and LLM response generation

### Technology Stack

```python
# Core RAG Framework
llama-index==0.12.35
llama-index-embeddings-huggingface
llama-index-llms-sambanovasystems
llama-index-llms-ollama

# Vector Database
qdrant-client

# Audio Transcription
assemblyai

# Embeddings & ML
transformers
sentence-transformers
torch

# Web Interface
streamlit (implicitly used)
```

---

## File Structure

```
/home/user/rag_audio/
├── app_pt.py                    # Main Streamlit application (Portuguese)
├── rag_code_pt.py              # Core RAG logic and classes
├── recreate_collection.py      # Utility to reset Qdrant collection
├── test_sambanova.py           # SambaNova connection testing script
├── debug_audio.py              # AssemblyAI transcription debugging
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── docker-compose.yml          # Qdrant container configuration
├── README.md                   # User documentation (Portuguese)
└── assets/                     # UI assets (logos)
    ├── AssemblyAI.png
    ├── deepseek.png
    └── llhama.jpg
```

### File Descriptions

#### `app_pt.py` (Main Application)
- **Purpose**: Streamlit web interface for the RAG system
- **Key Functions**:
  - `reset_chat()`: Clears chat history and context
- **Session State**:
  - `id`: Unique session identifier
  - `file_cache`: Caches processed files to avoid reprocessing
  - `messages`: Chat history
  - `transcripts`: Stores audio transcriptions
- **Workflow**:
  1. File upload via sidebar
  2. Transcription with AssemblyAI
  3. Embedding generation
  4. Vector storage in Qdrant
  5. Chat interface for queries

#### `rag_code_pt.py` (Core Logic)
- **Purpose**: Contains all RAG components and business logic
- **Classes**:
  - `EmbedData`: Embedding generation using HuggingFace
  - `QdrantVDB_QB`: Vector database management
  - `Retriever`: Semantic search functionality
  - `RAG`: Query processing and LLM integration
  - `Transcribe`: Audio transcription via AssemblyAI
- **Key Configuration**:
  - Default embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - Vector dimension: Auto-detected from model
  - Distance metric: DOT product
  - Quantization: Binary quantization for performance

#### `recreate_collection.py`
- **Purpose**: Utility script to reset the Qdrant collection
- **Usage**: Run when you need to clear all vectors and start fresh
- **Collection Name**: `chat_com_audios`

#### `test_sambanova.py`
- **Purpose**: Validates SambaNova Cloud configuration
- **Features**:
  - Tests API key validity
  - Verifies model availability
  - Suggests alternative models if primary fails
- **Recommended Models**:
  - `DeepSeek-R1-Distill-Llama-70B` (default)
  - `Meta-Llama-3.1-8B-Instruct`
  - `Meta-Llama-3.3-70B-Instruct`

#### `debug_audio.py`
- **Purpose**: Test AssemblyAI transcription independently
- **Usage**: Debug transcription issues without running full app

---

## Environment Configuration

### Required Environment Variables

```env
# AssemblyAI API Key (required for transcription)
ASSEMBLYAI_API_KEY=your_assemblyai_key_here

# SambaNova API Key (required for cloud LLM)
SAMBANOVA_API_KEY=your_sambanova_key_here

# LLM Model Selection
LLM_MODEL_NAME=DeepSeek-R1-Distill-Llama-70B
```

### LLM Model Selection Logic

The system automatically detects the LLM provider based on `LLM_MODEL_NAME`:

1. **Ollama** (local): If model name contains "ollama" or starts with "ollama:"
   - Example: `ollama:mistral`, `llama3.1:8b`, `tinyllama`
   - Configured with low VRAM settings for resource-constrained environments

2. **SambaNova Cloud** (default): All other model names
   - Example: `DeepSeek-R1-Distill-Llama-70B`, `Meta-Llama-3.1-8B-Instruct`
   - Higher context window (32K tokens)

**Code Location**: `rag_code_pt.py:126-148`

---

## Development Workflows

### Initial Setup

```bash
# 1. Clone repository (already done in this environment)
cd /home/user/rag_audio

# 2. Create virtual environment (if needed)
python -m venv .venv
source .venv/bin/activate  # Linux
# or: .venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with actual API keys

# 5. Start Qdrant vector database
docker-compose up -d

# 6. Verify SambaNova configuration (optional)
python test_sambanova.py

# 7. Run the application
streamlit run app_pt.py
```

### Running the Application

```bash
# Start Qdrant (if not already running)
docker-compose up -d

# Run Streamlit app
streamlit run app_pt.py
```

### Resetting Vector Database

```bash
# If you need to clear all stored vectors
python recreate_collection.py
```

### Testing Components

```bash
# Test AssemblyAI transcription
python debug_audio.py

# Test SambaNova connection
python test_sambanova.py
```

---

## Key Code Conventions

### Language
- **Primary Language**: Portuguese (variable names, comments, UI text)
- **Code Structure**: English-based library/framework conventions
- **Files**: `_pt.py` suffix indicates Portuguese version

### Coding Patterns

1. **Class-Based Architecture**: Each major component is a class
2. **Dependency Injection**: Classes receive dependencies via `__init__`
3. **Batch Processing**: Data is processed in batches for efficiency
4. **Session State**: Streamlit session state for caching and persistence
5. **Error Handling**: Try-except blocks with user-friendly error messages

### Vector Database Conventions

```python
# Collection name is hardcoded
collection_name = "chat_com_audios"

# Batch size for embeddings
batch_size = 32  # For embeddings
batch_size = 512  # For Qdrant uploads

# Vector configuration
distance_metric = models.Distance.DOT
quantization = models.BinaryQuantization  # For performance
on_disk = True  # For large datasets
```

### Embedding Strategy

- **Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Rationale**: Multilingual support with focus on Portuguese
- **Cache Location**: `./hf_cache` (gitignored)
- **Document Format**: `"Locutor {speaker}: {text}"`

### RAG Configuration

```python
# Retrieval: Top 2 most relevant segments
top_k = 2  # Implicit in rag_code_pt.py:154

# LLM Parameters
temperature = 0.7
context_window = 32000  # SambaNova
context_window = 8192   # Ollama

# Prompt Template: Portuguese, step-by-step reasoning
# Location: rag_code_pt.py:118-124
```

---

## Common AI Assistant Tasks

### 1. Adding Support for a New LLM Provider

**Location**: `rag_code_pt.py:126-148`

```python
# Extend _setup_llm() method in RAG class
def _setup_llm(self):
    if self.llm_name.startswith("new_provider:"):
        # Add your provider configuration
        return NewProviderLLM(model=self.llm_name, ...)
    elif self.llm_name.startswith("ollama:"):
        # Existing Ollama logic
        ...
```

### 2. Changing Embedding Model

**Locations**:
- `app_pt.py:57` (instantiation)
- `recreate_collection.py:8` (for collection reset)

**Important**: Vector dimensions must match. If changing models:
1. Update `embed_model_name` in both files
2. Run `recreate_collection.py` to reset vectors
3. Reprocess all audio files

### 3. Adjusting Retrieval Parameters

**Location**: `rag_code_pt.py:87-101`

```python
# To retrieve more contexts:
# Change line 154 in generate_context():
for entry in context[:5]:  # Changed from [:2]
```

**Location**: `rag_code_pt.py:154`

### 4. Modifying Prompt Template

**Location**: `rag_code_pt.py:118-124`

Current prompt instructs step-by-step reasoning in Portuguese. Modify `qa_prompt_tmpl_str` for different behavior.

### 5. Adding New Audio Format Support

**Location**: `app_pt.py:31`

```python
# Add format to accepted types
uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio",
    type=["mp3", "wav", "m4a", "flac", "ogg"]  # Add new formats
)
```

**Note**: AssemblyAI supports most common audio formats. Check their documentation for full list.

### 6. Customizing Speaker Diarization

**Location**: `rag_code_pt.py:182-186`

```python
config = aai.TranscriptionConfig(
    speaker_labels=True,
    speakers_expected=3,  # Change expected speaker count
    language_code="pt"    # Change language
)
```

### 7. Improving Streaming Response Display

**Location**: `app_pt.py:176-184`

Current implementation extracts content from streaming chunks. Modify error handling and chunk processing as needed.

---

## Important Implementation Notes

### Session Management

The app uses Streamlit session state for caching:
- **File Cache Key**: `f"{session_id}-{filename}"`
- **Cache Contents**: Entire `query_engine` (RAG instance)
- **Benefit**: Avoids re-transcription and re-embedding on page refresh

**Location**: `app_pt.py:15-18, 44-82`

### Collection Recreation

The Qdrant collection is **deleted and recreated** on each new file upload:

**Location**: `rag_code_pt.py:47-49`

```python
if self.client.collection_exists(collection_name=self.collection_name):
    self.client.delete_collection(self.collection_name)
```

**Implication**: Only one audio file can be stored at a time per collection. For multi-file support, this needs to be modified.

### Docker Configuration

**Important**: The docker-compose file has a hardcoded Windows path:

**Location**: `docker-compose.yml:11`

```yaml
volumes:
  - "C:/docker_composes/qdrant_storage/qdrant_storage:/qdrant/storage"
```

**For Linux/Mac**: Change to appropriate path or remove for ephemeral storage.

### Portuguese Language Focus

- All UI text is in Portuguese
- Embedding model is multilingual but optimized for Portuguese
- AssemblyAI is configured with `language_code="pt"`
- Prompt templates are in Portuguese

**When modifying**: Consider updating all three layers (UI, prompts, language config) for consistency.

### Performance Considerations

1. **Binary Quantization**: Enabled for Qdrant to reduce memory usage
   - **Location**: `rag_code_pt.py:62-64`

2. **On-Disk Storage**: Vectors stored on disk, not just RAM
   - **Location**: `rag_code_pt.py:56`

3. **Batch Processing**: Embeddings generated in batches of 32
   - **Location**: `app_pt.py:21, rag_code_pt.py:11-14`

4. **Low VRAM Mode**: Ollama configured for resource-constrained environments
   - **Location**: `rag_code_pt.py:136-140`

### Error Handling Patterns

```python
# Pattern 1: User-facing errors with st.error
try:
    # operation
except Exception as e:
    st.error(f"Ocorreu um erro: {e}")
    st.stop()

# Pattern 2: Debug prints for transcription
try:
    transcript = ...
except Exception as e:
    print(f"❌ Erro na transcrição: {e}")
    raise
```

---

## Testing Strategy

### Manual Testing Checklist

1. **Audio Upload**
   - [ ] Test each supported format (MP3, WAV, M4A)
   - [ ] Test with multi-speaker audio
   - [ ] Test with single speaker
   - [ ] Test with different languages (if changing from Portuguese)

2. **Transcription**
   - [ ] Verify speaker labels are correct
   - [ ] Check transcription accuracy
   - [ ] Verify Portuguese language detection

3. **RAG Queries**
   - [ ] Ask factual questions about content
   - [ ] Test with queries requiring multiple segments
   - [ ] Verify "I don't know" responses for out-of-scope questions

4. **LLM Providers**
   - [ ] Test SambaNova with DeepSeek model
   - [ ] Test Ollama with local model
   - [ ] Verify streaming responses work

5. **Edge Cases**
   - [ ] Upload same file twice (test caching)
   - [ ] Clear chat and start new conversation
   - [ ] Test with very long audio (>1 hour)
   - [ ] Test with very short audio (<30 seconds)

### Unit Testing (Not Currently Implemented)

Consider adding tests for:
- `batch_iterate()` function
- `EmbedData.embed()` method
- `Retriever.search()` functionality
- Prompt template formatting

---

## Common Pitfalls & Solutions

### 1. Dimension Mismatch Error

**Error**: Vector dimension doesn't match collection configuration

**Cause**: Changed embedding model without recreating collection

**Solution**:
```bash
python recreate_collection.py
# Then reprocess audio files
```

### 2. Qdrant Connection Error

**Error**: Cannot connect to Qdrant at localhost:6333

**Cause**: Docker container not running

**Solution**:
```bash
docker-compose up -d
# Verify with:
docker ps | grep qdrant
```

### 3. AssemblyAI API Error

**Error**: 401 Unauthorized or API key error

**Cause**: Missing or invalid `ASSEMBLYAI_API_KEY`

**Solution**: Verify `.env` file has correct key from http://bit.ly/4bGBdux

### 4. SambaNova Model Not Available

**Error**: Model not found or not accessible

**Cause**: Model may not be in your tier or API key invalid

**Solution**:
```bash
python test_sambanova.py
# Follow suggestions for alternative models
```

### 5. Out of Memory (OOM)

**Cause**: Large audio files or resource-intensive models

**Solutions**:
- Use smaller LLM: `LLM_MODEL_NAME=tinyllama` (Ollama)
- Reduce batch size in `app_pt.py:21`
- Enable Ollama low VRAM mode (already default)

### 6. Streaming Response Not Working

**Symptom**: Response appears all at once instead of streaming

**Cause**: LLM provider may not support streaming or error in chunk parsing

**Debug**: Check `app_pt.py:178-184` try-except block for exceptions

---

## Git Workflow

### Current Branch
- **Development Branch**: `claude/claude-md-mi3hx3fr1kfj138t-016psywKmmYoVREf3BKen3vg`
- **Main Branch**: Not specified (repository root)

### Gitignore Highlights

```gitignore
# Environments
.venv/, venv/, env/

# Sensitive data
.env

# ML Cache
hf_cache/

# Archives
*.zip, *.log
```

### Commit Message Conventions

Based on recent commits:
- Use descriptive messages: "Update README.md", "Update .env.example"
- Reference specific files when appropriate
- Keep commits focused and atomic

---

## Future Improvements

### Suggested Enhancements

1. **Multi-File Support**
   - Store multiple audio files in different Qdrant collections
   - Add file selection dropdown in UI

2. **Conversation Memory**
   - Integrate chat history into RAG context
   - Add conversational follow-up capabilities

3. **Advanced Speaker Analysis**
   - Speaker identification (not just labels)
   - Speaker emotion/sentiment analysis

4. **Export Functionality**
   - Export transcripts as TXT/PDF
   - Export Q&A history

5. **Performance Monitoring**
   - Track response times
   - Monitor token usage
   - Add usage analytics

6. **Testing Suite**
   - Unit tests for core functions
   - Integration tests for RAG pipeline
   - CI/CD pipeline

7. **Internationalization**
   - Support for multiple UI languages
   - Easy language switching

8. **Authentication**
   - User login system
   - Per-user file storage
   - API key management UI

---

## External Resources

### Documentation Links

- **AssemblyAI**: https://www.assemblyai.com/docs
- **Qdrant**: https://qdrant.tech/documentation
- **LlamaIndex**: https://docs.llamaindex.ai
- **Streamlit**: https://docs.streamlit.io
- **SambaNova**: https://sambanova.ai/
- **HuggingFace Sentence Transformers**: https://www.sbert.net

### API Key Registration

- **AssemblyAI**: http://bit.ly/4bGBdux
- **SambaNova**: https://sambanova.ai/

### Original Project Credit

Adapted from: https://github.com/patchy631/ai-engineering-hub/tree/main/chat-with-audios

---

## Quick Reference

### File Paths
```
Main App: /home/user/rag_audio/app_pt.py
Core Logic: /home/user/rag_audio/rag_code_pt.py
Environment: /home/user/rag_audio/.env
Dependencies: /home/user/rag_audio/requirements.txt
```

### Port Configuration
```
Qdrant API: 6333
Qdrant gRPC: 6334
Streamlit: Auto-assigned (typically 8501)
```

### Default Configuration
```python
COLLECTION_NAME = "chat_com_audios"
BATCH_SIZE = 32  # embeddings
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "DeepSeek-R1-Distill-Llama-70B"
TOP_K = 2  # retrieved segments
TEMPERATURE = 0.7
```

---

## AI Assistant Guidelines

When working on this codebase:

1. **Preserve Language**: Maintain Portuguese for user-facing content
2. **Test Changes**: Use `test_sambanova.py` and `debug_audio.py` before full app testing
3. **Update Documentation**: If changing core functionality, update this CLAUDE.md
4. **Consider Performance**: Audio files and embeddings are resource-intensive
5. **Respect Privacy**: Audio files may contain sensitive information
6. **Check Dependencies**: Verify compatibility when updating LlamaIndex or other core libraries
7. **Environment First**: Always check `.env` configuration before troubleshooting
8. **Collection Management**: Remember that collections are recreated on each upload

---

**Last Updated**: 2025-11-17
**Project Version**: Based on commit `4cf38bb`
