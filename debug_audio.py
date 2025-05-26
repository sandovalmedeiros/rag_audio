import os
from dotenv import load_dotenv
import assemblyai as aai
from typing import List

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Lê a chave da API
api_key = os.getenv("ASSEMBLYAI_API_KEY")
if not api_key:
    raise ValueError("❌ A variável ASSEMBLYAI_API_KEY não foi encontrada no .env")

aai.settings.api_key = api_key
transcriber = aai.Transcriber()

# Altere esse caminho para apontar para o seu arquivo de teste
AUDIO_PATH = "podcast_v2_1747933463.wav"

def transcribe_audio(audio_path: str) -> List[str]:
    config = aai.TranscriptionConfig(
        speaker_labels=True,
        speakers_expected=2
    )
    try:
        print(f"🔍 Transcrevendo: {audio_path}")
        transcript = transcriber.transcribe(audio_path, config=config)
    except Exception as e:
        print(f"❌ Erro na transcrição: {e}")
        return []

    print("✅ Transcrição concluída. Segmentos por locutor:\n")
    for utt in transcript.utterances:
        print(f"[Speaker {utt.speaker}]: {utt.text}")
    return [utt.text for utt in transcript.utterances]

# Executa o teste
if __name__ == "__main__":
    if not os.path.exists(AUDIO_PATH):
        print(f"❌ Arquivo não encontrado: {AUDIO_PATH}")
    else:
        transcribe_audio(AUDIO_PATH)
