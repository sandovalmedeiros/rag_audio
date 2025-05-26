#!/usr/bin/env python3
"""
Script para testar a configuração da SambaNova Cloud
Execute este script antes de usar o app principal para verificar se está tudo funcionando.
"""

import os
from dotenv import load_dotenv
from llama_index.llms.sambanovasystems import SambaNovaCloud

def test_sambanova_connection():
    """Testa a conexão com a SambaNova Cloud"""
    
    print("🔍 Testando configuração da SambaNova Cloud...")
    
    # Carrega variáveis de ambiente
    load_dotenv()
    
    # Verifica se as chaves estão configuradas
    sambanova_key = os.getenv("SAMBANOVA_API_KEY")
    assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY")
    model_name = os.getenv("LLM_MODEL_NAME", "DeepSeek-R1-Distill-Llama-70B")
    
    print(f"📋 Modelo configurado: {model_name}")
    
    if not sambanova_key:
        print("❌ ERRO: SAMBANOVA_API_KEY não encontrada no arquivo .env")
        print("   Adicione sua chave da SambaNova no arquivo .env")
        return False
        
    if not assemblyai_key:
        print("⚠️  AVISO: ASSEMBLYAI_API_KEY não encontrada no arquivo .env")
        print("   Você precisará desta chave para transcrever áudios")
    
    print("✅ Chave da SambaNova encontrada")
    
    try:
        # Testa conexão com SambaNova
        print("🔗 Testando conexão com SambaNova Cloud...")
        
        llm = SambaNovaCloud(
            model=model_name,
            temperature=0.7,
            context_window=32000,
        )
        
        # Teste simples
        response = llm.complete("Olá! Este é um teste de conexão. Responda apenas 'Conexão bem-sucedida!'")
        
        print("✅ Conexão com SambaNova Cloud bem-sucedida!")
        print(f"📝 Resposta do modelo: {response.text}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO na conexão com SambaNova Cloud: {e}")
        print("\n🔧 Possíveis soluções:")
        print("1. Verifique se sua chave SAMBANOVA_API_KEY está correta")
        print("2. Certifique-se de que tem créditos na sua conta SambaNova")
        print("3. Verifique se o modelo está disponível em sua tier")
        print("4. Tente um modelo diferente (ex: Meta-Llama-3.1-8B-Instruct)")
        
        return False

def test_model_alternatives():
    """Sugere modelos alternativos caso o principal não funcione"""
    
    print("\n🔄 Modelos alternativos disponíveis na SambaNova Cloud:")
    
    alternative_models = [
        "Meta-Llama-3.1-8B-Instruct",      # Modelo menor e confiável
        "Meta-Llama-3.3-70B-Instruct",     # Modelo Llama mais recente
        "DeepSeek-V3-0324",                # DeepSeek mais rápido
        "Meta-Llama-3.2-3B-Instruct",      # Modelo muito leve
    ]
    
    for i, model in enumerate(alternative_models, 1):
        print(f"{i}. {model}")
    
    print("\n💡 Para usar um modelo alternativo:")
    print("   Edite o arquivo .env e altere a linha:")
    print(f"   LLM_MODEL_NAME={alternative_models[0]}")

if __name__ == "__main__":
    print("🚀 TESTE DE CONFIGURAÇÃO - SAMBANOVA CLOUD")
    print("=" * 50)
    
    success = test_sambanova_connection()
    
    if not success:
        test_model_alternatives()
        print("\n❌ Teste falhou. Corrija os problemas acima antes de continuar.")
    else:
        print("\n🎉 Configuração está correta! Você pode usar o app principal.")
        print("\n📝 Próximos passos:")
        print("1. Execute: streamlit run app_pt.py")
        print("2. Faça upload de um arquivo de áudio")
        print("3. Comece a conversar sobre o conteúdo!")
    
    print("\n" + "=" * 50)