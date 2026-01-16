
import sys
import os
import json

# Setup Path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.core.ai_brain import AIBrain
from app.utils import config

def simulate_deepseek_call():
    print("🚀 Iniciando Simulação de Validação DeepSeek...")
    
    # 1. Load API Key
    api_key = None
    config_path = os.path.join("config", "user_config_roe.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
            api_key = data.get('deepseek_key')
    
    if not api_key:
        print("❌ Erro: API Key não encontrada em config/user_config_roe.json")
        print("   Por favor configure a chave no App primeiro.")
        return

    print(f"🔑 API Key carregada: {api_key[:5]}...{api_key[-5:]}")

    # 2. Mock Technical Context (Full Data)
    technical_summary = {
        'rsi': 28.5,           # Oversold
        'stoch_rsi_k': 0.15,   # Oversold
        'atr': 125.50,         # High Volatility
        'macd': -150.2,
        'bb_width': 0.035,     # Wide Bands
        'adx': 45.2,           # Strong Trend
        'obv_slope': 1500.5,   # Positive Volume
        'dist_support': 0.05,  # Near Support (0.05%)
        'dist_resistance': 2.5,
        'pattern_score': 2.0,  # Bullish Patterns
        'trend_score': -1.5,   # Bearish Trend
        'pattern_triangle': 0,
        'is_pivot_high': 0,
        'is_pivot_low': 1,     # Just made a low
        'fib_500': 1.2
    }

    market_context = {
        "regime": "ALTA VOLATILIDADE",
        "trend": "BAIXA (Curto Prazo)",
        "hint": "Scalping de Reversão"
    }

    current_price = 95000.0
    predicted_price = 95250.0 # +0.26%
    
    print("\n📦 Payload de Dados (Simulado):")
    print(json.dumps(market_context, indent=2))
    print(json.dumps(technical_summary, indent=2))
    
    # 3. Call Brain
    brain = AIBrain() # Init (might try to load model, ignore warnings)
    
    print("\n🤖 Enviando Request para DeepSeek API...")
    try:
        response = brain.validate_signal_with_deepseek(
            current_price,
            predicted_price,
            technical_summary,
            market_context=market_context,
            api_key=api_key
        )
        
        print("\n✅ RESPOSTA RECEBIDA:")
        print(json.dumps(response, indent=4, ensure_ascii=False))
        
        if response.get('approved'):
            print("\n🎉 Conclusão: Trade APROVADO pela IA.")
        else:
            print(f"\n✋ Conclusão: Trade REJEITADO. Razão: {response.get('reason')}")
            
    except Exception as e:
        print(f"\n❌ Erro durante a chamada: {e}")

if __name__ == "__main__":
    simulate_deepseek_call()
