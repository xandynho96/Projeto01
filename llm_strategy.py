import requests
import json
import os
import config
from typing import List, Dict, Any

class DeepSeekStrategist:
    def __init__(self):
        self.api_key = config.DEEPSEEK_API_KEY
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.model = "deepseek-chat"
        
        if not self.api_key:
            print("⚠️ Chave de API DeepSeek ausente. Geração de estratégia por IA desativada.")

    def generate_strategies(self, indicators: List[str], count: int = 3) -> List[Dict[str, Any]]:
        """
        Generates trading strategies using DeepSeek LLM.
        Returns a list of dictionaries representing strategies.
        """
        if not self.api_key:
            return []

        print(f"🧠 Solicitando ao DeepSeek {count} estratégias matadoras baseadas em {len(indicators)} indicadores...")

        system_prompt = """You are an expert Quant Trader AI. Your goal is to generate profitable 'scalping' strategies for Bitcoin (1m timeframe).
You must output ONLY valid JSON. The format must be a list of strategies.
Each strategy has:
- "trend_mode": One of ["UPTREND", "DOWNTREND", "SIDEWAYS", "HIGH_VOL", null]
- "conditions": A list of 2 or 3 conditions.
Each condition has:
- "indicator": Must be one from the provided list.
- "operator": ">" or "<"
- "threshold": A float value appropriate for that indicator.

Example JSON Output:
[
  {
    "trend_mode": "UPTREND",
    "conditions": [
      {"indicator": "rsi", "operator": "<", "threshold": 30},
      {"indicator": "dist_ema_200", "operator": ">", "threshold": 0.01}
    ]
  }
]
"""

        user_content = f"""
Available Indicators: {', '.join(indicators)}

Generate {count} distinct, high-probability scalping strategies.
For 'dist_support' and 'dist_resistance', values are usually 0.00 to 0.10 (0% to 10%).
For 'adx_slope', values are usually -5 to 5.
For 'pattern_marubozu', threshold is usually 0.5 (boolean-like).
Make them aggressive but logical.
"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 1000,
            "temperature": 1.1 # High creativity
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # Clean formatting (sometimes LLM adds markdown)
            content = content.replace("```json", "").replace("```", "").strip()
            
            strategies = json.loads(content)
            print(f"🧠 DeepSeek sugeriu {len(strategies)} estratégias.")
            return strategies

        except Exception as e:
            print(f"❌ Erro na API DeepSeek: {e}")
            return []

if __name__ == "__main__":
    # Test
    strategist = DeepSeekStrategist()
    test_inds = ['rsi', 'adx', 'dist_ema_200', 'pattern_marubozu']
    print(strategist.generate_strategies(test_inds, 1))
