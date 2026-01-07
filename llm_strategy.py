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

    def generate_strategies(self, indicators: List[str], count: int = 3, market_regime: str = "SIDEWAYS", description: str = "", current_data_summary: Dict = {}, existing_strategies: List[str] = [], trade_history: List[Dict] = []) -> List[Dict[str, Any]]:
        """
        Generates trading strategies using DeepSeek LLM, tailored to the specific Market Regime and Context.
        - current_data_summary: Technicals (RSI, Volatility, etc.)
        - existing_strategies: List of logic strings to avoid duplicates.
        - trade_history: Performance context to learn from.
        """
        if not self.api_key:
            return []

        print(f"🧠 [DeepSeek] Solicitando {count} estratégias para regime: {market_regime}...")

        system_prompt = """You are a Senior Crypto Quantitative Trader with 20 years of experience.
Your goal is to generate high-alpha 'scalping' strategies for Bitcoin (1m timeframe) that work specifically in the current Market Environment.

You must output ONLY valid JSON. The format must be a list of strategies.
Each strategy has:
- "trend_mode": The input market regime (e.g., UPTREND, DOWNTREND, SIDEWAYS, HIGH_VOL).
- "conditions": A list of 2 or 3 conditions.
Each condition has:
- "indicator": Must be one from the provided list.
- "operator": ">" or "<"
- "threshold": A float value appropriate for that indicator.

Rules:
- **AVOID DUPLICATES**: Do not suggest strategies that are listed in the 'Exclusion List'.
- **LEARN FROM HISTORY**: If the 'Trade History' shows frequent losses with certain indicators (e.g. RSI), try a different approach (e.g. Trend Following).
- **ADAPT TO DATA**: Use the provided 'Current Market Metrics' to set realistic thresholds (e.g. if current RSI is 80, don't suggest RSI > 90 as it's too rare).

- For UPTREND: Focus on 'Buy the Dip' or Momentum Breakout. (Indicators: RSI < 30 (dip) or ADX > 25 (trend))
- For DOWNTREND: Focus on 'Sell the Rally' or Momentum Breakdown.
- For SIDEWAYS: Focus on Mean Reversion (RSI > 70 to Sell, RSI < 30 to Buy, Bollinger Bands).
- For HIGH_VOL: Focus on Breakouts or wide Mean Reversion.

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

        # Format Context
        exclusion_text = "\n- ".join(existing_strategies[:50]) if existing_strategies else "None"
        history_text = str(trade_history[-10:]) if trade_history else "None"
        metrics_text = json.dumps(current_data_summary, indent=2) if current_data_summary else "Not Provided"

        user_content = f"""
Current Market Context:
- Regime: {market_regime}
- Description: {description}
- **Market Metrics**: 
{metrics_text}

**Performance Context**:
- Recent Trade History: {history_text}

**Exclusion List (Do NOT create these)**:
- {exclusion_text}

Available Indicators: {', '.join(indicators)}

Generate {count} distinct, professional scalping strategies specifically optimized for the metrics above.
For 'dist_support' and 'dist_resistance', values are usually 0.00 to 0.10 (0% to 10%).
For 'adx_slope', values are usually -5 to 5.
For 'pattern_marubozu', threshold is 0.5.
Think like a institutional algorithm. Avoid known failures.
"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 1200,
            "temperature": 1.1 # High creativity for alpha generation
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # Clean formatting (sometimes LLM adds markdown)
            content = content.replace("```json", "").replace("```", "").strip()
            
            strategies = json.loads(content)
            print(f"🧠 [DeepSeek] Sugeriu {len(strategies)} estratégias para {market_regime}.")
            return strategies

        except Exception as e:
            print(f"❌ Erro na API DeepSeek: {e}")
            return []

if __name__ == "__main__":
    # Test
    strategist = DeepSeekStrategist()
    test_inds = ['rsi', 'adx', 'dist_ema_200', 'pattern_marubozu']
    print(strategist.generate_strategies(test_inds, 1, "UPTREND", "Strong Bullish Impulse"))
