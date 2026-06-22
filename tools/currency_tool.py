import httpx
from datetime import datetime
import os

EXCHANGE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")
EXCHANGE_BASE_URL = "https://v6.exchangerate-api.com/v6"


class CurrencyTool:
    """
    Fetches live currency exchange rates from ExchangeRate-API.
    Uses synchronous httpx calls so LangGraph sync nodes can call directly.
    """

    def get_rate(self, base: str, target: str) -> dict:
        """Fetch live exchange rate between two currencies."""
        base = base.upper().strip()
        target = target.upper().strip()

        key = os.getenv("EXCHANGE_RATE_API_KEY", EXCHANGE_API_KEY)
        url = f"{EXCHANGE_BASE_URL}/{key}/pair/{base}/{target}"

        response = httpx.get(url, timeout=10.0)

        if response.status_code != 200:
            raise RuntimeError(f"Currency API error {response.status_code}: {response.text}")

        data = response.json()

        if data.get("result") != "success":
            raise RuntimeError(f"Invalid currency pair: {base}/{target}")

        return {
            "base": base,
            "target": target,
            "rate": data["conversion_rate"],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def convert(self, amount: float, base: str, target: str) -> dict:
        """Convert an amount from base currency to target currency."""
        rate_data = self.get_rate(base, target)
        converted = round(amount * rate_data["rate"], 2)

        return {
            "amount": amount,
            "base": rate_data["base"],
            "target": rate_data["target"],
            "rate": rate_data["rate"],
            "converted": converted,
            "timestamp": rate_data["timestamp"],
        }
