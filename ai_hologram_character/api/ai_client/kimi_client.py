import os
from openai import OpenAI

class KimiClient:
    """KimiClient is a client for the Kimi API."""

    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(
          api_key=os.getenv("MOONSHOT_API_KEY"),
          base_url="http://127.0.0.1:9988/v1" #os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
        )

    