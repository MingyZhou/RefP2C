import os

from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "")

openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
