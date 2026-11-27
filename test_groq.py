import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# 1. Check active models on your account
models_resp = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=30,
)
print("Models status:", models_resp.status_code)
print("Available IDs:", [m["id"] for m in models_resp.json().get("data", [])])

# 2. Test chat completion with an active model ID
chat_resp = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "Reply with: Groq API is working"}],
        "temperature": 0.2,
    },
    timeout=60,
)
print("Chat status:", chat_resp.status_code)
print(chat_resp.json()["choices"][0]["message"]["content"])