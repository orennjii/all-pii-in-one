import os
from google import genai
from google.genai import types
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for model_info in client.models.list():
    print(model_info.name)