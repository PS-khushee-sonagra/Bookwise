import os
import sys

# Add project root to sys.path to allow importing app
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from google import genai

from app.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model="gemini-flash-latest",
    contents="Say Hello"
)

print(response.text)
