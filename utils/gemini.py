from google import genai
from google.genai import types
import os

client = genai.Client()

def generate_chat_title(question: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction="""Generate a short and clean 3–5 word title summarizing this question.
    The title must:
    - be concise
    - not contain punctuation
    - not contain quotes
    - be title-case (e.g., Visa Requirements USA)
    """,
            ),
            contents=question,
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating title: {e}")
        return "New Chat"
