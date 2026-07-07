# A tiny script to prove our Gemini API key + SDK setup works.
# It sends one sentence to the model and prints whatever it replies.

from dotenv import load_dotenv
from google import genai

# Load the GEMINI_API_KEY from the .env file into the environment
# so the line below can find it.
load_dotenv()

# Create a client. It automatically reads GEMINI_API_KEY from the environment.
client = genai.Client()

# Send one prompt to the model and get a response back.
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in one sentence.",
)

# Print just the text of the model's reply.
print(resp.text)
