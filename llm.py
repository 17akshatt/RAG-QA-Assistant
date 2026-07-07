# A small, reusable wrapper around the Gemini API call.
#
# Every other file in this project that needs to talk to the model will
# import ask() from here, instead of repeating the "create a client, send a
# prompt" code over and over.

import random
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Load GEMINI_API_KEY from the .env file into the environment.
load_dotenv()

# One client, reused for every call the app makes.
client = genai.Client()

MODEL_NAME = "gemini-2.5-flash"

# Gemini error codes worth retrying: 503 (model overloaded/unavailable),
# 500 (internal server error), and 429 (rate limited). These are usually
# temporary blips, unlike a 400 (bad request), which will never succeed
# no matter how many times we retry it.
RETRYABLE_CODES = {429, 500, 503}
MAX_RETRIES = 5
BASE_DELAY_SECONDS = 1


def ask(prompt, system_instruction=None, temperature=0.7):
    """
    Send a prompt to Gemini and return its text reply.

    prompt: the question or text you want the model to respond to.
    system_instruction: optional rules/role for the model to follow
        (e.g. "You are a pirate" or "Answer in one sentence").
    temperature: how random the reply is. 0.0 = focused and consistent,
        1.0+ = more varied and creative. Default 0.7 is a balanced middle.

    If Gemini returns a transient error (like "503 UNAVAILABLE" when the
    model is overloaded), this retries automatically with an increasing
    delay between attempts ("exponential backoff") instead of crashing.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
            return response.text
        except errors.APIError as e:
            is_last_attempt = attempt == MAX_RETRIES
            if e.code not in RETRYABLE_CODES or is_last_attempt:
                raise

            # Exponential backoff: 1s, 2s, 4s, 8s... plus a little random
            # jitter so we don't hammer the API at the exact same instant
            # if multiple requests are retrying together.
            delay = BASE_DELAY_SECONDS * (2**attempt) + random.uniform(0, 1)
            print(
                f"Gemini returned {e.code} {e.status} — retrying in "
                f"{delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(delay)
