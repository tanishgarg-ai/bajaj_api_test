import os
import math
import asyncio
import traceback
import re
from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Config
EMAIL = os.getenv("EMAIL", "your-email@example.com")
API_KEY = os.getenv("GEMINI_KEY")

# Check API Key immediately
if not API_KEY:
    print("CRITICAL ERROR: GEMINI_KEY is missing from environment variables.")
    client = None
else:
    try:
        client = genai.Client(api_key=API_KEY).aio
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Gemini Client: {e}")
        client = None

app = FastAPI(title="Math & AI API")


class InputData(BaseModel):
    fibonacci: Optional[int] = None
    prime: Optional[List[int]] = None
    lcm: Optional[List[int]] = None
    hcf: Optional[List[int]] = None
    AI: Optional[str] = None


# --- Math Utility Functions ---

def calculate_fibonacci(n: int) -> List[int]:
    if n <= 0: return []
    if n == 1: return [0]
    res = [0, 1]
    for i in range(2, n):
        res.append(res[-1] + res[-2])
    return res[:n]


def filter_primes(arr: List[int]) -> List[int]:
    def is_prime(x: int) -> bool:
        if x < 2: return False
        for i in range(2, int(math.sqrt(x)) + 1):
            if x % i == 0: return False
        return True

    return [x for x in arr if is_prime(x)]


def calculate_lcm(arr: List[int]) -> int:
    if not arr: return 0
    lcm = arr[0]
    for i in arr[1:]:
        if i == 0: continue
        lcm = abs(lcm * i) // math.gcd(lcm, i)
    return lcm


def calculate_hcf(arr: List[int]) -> int:
    if not arr: return 0
    h = arr[0]
    for i in arr[1:]:
        h = math.gcd(h, i)
    return h



async def generate_ai_response(prompt: str):
    """Generates content with fallback from 2.0-flash to 1.5-flash on quota errors."""
    if client is None:
        raise HTTPException(status_code=503, detail="Gemini Client not initialized.")

    # Try 2.0 first, then 1.5 if 2.0 is exhausted
    models_to_try = ["gemini-3-flash-preview", "gemini-2.0-flash"]
    last_exception = None

    # System instruction to enforce the one-word rule
    system_instruction = "You are a helpful assistant. You must respond to every user query with exactly one word. No punctuation, no sentences, just one single word."

    for model_name in models_to_try:
        try:
            response = await client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "system_instruction": system_instruction
                }
            )
            # Ensure we only return the first word if the model happens to ignore the instruction
            result = response.text.strip().split()
            return result[0] if result else "N/A"

        except Exception as e:
            err_str = str(e)
            # If it's a rate limit error, we check if we should try the next model
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"Quota exhausted for {model_name}. Trying fallback...")
                last_exception = e
                continue
            raise e  # Raise immediately if it's a different kind of error (e.g. invalid key)

    # If all models fail, raise the last quota error
    raise last_exception


# --- API Endpoints ---

@app.post("/bfhl")
async def bfhl(data: InputData):
    provided_fields = data.model_dump(exclude_unset=True)

    if len(provided_fields) != 1:
        raise HTTPException(status_code=400, detail="Exactly one operation key must be provided.")

    key = list(provided_fields.keys())[0]
    result = None

    try:
        if key == "fibonacci":
            result = calculate_fibonacci(data.fibonacci)
        elif key == "prime":
            result = filter_primes(data.prime)
        elif key == "lcm":
            result = calculate_lcm(data.lcm)
        elif key == "hcf":
            result = calculate_hcf(data.hcf)
        elif key == "AI":
            result = await generate_ai_response(data.AI)

        return {
            "is_success": True,
            "official_email": EMAIL,
            "data": result
        }

    except Exception as e:
        err_msg = str(e)

        # Specific handling for Quota/Rate Limit Errors
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            # Try to extract the wait time from the error message using regex
            wait_match = re.search(r"retry in ([\d\.]+)s", err_msg)
            wait_time = wait_match.group(1) if wait_match else "a few seconds"

            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Please retry in {wait_time}s. Consider switching to a paid plan or waiting for the quota to reset."
            )

        print("--- EXCEPTION CAUGHT IN /BFHL ---")
        traceback.print_exc()
        print("---------------------------------")

        raise HTTPException(status_code=500, detail=f"Internal Error: {err_msg}")


@app.get("/health")
async def health():
    return {
        "is_success": True,
        "official_email": EMAIL,
        "status": "online",
        "gemini_ready": client is not None
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
