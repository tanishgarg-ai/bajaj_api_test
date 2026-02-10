import os
import math
import asyncio
import time
from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
EMAIL = os.getenv("EMAIL", "your-email@example.com")
API_KEY = os.getenv("GEMINI_KEY")

if not API_KEY:
    raise ValueError("GEMINI_KEY not found in environment variables")

# Initialize the modern Gemini Async Client
# Note: .aio provides the asynchronous interface
client = genai.Client(api_key=API_KEY).aio

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


# --- AI Logic with Retry Mechanism ---

async def generate_ai_response(prompt: str):
    """Generates content using Gemini with exponential backoff retries."""
    retries = 5
    for i in range(retries):
        try:
            response = await client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            if i == retries - 1:
                raise e
            # Exponential backoff: 1s, 2s, 4s, 8s, 16s
            await asyncio.sleep(2 ** i)
    return None


# --- API Endpoints ---

@app.post("/bfhl")
async def bfhl(data: InputData):
    # Get only the fields that were actually sent in the request
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
        # Log the error internally here if needed
        raise HTTPException(status_code=500, detail="Internal Server Error processing the request.")


@app.get("/health")
async def health():
    return {
        "is_success": True,
        "official_email": EMAIL,
        "status": "online"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)