from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import math
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.getenv("EMAIL")
API_KEY = os.getenv("GEMINI_KEY")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-pro")

app = FastAPI()


class InputData(BaseModel):
    fibonacci: Union[int, None] = None
    prime: Union[List[int], None] = None
    lcm: Union[List[int], None] = None
    hcf: Union[List[int], None] = None
    AI: Union[str, None] = None


def fib(n):
    res = [0, 1]
    for i in range(2, n):
        res.append(res[-1] + res[-2])
    return res[:n]


def primes(arr):
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(math.sqrt(x)) + 1):
            if x % i == 0:
                return False
        return True

    return [x for x in arr if is_prime(x)]


def lcm_arr(arr):
    lcm = arr[0]
    for i in arr[1:]:
        lcm = lcm * i // math.gcd(lcm, i)
    return lcm


def hcf_arr(arr):
    h = arr[0]
    for i in arr[1:]:
        h = math.gcd(h, i)
    return h


@app.post("/bfhl")
async def bfhl(data: InputData):

    keys = [k for k, v in data.dict().items() if v is not None]

    if len(keys) != 1:
        raise HTTPException(400, "Only one key allowed")

    try:

        if data.fibonacci is not None:
            result = fib(data.fibonacci)

        elif data.prime is not None:
            result = primes(data.prime)

        elif data.lcm is not None:
            result = lcm_arr(data.lcm)

        elif data.hcf is not None:
            result = hcf_arr(data.hcf)

        elif data.AI is not None:
            response = model.generate_content(data.AI)
            result = response.text.strip()

        return {
            "is_success": True,
            "official_email": EMAIL,
            "data": result
        }

    except Exception:
        raise HTTPException(500, "Internal Server Error")


@app.get("/health")
def health():
    return {
        "is_success": True,
        "official_email": EMAIL
    }
