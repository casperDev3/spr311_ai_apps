from typing import Union
from fastapi import FastAPI
from main import run_ollama

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/ollama/")
def ollama_endpoint(request: dict):
    prompt = request["prompt"]
    model = request["model"]
    response = run_ollama(prompt, model=model)
    if response is None:
        return {
            "success": False,
            "status_code": 500,
            "error": "Failed to get response from Ollama."
        }
    return {
        "success": True,
        "status_code": 200,
        "response": response
    }


@app.get("/health/")
def health_check():
    return {"status": "ok"}