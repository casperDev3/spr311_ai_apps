from typing import Union
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
from main import run_ollama

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Додаємо CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальна змінна для відстеження запущених задач
active_tasks = {}


@app.get("/")
def read_root():
    return {"Hello": "World"}


# Додаємо обидва варіанти - з слешем і без
@app.post("/ollama")
async def ollama_endpoint(request: dict):
    return await process_ollama_request(request)


@app.post("/ollama/")
async def ollama_endpoint_with_slash(request: dict):
    return await process_ollama_request(request)


async def process_ollama_request(request: dict):
    try:
        prompt = request.get("prompt")
        model = request.get("model")

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        if not model:
            model = "llama2"  # default model

        logger.info(f"Processing request - Model: {model}, Prompt: {prompt[:100]}...")

        # Запускаємо в окремому потоці, щоб не блокувати event loop
        try:
            # Використовуємо asyncio.to_thread для запуску синхронної функції
            response = await asyncio.to_thread(run_ollama, prompt, model)

            if response is None:
                logger.error("Ollama returned None response")
                return {
                    "success": False,
                    "status_code": 500,
                    "error": "Failed to get response from Ollama."
                }

            logger.info("Successfully processed Ollama request")
            return {
                "success": True,
                "status_code": 200,
                "response": response
            }

        except Exception as e:
            logger.error(f"Error in Ollama processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Ollama processing error: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Асинхронна версія для довгих запитів
@app.post("/ollama-async/")
async def ollama_async_endpoint(request: dict, background_tasks: BackgroundTasks):
    try:
        prompt = request.get("prompt")
        model = request.get("model", "llama2")

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        # Для асинхронної обробки повертаємо відразу, а обробку робимо в фоні
        task_id = str(hash(f"{prompt}{model}"))

        return {
            "success": True,
            "status_code": 202,
            "message": "Request accepted for processing",
            "task_id": task_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/health/")
def health_check_with_slash():
    return {"status": "ok"}


# Додаємо endpoint для перевірки статусу задачі
@app.get("/task-status/{task_id}")
def get_task_status(task_id: str):
    task_result = active_tasks.get(task_id)
    if task_result:
        return {
            "success": True,
            "status": "completed",
            "result": task_result
        }
    else:
        return {
            "success": True,
            "status": "processing"
        }