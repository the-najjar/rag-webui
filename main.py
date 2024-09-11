import json
import logging
import os
import webbrowser
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from requests.exceptions import ConnectionError
from urllib3.exceptions import NewConnectionError

from config import ollama_base_api_url
from populate_database import check_if_existing_documents, clear_database, index_documents
from query_data import query_rag

logging.basicConfig(level=logging.INFO)

os.makedirs("data", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logging.info("🧹 Clearing the database...")
    clear_database()
    os.makedirs("chroma", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    logging.info("🚀 FastAPI application is starting up!")
    logging.info("🖥️ Go To http://127.0.0.1:8000 (Ctrl + Click) to open the app in your browser.")
    # automatically open the browser if the the webpage is not already opened
    webbrowser.open("http://127.0.01:8000")
    yield
    # Shutdown logic
    logging.info("🧹 Clearing the database...")
    clear_database()
    logging.info("🛑 FastAPI application is shutting down!")


app = FastAPI(lifespan=lifespan)


# Define a Pydantic model for the request body
class UserPromptReqeustBody(BaseModel):
    prompt: str


app.mount("/data", StaticFiles(directory="data"), name="data")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename
        file_path = os.path.join("data", filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        # Populate the database with the new file
        index_documents()

        return {"filename": filename, "message": "File uploaded successfully", "file_path": file_path}
    except Exception as e:
        return {"error": str(e)}


@app.post("/fetch-response")
async def call_llm_studio_api(request_body: UserPromptReqeustBody):
    async def event_generator():
        try:
            # check connection to ollama server first
            if not os.getenv("OLLAMA_SERVER_URL"):
                yield f"data: {json.dumps({
                    'content': '❗❗ OLLAMA_SERVER_URL is not set. Please set it in the .env file.'
                })}\n\n"
                yield "data: [DONE]\n\n"
                return

            # ping ollama server
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{ollama_base_api_url}", timeout=5.0)
                    response.raise_for_status()
            except httpx.HTTPError as e:
                yield f"data: {json.dumps({
                    'content': f'❗❗ Failed to connect to Ollama server: {str(e)}'
                })}\n\n"
                yield "data: [DONE]\n\n"
                return

            if not check_if_existing_documents():
                yield f"data: {json.dumps({'content': '❗❗ No data available. Please upload a file first.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            try:
                response, sources = query_rag(request_body.prompt)
                # keep only unique sources
                sources = sorted(list(set(sources)))
                yield f"data: {json.dumps({'content': response})}\n\n"
                yield f"data: {json.dumps({'sources': sources})}\n\n"
            except (ConnectionError, NewConnectionError):
                yield f"data: {json.dumps({'content': '❗❗ Please run ollama first.'})}\n\n"
                yield f"data: {json.dumps({'sources': []})}\n\n"
            except Exception as e:
                if "Max retries" in str(e):
                    yield f"data: {json.dumps({
                        'content': '❗❗ Please run ollama first or check OLLAMA_SERVER_URL inside '
                                   '.env file if its matching the ollama server url.'
                    })}\n\n"
                    yield f"data: {json.dumps({'sources': []})}\n\n"
                else:
                    yield f"data: {json.dumps({'content': f'❗❗ Error querying RAG: {str(e)}'})}\n\n"
                    yield f"data: {json.dumps({'sources': []})}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'content': f'❗❗ Unexpected error: {str(e)}'})}\n\n"
            yield f"data: {json.dumps({'sources': []})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
