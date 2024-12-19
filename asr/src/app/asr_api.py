"""
Main app file
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
from app.config import python_env
from app.model import ASRModel
from fastapi import FastAPI, Request, Response, UploadFile

# TODO move model to torchserve handler to have more efficient batching for GPU
model = None
is_model_ready = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    logging.info("[lifespan] Load model")
    model = ASRModel()
    is_model_ready.set()

    yield
    logging.info("[lifespan] Shutting down app")
    model = None


app = FastAPI(
    lifespan=lifespan,
    docs_url="/docs" if python_env == "dev" else None,
    redoc_url="/redoc" if python_env == "dev" else None,
)


@app.get("/ping")
async def ping():
    return "pong"


@app.get("/healthz")
async def health():
    if model is None:
        return Response(json.dumps({"status": "NOT OK"}), status_code=httpx.codes.SERVICE_UNAVAILABLE)
    return Response(json.dumps({"status": "OK"}), status_code=httpx.codes.OK)


@app.post("/asr")
async def asr(file: UploadFile):
    await is_model_ready.wait()
    content = await file.read()
    loop = asyncio.get_event_loop()
    out = await loop.run_in_executor(None, model, content, file.filename.split(".")[-1])
    return Response(out.model_dump_json(), status_code=httpx.codes.OK)
