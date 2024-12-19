"""
The data model
"""

import asyncio
from typing import List, Tuple

import torch
from app.config import pth_model, pth_processor, sampling_rate
from pydantic import BaseModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class ASROutput(BaseModel):
    transcription: str
    # [ss] a bit odd to be string type, but as requested by the task
    duration: str
