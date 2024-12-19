"""
The ML model
"""

import asyncio
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from app.config import pth_model, pth_processor, sampling_rate
from app.data_model import ASROutput
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logger = logging.getLogger(__name__)


class ASRModel:
    """The Automatic Speech Recognition model"""

    def __init__(
        self,
        min_audio_duration__remove_silences: int = 12.5,
        pth_processor: Path = pth_processor,
        pth_model: Path = pth_model,
    ):
        """
        Args:
            min_audio_duration__remove_silences (int): The threshold removing silence regions
                (to tackle GPU OOM error for long audio files)
            pth_processor (Path): The path to the pretrained processor directory
            pth_model (Path): The path to the pretrained model directory

        """
        self.min_audio_duration__remove_silences = min_audio_duration__remove_silences

        self.processor = Wav2Vec2Processor.from_pretrained(pth_processor)
        self.model = Wav2Vec2ForCTC.from_pretrained(pth_model)

        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = self.model.to(self.device)
            self.predict = self.predict_gpu
        else:
            self.device = "cpu"
            self.predict = self.predict_cpu

    def predict(self, file: str) -> ASROutput:
        """dummy"""
        pass

    def get_raw_mono_0dBFS_audio(self, file: bytes, file_type: str) -> AudioSegment:
        """Returns the raw audio as a mono AudioSegment with peak amplitude of 0 dBFS"""
        audio = AudioSegment.from_file(BytesIO(file), format=file_type)
        # Convert to mono if not already
        if audio.channels > 1:
            audio = audio.set_channels(1)
        audio = audio.apply_gain(-audio.max_dBFS)
        return audio

    def extract_utterances(
        cls,
        audio: AudioSegment,
        silence_thresh: int = -40,
        min_silence_len: int = 1000,
    ) -> AudioSegment:
        """Extracts the utterances from the audio segment

        Args:
            audio (AudioSegment): the audio segment
            silence_thresh (int, optional): the silence threshold in dBFS.
            min_silence_len (int, optional): the minimum silence length in ms.

        Returns:
            The audio segment with only the utterances
        """

        # Detect non-silent segments [(start, end), ...]
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
        )
        utterances = [audio[start:end] for start, end in nonsilent_ranges]
        # case: last segment is a little cut off
        # -> we add 1 second of audio after the last segment
        if nonsilent_ranges:
            start, end = nonsilent_ranges[-1]
            utterances.append(audio[end : end + int(1e3)])
        combined_audio = sum(utterances)
        return combined_audio

    def extract_1st_utterance(
        cls,
        audio: AudioSegment,
        silence_thresh: int = -40,
        min_silence_len: int = 1000,
        max_clip_duration: int = int(15e3),
    ) -> AudioSegment:
        """Extracts the utterances from the audio segment

        Args:
            audio (AudioSegment): the audio segment
            silence_thresh (int, optional): the silence threshold in dBFS.
            min_silence_len (int, optional): the minimum silence length in ms.
            max_clip_duration (int, optional): the maximum expected duration of the clip in ms, for the Common Voice dataset

        Returns:
            The audio segment with only the first utterance
        """

        # Detect non-silent segments [(start, end), ...]
        nonsilent_ranges = detect_nonsilent(
            audio[:max_clip_duration],
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
        )
        utterances = [audio[start:end] for start, end in nonsilent_ranges]
        # case: last segment is a little cut off
        # -> we add 1 second of audio after the last segment
        if nonsilent_ranges:
            start, end = nonsilent_ranges[-1]
            utterances.append(audio[end : end + int(1e3)])
        combined_audio = sum(utterances)
        return combined_audio

    def preprocess__common_voice(
        self,
        pth: Path,
        file_type: str,
    ) -> Tuple[Union[AudioSegment, None], Union[np.ndarray, None]]:
        """Preprocesses the audio located at `pth` to output the audio segment and the numpy array of the audio.

        Args:
            pth (Path): The path to the audio file
            file_type (str): The file type (e.g. 'mp3')

        Returns:
            (audio, np_audio): The audio's audio segment and normalized numpy array ([-1,1])
        """
        audio = self.get_raw_mono_0dBFS_audio(pth.read_bytes(), file_type)
        audio = self.extract_1st_utterance(audio)
        # case: all silence within the `max_clip_duration`
        if audio == 0:
            return None, None

        # set sampling rate
        audio = audio.set_frame_rate(sampling_rate)

        np_audio = np.array(audio.get_array_of_samples(), dtype=np.float32)
        # Normalize to [-1, 1]
        np_audio /= 2 ** (8 * audio.sample_width - 1)  # Normalize based on sample width

        return audio, np_audio

    def preprocess(self, file: bytes, file_type: str) -> Tuple[str, float]:
        """Preprocesses the audio `file` to get the input tensor for the model and the audio duration

        Returns:
            A tuple of (audio_tensor, duration)
        """
        audio = self.get_raw_mono_0dBFS_audio(file, file_type)

        # TODO [ss] the proper way is to split by silence and process each segment separately (perhaps batched)
        # But, I take the easy way out for now
        if audio.duration_seconds > self.min_audio_duration__remove_silences:
            logger.warning(
                f"[ASRModel] Audio duration {audio.duration_seconds} > {self.min_audio_duration__remove_silences}. Removing silences"
            )
            audio = self.extract_utterances(audio)

        # set sampling rate
        audio = audio.set_frame_rate(sampling_rate)

        np_audio = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # Normalize to [-1, 1]
        np_audio /= 2 ** (8 * audio.sample_width - 1)  # Normalize based on sample width

        return np_audio, audio.duration_seconds

    def __call__(self, file: bytes, file_type: str) -> ASROutput:
        """Does ASR on the audio `file`

        Args:
            file: The audio file
            file_type: The file type
        """

        audio_tensor, duration = self.preprocess(file, file_type)
        transcription = self.predict(audio_tensor)
        return ASROutput(transcription=transcription, duration=str(duration))

    def predict_gpu(self, audio_tensor: np.ndarray) -> str:
        """(GPU) Does ASR on the audio `file`
        NOTE: Not very efficient for batch processing as-is. Need batching service like TorchServe

        Args:
            audio_tensor: The audio tensor, normalized to [-1,1], has shape [n_samples]

        Returns:
            The transcription.
        """
        input_values = self.processor(
            audio_tensor,
            return_tensors="pt",
            padding="longest",
            sampling_rate=sampling_rate,
        ).input_values
        input_values = input_values.cuda()
        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription

    def predict_cpu(self, audio_tensor: np.ndarray) -> str:
        """(CPU) Does ASR on the audio `file`"""
        input_values = self.processor(
            audio_tensor,
            return_tensors="pt",
            padding="longest",
            sampling_rate=sampling_rate,
        ).input_values
        logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription

    # @classmethod
    # def compute_duration(cls, input_values: torch.Tensor) -> float:
    #     """Assumes `input_values` is a [1,n_samples], sampled at `sampling_rate`"""
    #     return input_values.shape[1] / sampling_rate
