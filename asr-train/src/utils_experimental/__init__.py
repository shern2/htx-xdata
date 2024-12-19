"""Experimental utilities..."""

import logging
from pathlib import Path
from typing import List

import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier  # type: ignore
from torch.utils.data import DataLoader


class AccentClassifier:

    def __init__(self, model_save_dir: str):
        self.classifier = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
            savedir=model_save_dir,
            run_opts={"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"},
        )

    def load_and_preprocess_audio(self, pth: Path, sample_rate: int = 16000):
        """
        Load and preprocess an audio file.
        Converts it to the required sampling rate and returns the waveform.
        """
        # Load audio file
        waveform, sr = torchaudio.load(pth)

        # Resample if needed
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform

    def batch_predict_accent(self, pths: List[Path]) -> List[str]:
        """
        Perform batch accent prediction on a list of audio file paths
        """
        # Preprocess all audio files
        waveforms = [self.load_and_preprocess_audio(file).squeeze(0) for file in pths]

        # Pad waveforms to the same length for batching
        lengths = [wave.shape[0] for wave in waveforms]
        max_length = max(lengths)
        padded_waveforms = torch.stack(
            [torch.nn.functional.pad(wave, (0, max_length - len(wave))) for wave in waveforms]
        )

        # Perform batch prediction
        predictions = self.classifier.classify_batch(padded_waveforms)
        predicted_labels = predictions[3]  # Output includes predicted labels at index 3

        return predicted_labels


def greedy_n_lm_score_val_loss(
    val_loader: DataLoader,
    w_lm_score: float,
) -> float:

    # Pre-requisite
    # Install kenlm e.g.
    # MAX_ORDER=xxx pip install https://github.com/kpu/kenlm/archive/master.zip
    # References:
    # https://github.com/kpu/kenlm/blob/master/README.md#python-module
    # https://huggingface.co/blog/wav2vec2-with-ngram (may be outdated)

    import kenlm  # type: ignore
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    lm_path = "common_crawl_5gram.arpa"  # TODO replace
    lm_model = kenlm.Model(lm_path)

    preds = []
    labels = []
    val_loss = 0.0
    for batch in val_loader:
        with torch.no_grad():
            out = model(batch["input_values"].to(device), labels=batch["labels"].to(device))

        # Greedy decoding
        predicted_ids = torch.argmax(out.logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        # Assume higher is better
        # lm_score = lm_model.score(transcription)
        # acoustic loss + w_lm_score * LM loss
        val_loss += out.loss.item() + w_lm_score * lm_score
        preds.extend(transcription)
        labels.extend(processor.batch_decode(batch["labels"], group_tokens=False))

    val_loss /= len(val_loader)

    return val_loss


def beam_search_and_lm_scoring():

    # Pre-requisite
    # Install kenlm e.g.
    # MAX_ORDER=xxx pip install https://github.com/kpu/kenlm/archive/master.zip
    # References:
    # https://github.com/kpu/kenlm/blob/master/README.md#python-module
    # https://huggingface.co/blog/wav2vec2-with-ngram (may be outdated)

    import kenlm  # type: ignore
    import torch
    from ctcdecode import CTCBeamDecoder  # type: ignore
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    # Example audio input (16kHz, 1 second)
    input_values = torch.randn(1, 16000)  # TODO replace

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    lm_path = "common_crawl_5gram.arpa"  # TODO replace
    lm_model = kenlm.Model(lm_path)

    labels = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(labels.keys(), key=lambda k: labels[k])  # Sort vocab by index

    # TODO tune?
    decoder = CTCBeamDecoder(
        labels=sorted_vocab,
        beam_width=10,  # Number of beams
        alpha=0.5,  # LM weight
        beta=1.0,  # Word insertion penalty
        num_processes=4,  # Parallel decoding
    )

    with torch.no_grad():
        logits = model(input_values).logits  # Shape: (batch_size, seq_len, vocab_size)

    beam_results, beam_scores, timesteps, out_lens = decoder.decode(logits)

    for i in range(beam_results.size(0)):
        beams = []
        for beam_id in range(beam_results.size(1)):
            sequence = beam_results[i][beam_id][: out_lens[i][beam_id]].tolist()
            transcription = "".join([sorted_vocab[idx] for idx in sequence])
            beams.append((transcription, beam_scores[i][beam_id].item()))

        # Score hypotheses using KenLM
        scored_beams = []
        for transcription, beam_score in beams:
            lm_score = lm_model.score(transcription)
            combined_score = beam_score + lm_score  # Combine acoustic and LM scores
            scored_beams.append((transcription, combined_score))

        # Sort by combined score
        best_hypothesis = sorted(scored_beams, key=lambda x: x[1], reverse=True)[0]
        print(f"Best Transcription: {best_hypothesis[0]}, Combined Score: {best_hypothesis[1]}")
