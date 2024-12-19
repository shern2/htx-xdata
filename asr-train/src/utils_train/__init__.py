"""Utilities for training"""

import logging
from typing import List, Tuple

import torch
from app.config import sampling_rate
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batch_predict(
    data_loader: DataLoader,
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    device: str,
    include_loss: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """Run inference on a dataloader and return predictions and labels.

    Args:
        data_loader (DataLoader): The DataLoader to use for inference.
        model (Wav2Vec2ForCTC): The model to use for inference.
        processor (Wav2Vec2Processor): The processor to use for inference.
        device (str): The device to use for inference.
        include_loss (bool): Whether to include the loss in the output.

    Returns:
        if include_loss=False,
        (preds, labels, filenames): Tuple of lists of predictions, labels, and filenames.
        otherwise
        (preds, labels, filenames, val_loss): Tuple of lists of predictions, labels, filenames, validation loss.
    """
    preds = []
    labels = []
    filenames = []
    model.eval()

    val_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            with torch.amp.autocast("cuda"):
                out = model(batch["input_values"].to(device), labels=batch["labels"].to(device))
                if not torch.isnan(out.loss) and not torch.isinf(out.loss):
                    val_loss += out.loss.item()
                else:
                    logging.warning("Encountered NaN or Inf in validation loss. Skipping this batch.")
                    try:
                        logging.warning(f"batch['filename']: {batch['filename']}")
                    except Exception:
                        logging.warning("cannot print batch['filename']", exc_info=True)
                predicted_ids = torch.argmax(out.logits, dim=-1)
                preds.extend(processor.batch_decode(predicted_ids))
                labels.extend(processor.batch_decode(batch["labels"], group_tokens=False))
    val_loss /= len(data_loader)

    if include_loss:
        return preds, labels, filenames, val_loss
    return preds, labels, filenames


class CommonVoiceDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        processor: Wav2Vec2Processor,
        batch_size=32,
        initial_random_seed=77,
    ):
        """
        Args:
            dataset (Dataset): The dataset to be loaded.
            processor (Wav2Vec2Processor): Processor object for feature extraction and tokenization.
            batch_size (int, optional): The batch size to yield.
            initial_random_seed (int, optional): The initial random seed for shuffling the dataset across epochs.
                Every subsequent DataLoader yielded will have a different shuffle order (deterministic), affected by this seed.
        """
        self.dataset = dataset
        self.processor = processor
        self.batch_size = batch_size
        self.random_seed = initial_random_seed
        self.generator = torch.Generator()

    def collate_fn(self, batch):

        inputs = self.processor(
            [itm["audio"] for itm in batch],
            [itm["text"] for itm in batch],
            return_tensors="pt",
            padding="longest",
            sampling_rate=sampling_rate,
        )

        # inputs["input_lengths"] = torch.tensor([input.shape[0] for input in inputs["input_values"]], dtype=torch.int32)
        # inputs["target_lengths"] = torch.tensor([len(itm["text"]) for itm in batch], dtype=torch.int32)

        # just for metadata
        inputs["filename"] = [itm["filename"] for itm in batch]

        return inputs

    def get_dataloader(
        self,
        shuffle=True,
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ) -> DataLoader:
        """Gets a new DataLoader object for the dataset.
        If `shuffle` is enabled, every subsequent DataLoader yielded will have a different shuffle order, determined by the `initial_random_seed`.

        Args:
            shuffle (bool, optional): Whether to shuffle the dataset.
            num_workers (int, optional): Number of worker processes to use.
            prefetch_factor (int, optional): `prefetch_factor * num_workers` batches prefetched across all workers
        """
        self.generator.manual_seed(self.random_seed)
        self.random_seed += 1  # Increment random seed for next epoch

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            prefetch_factor=prefetch_factor,
            num_workers=num_workers,
            generator=self.generator,
        )
