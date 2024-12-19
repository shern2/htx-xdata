"""
Main entry point for the application.
"""

import logging
from pathlib import Path
from shlex import split

from app.config import n_workers, pth_model, pth_processor


def setup_data_dir_if_needed():
    """Setup the model artefacts, if needed."""
    if not pth_processor.exists():
        logging.info("Downloading processor")
        from transformers import Wav2Vec2Processor

        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        processor.save_pretrained(pth_processor)
    if not pth_model.exists():
        logging.info("Downloading model")
        from transformers import Wav2Vec2ForCTC

        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
        model.save_pretrained(pth_model)
    logging.info("Setup completed")


def start_process_manager():
    import subprocess

    logging.info(Path.cwd())

    setup_data_dir_if_needed()

    cmd = (
        (
            "gunicorn"
            " src.app.asr_api:app"
            " --bind 0.0.0.0:8001"
            f" --workers {n_workers}"
            " --worker-class uvicorn.workers.UvicornWorker"
            " --timeout 90"
            " --graceful-timeout 10"
            " --keep-alive 2"
        )
        # max requests per worker before restarting (good for memory leaks)
        + ((" --max-requests 20000 --max-requests-jitter 2000") if n_workers > 1 else "")
    )
    subprocess.run(split(cmd), check=True)


if __name__ == "__main__":
    start_process_manager()
