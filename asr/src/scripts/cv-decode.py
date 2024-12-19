import asyncio
import logging
from pathlib import Path
from typing import Tuple

import httpx
import pandas as pd
from app.config import pth_valid_dev_raw, valid_dev_raw_dir
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)

logging.getLogger().setLevel(logging.INFO)

max_concurrency = 7
semaphore = asyncio.Semaphore(max_concurrency)
ahttp = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=int(max_concurrency * 1.5), max_keepalive_connections=max_concurrency),
    timeout=40,
)


@retry(
    stop=stop_after_attempt(14),
    wait=wait_exponential_jitter(0.2, max=7),
    before_sleep=before_sleep_log(logging, logging.INFO),
    reraise=True,
)
async def get_transcription(row: pd.Series) -> Tuple[str, str]:
    """Query the ASR service for the transcription of the audio file

    Args:
        row (pd.Series): a row from the dataframe with the filename

    Returns:
        (filename, transcription): a tuple with the filename and the transcription
    """
    pth = valid_dev_raw_dir.parent / row["filename"]
    async with semaphore:
        resp = await ahttp.post(
            "http://localhost:8001/asr",
            files={"file": (pth.as_posix(), pth.open("rb"), "audio/mpeg")},
        )
    resp.raise_for_status()
    return row["filename"], resp.json()["transcription"]


def backup_file(pth: Path):
    pth_bkp = pth.with_suffix(f"{pth.suffix}.bak")
    if not pth_bkp.exists():
        from shutil import copyfile

        copyfile(pth, pth_bkp)
    return pth_bkp


async def process_files():
    backup_file(pth_valid_dev_raw)
    df = pd.read_csv(pth_valid_dev_raw)
    assert df["filename"].duplicated().sum() == 0, "Duplicated filenames in the dataset, pls check"

    tpls = await asyncio.gather(*[get_transcription(row) for _, row in df.iterrows()])
    df_pred = pd.DataFrame(tpls, columns=["filename", "generated_text"])
    df_out = df.merge(df_pred, on="filename")

    # Save the output
    df_out.to_csv(pth_valid_dev_raw, index=False)


def main():

    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_files())


if __name__ == "__main__":
    main()
