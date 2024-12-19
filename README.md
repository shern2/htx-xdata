## Pre-requisites / notes

1. Assume you have `docker`, `docker compose` installed (Developed using `Docker version 27.3.1`)
1. Assume you have `conda` installed.
1. Assume you have ~100-150GB additional disk space for training dataset preprocessing.
1. This assessment was run on an Nvidia 2080Ti (11GB GPU RAM). Parameters like `n_workers` in `app/config.py` and training code are tailored to run on it.
1. Assume the Common Voice dataset directory `common_voice/` is located under `$PROJECT_DIR/asr/data/`.
1. If you are using VSCode, it may be beneficial to add the following to your workspace settings: `"files.exclude": {"**/*.mp3":true}`.


## Setup instructions

```bash
PROJECT_DIR='your project/repo directory'
COMMON_VOICE_DATA_DIR='[dir]/common_voice'

# ensure that data directory is present for all tasks
mkdir -p $PROJECT_DIR/asr/data $PROJECT_DIR/asr-train/data $PROJECT_DIR/hotword-detection/data
ln -s COMMON_VOICE_DATA_DIR $PROJECT_DIR/asr/data/common_voice
ln -s COMMON_VOICE_DATA_DIR $PROJECT_DIR/asr-train/data/common_voice
ln -s COMMON_VOICE_DATA_DIR $PROJECT_DIR/hotword-detection/data/common_voice

# For Task 3 onwards
conda create -yn htx python=3.8
conda activate htx
pip install -r $PROJECT_DIR/asr/requirements.txt
python -m ipykernel install --user --name htx


# For Task 5 code setup
ln -s $PROJECT_DIR/asr-train/src $PROJECT_DIR/hotword-detection/src
```

## Running Task 2 (docker-based)

```bash
cd $PROJECT_DIR/asr

docker compose build
docker compose up -d
```

### Task 2b and 2c - inference API testing

Run the test curl command(s) similar to `$PROJECT_DIR/asr/notebooks/test-docker-inference.ipynb`


### Task 2d - batch API inference

For your convenience, the predicted results are in `asr/output/cv-valid-dev_task2d_cv-decode.csv`
```bash
# Will copy the raw dev files to `valid_dev_wip_dir` as defined in `src/app/config.py` first
# As the command runs, notice that the files are deleted as they are processed, as per task 2e.
# The output is located at `pth_valid_dev_wip` with the `generated_text` column, as per task 2d.
# [kiv;debug] docker exec asr-api python3 ./src/run.py
docker exec asr-api python3 /app/src/scripts/cv-decode.py

# Feel free to tune the `n_workers` parameter in `config.py` if your GPU can handle it.

# When ready to stop the container
docker compose stop # or `docker compose down` to remove container
```


### Task 3 - preprocessing and training

1. You will notice may `RUN = False` at the top of multiple cells. You can turn them on to check out the various Exploratory Data Analysis (EDA), processing steps.
1. You may need to update the `PROJECT_DIR` at the top of the notebook for code to find the relevant modules.
1. After training, you will need to update the `best_checkpoint_dir` in `app/config.py` to your desired checkpoint before running the `Test` section.
    1. It will perform task 3b "Rename your fine-tuned AI model: wav2vec2-large-960h-cv".
1. For task 3c, comparison and performance are in the description at the top as well as the `Test` section.


### Task 5b - similarity output

1. A copy of the `cv-valid-dev.csv` with `similarity` column added is in `hotword-detection/output/` folder.

