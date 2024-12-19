from pathlib import Path

python_env = "dev"
# python_env = "uat"
# python_env = "prod"

# number of gunicorn workers
n_workers = 1


# === model params === #
data_dir = Path("/app/data")
pth_processor = data_dir / "processor"
pth_model = data_dir / "model"

common_voice_dir = data_dir / "common_voice"
# wip_dir = data_dir / "wip"

# wip_dir.mkdir(exist_ok=True, parents=True)

valid_dev_raw_dir = common_voice_dir / "cv-valid-dev/cv-valid-dev"
valid_train_raw_dir = common_voice_dir / "cv-valid-train/cv-valid-train"
valid_test_raw_dir = common_voice_dir / "cv-valid-test/cv-valid-test"
pth_valid_dev_raw = common_voice_dir / "cv-valid-dev.csv"
pth_valid_train_raw = common_voice_dir / "cv-valid-train.csv"
pth_valid_test_raw = common_voice_dir / "cv-valid-test.csv"

# valid_dev_wip_dir = wip_dir / "cv-valid-dev/cv-valid-dev"
# pth_valid_dev_wip = wip_dir / "cv-valid-dev.csv"


# audio sampling rate
sampling_rate = 16000
