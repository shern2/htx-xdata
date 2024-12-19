from pathlib import Path

python_env = "dev"
# python_env = "uat"
# python_env = "prod"

# number of gunicorn workers
n_workers = 1


# === model params === #
data_dir = Path("./asr-train/data").absolute()
pth_processor = data_dir / "processor"
pth_model = data_dir / "model"
pth_finetuned_model = data_dir / "wav2vec2-large-960h-cv"

common_voice_dir = data_dir / "common_voice"
# wip_dir = data_dir / "wip"

# wip_dir.mkdir(exist_ok=True, parents=True)

valid_dev_raw_dir = common_voice_dir / "cv-valid-dev/cv-valid-dev"
valid_train_raw_dir = common_voice_dir / "cv-valid-train/cv-valid-train"
valid_test_raw_dir = common_voice_dir / "cv-valid-test/cv-valid-test"
pth_valid_dev_raw = common_voice_dir / "cv-valid-dev.csv"
pth_valid_train_raw = common_voice_dir / "cv-valid-train.csv"
pth_valid_test_raw = common_voice_dir / "cv-valid-test.csv"

ds_dev_dir = data_dir / "ds_dev"
ds_train_dir = data_dir / "ds_train"
ds_val_dir = data_dir / "ds_val"
ds_test_dir = data_dir / "ds_test"

# Optuna trials directory
trials_dir = data_dir / "trials"

# Directory for analysis
analysis_dir = data_dir / "analysis"
pth_df_wer_dev = analysis_dir / "df_dev_wer.csv"
pth_df_wer_test = analysis_dir / "df_test_wer.csv"

analysis_dir.mkdir(exist_ok=True, parents=True)

best_checkpoint_dir = trials_dir / "trial-0/model/0.13299"  # NOTE: Update this to your best model directory
final_model_dir = data_dir / "wav2vec2-large-960h-cv"

pth_hotwords_txt = Path("./hotword-detection/output/detected.txt")
pth_hotwords_txt.parent.mkdir(exist_ok=True, parents=True)

pth_analyze_largest_wer_diff = Path("./asr-train/output/analyze_largest_wer_diff.csv")

# audio sampling rate
sampling_rate = 16000
