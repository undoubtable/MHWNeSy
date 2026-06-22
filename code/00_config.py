from pathlib import Path
import os

# Project root.
# Default: parent directory of this code folder.
# Optional override:
#   export MHWNEURRL_ROOT=/path/to/MHWNeurRL
ROOT = Path(os.environ.get("MHWNEURRL_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CODE_DIR = ROOT / "code"
LEGACY_CODE_DIR = CODE_DIR / "legacy"

RAW_NC = DATA_DIR / "oisst_scs_1982_2023.nc"

LABEL_DIR = OUTPUT_DIR / "01_mhw_labels"
FIGURE_DIR = OUTPUT_DIR / "02_label_visualization"
FORECAST_DIR = OUTPUT_DIR / "03_forecast_dataset_h10_l5"
UNET_RUN_DIR = OUTPUT_DIR / "04_unet_baseline_h10_l5"
EVENT_DIR = OUTPUT_DIR / "06_neurrl_event_dataset_h10_l5"

LABEL_NC = LABEL_DIR / "mhw_labels_strict_hobday_1982_2023.nc"
UNET_BEST = UNET_RUN_DIR / "best_model.pt"

for d in [
    OUTPUT_DIR,
    LABEL_DIR,
    FIGURE_DIR,
    FORECAST_DIR,
    UNET_RUN_DIR,
    EVENT_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
