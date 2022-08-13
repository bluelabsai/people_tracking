from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
STORE_DIR = Path(BASE_DIR, "store")
MODELS_DIR = Path(STORE_DIR, "models")

# Create dir
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

URL_MODELS = {
    "yolov3": "1ku-iE-0V4Rcd9rrF28SCbZnAnQoa1jgp",
    "mobilenet_v2": "1a7U05ttb693hx7CR82Sn__P-t_NIUX2B"
}