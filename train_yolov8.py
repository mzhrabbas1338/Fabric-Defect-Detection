import os
from ultralytics import YOLO
import glob

# -------- CONFIGURATION --------
DATA_CONFIG_PATH = "fabric.yaml"
BASE_MODEL = "yolov8s.pt"   # Use a larger model for better learning
EPOCHS = 100                # More epochs for small datasets
IMAGE_SIZE = 416
BATCH_SIZE = 8              # Lower batch size if you have little data or low RAM
RUN_NAME = "fabric_detect"
PROJECT_DIR = "runs/train"
OVERFIT_DEBUG = False       # Set True to overfit on 5 images for debugging

def check_data_integrity():
    import yaml
    with open(DATA_CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)
    for split in ['train', 'val']:
        img_dir = data[split]
        lbl_dir = img_dir.replace("images", "labels")
        img_files = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
        lbl_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
        if len(img_files) == 0:
            raise RuntimeError(f"No images found in {img_dir}")
        if len(lbl_files) == 0:
            raise RuntimeError(f"No label files found in {lbl_dir}")
        for img_path in img_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base + ".txt")
            if not os.path.exists(lbl_path):
                print(f"Warning: No label for image {img_path}")
        print(f"{split.capitalize()} set: {len(img_files)} images, {len(lbl_files)} label files.")

def train_model():
    if not os.path.exists(DATA_CONFIG_PATH):
        raise FileNotFoundError(f"YAML config not found at: {DATA_CONFIG_PATH}")

    print("[INFO] Checking data integrity...")
    check_data_integrity()

    print(f"[INFO] Starting YOLOv8 training with model: {BASE_MODEL}")

    model = YOLO(BASE_MODEL)

    # Overfit debug mode: use only 5 images for quick check
    overrides = {}
    if OVERFIT_DEBUG:
        overrides = {
            "epochs": 50,
            "imgsz": IMAGE_SIZE,
            "batch": 1,
            "data": {
                "train": sorted(glob.glob("train/images/*.jpg"))[:5],
                "val": sorted(glob.glob("train/images/*.jpg"))[:5]
            }
        }
        print("[DEBUG] Overfit mode: training on 5 images only.")

    model.train(
        data=DATA_CONFIG_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name=RUN_NAME,
        project=PROJECT_DIR,
        patience=20,
        lr0=0.001,           # Lower initial LR for stability
        optimizer="auto",    # Let YOLO choose the best optimizer
        cos_lr=True,         # Cosine LR schedule
        close_mosaic=10,     # Close mosaic after 10 epochs for better fine-tuning
        **overrides
    )

    print(f"[INFO] Training complete. Results saved in '{PROJECT_DIR}/{RUN_NAME}'")

if __name__ == "__main__":
    train_model()