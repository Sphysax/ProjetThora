import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from train_v1_common import (
    BEST_MODEL_PATH,
    DEVICE,
    model,
    train_dataset,
    valid_transforms,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_UNIQUE_TEST_DIR = BASE_DIR /"archive"/ "unique_test"


def load_image(image_path: Path) -> torch.Tensor:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        image_tensor = valid_transforms(img)
        return image_tensor.unsqueeze(0)


def find_latest_image_file(folder_path: Path) -> Path:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
    image_files = [
        child for child in folder_path.iterdir()
        if child.is_file() and child.suffix.lower() in extensions
    ]

    if not image_files:
        raise FileNotFoundError(f"No valid image found in folder: {folder_path}")

    latest_image = max(image_files, key=lambda p: p.stat().st_mtime)
    return latest_image


def predict_image(image_path: Path) -> None:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model checkpoint not found: {BEST_MODEL_PATH}\n"
            "Run training first to create best_pneumonia_model.pth."
        )

    if image_path.is_dir():
        image_path = find_latest_image_file(image_path)

    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print(f"Loading image: {image_path}")
    image_tensor = load_image(image_path).to(DEVICE)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)
        prediction_idx = torch.argmax(probabilities).item()
        predicted_label = train_dataset.classes[prediction_idx]

    print("\nPrediction")
    print("-" * 30)
    print(f"Image          : {image_path}")
    print(f"Predicted class: {predicted_label}")
    print("Probabilities:")

    for idx, label in enumerate(train_dataset.classes):
        print(f"  {label:12}: {probabilities[idx].item():.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a single image using the best pneumonia model."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_UNIQUE_TEST_DIR,
        help=(
            "Optional path to an image file or a folder containing one unique test "
            "image. If omitted, defaults to the 'unique test' folder."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_image(args.path)
