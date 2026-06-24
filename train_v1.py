import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import densenet121, DenseNet121_Weights

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# ==================================================
# GPU INFO
# ==================================================

print("=" * 50)
print("PyTorch Pneumonia Training")
print("=" * 50)

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

# ==================================================
# CONFIG
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "archive" / "chest_xray"

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_DIR}"
    )

print("\nDataset:", DATA_DIR)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# ==================================================
# PATHS
# ==================================================

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

for p in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not p.exists():
        raise FileNotFoundError(f"Missing folder: {p}")

# ==================================================
# TRANSFORMS
# ==================================================

weights = DenseNet121_Weights.DEFAULT

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==================================================
# DATASETS
# ==================================================

train_dataset = ImageFolder(
    root=str(TRAIN_DIR),
    transform=train_transforms
)

val_dataset = ImageFolder(
    root=str(VAL_DIR),
    transform=valid_transforms
)

test_dataset = ImageFolder(
    root=str(TEST_DIR),
    transform=valid_transforms
)

print("\nClasses:", train_dataset.classes)

print("Train Images:", len(train_dataset))
print("Val Images  :", len(val_dataset))
print("Test Images :", len(test_dataset))

# ==================================================
# DATALOADERS
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==================================================
# MODEL
# ==================================================

model = densenet121(weights=weights)

model.classifier = nn.Linear(
    model.classifier.in_features,
    2
)

model = model.to(DEVICE)

print("\nDenseNet121 loaded successfully.")

# ==================================================
# LOSS + OPTIMIZER
# ==================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==================================================
# TRAIN
# ==================================================

def train_one_epoch():

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# ==================================================
# EVALUATION
# ==================================================

def evaluate(loader):

    model.eval()

    predictions = []
    targets = []

    total_loss = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )

            targets.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    return total_loss / len(loader), accuracy

# ==================================================
# TRAINING LOOP
# ==================================================

def run_training():
    best_accuracy = 0.0

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch()
        val_loss, val_accuracy = evaluate(val_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_pneumonia_model.pth")
            print(">>> Best model saved!")

# ==================================================
# TEST
# ==================================================

def run_test():
    print("\nLoading best model...")
    model.load_state_dict(
        torch.load("best_pneumonia_model.pth", map_location=DEVICE)
    )

    test_loss, test_accuracy = evaluate(test_loader)

    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)
    print(f"Test Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_accuracy:.4f}")
    print("=" * 50)


if __name__ == '__main__':

    start = time.time()
    train_loss = train_one_epoch()
    print("Epoch time:", time.time() - start)
   # run_training()
   # run_test()