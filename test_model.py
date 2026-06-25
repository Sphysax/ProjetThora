import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_v1_common import (
    BEST_MODEL_PATH,
    DEVICE,
    model,
    test_dataset,
    test_loader,
    print_dataset_summary,
    print_environment_info,
)


def run_test():
    print_environment_info()
    print_dataset_summary()

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model checkpoint not found: {BEST_MODEL_PATH}\n"
            "Run training first to create best_pneumonia_model.pth."
        )

    print("\nLoading best pretrained model for test...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    print("\n" + "=" * 60)
    print("PNEUMONIA MODEL TEST RESULTS")
    print("=" * 60)
    print(f"Device        : {DEVICE}")
    print(f"Test Images   : {len(test_dataset)}")
    print(f"Accuracy      : {accuracy * 100:.2f}%")

    print("\nClasses:")
    print(test_dataset.classes)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("=" * 60)


if __name__ == '__main__':
    run_test()
