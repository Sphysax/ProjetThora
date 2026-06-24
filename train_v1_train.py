from train_v1_common import (
    BEST_MODEL_PATH,
    EPOCHS,
    print_dataset_summary,
    print_environment_info,
    train_one_epoch,
    evaluate,
    val_loader,
    model,
)
import torch


def run_training():
    print_environment_info()
    print_dataset_summary()

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
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(">>> Best model saved!")


if __name__ == '__main__':
    run_training()
