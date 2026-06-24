from train_v1_common import (
    BEST_MODEL_PATH,
    print_dataset_summary,
    print_environment_info,
    evaluate,
    val_loader,
    model,
    DEVICE,
)
import torch


def run_validation():
    print_environment_info()
    print_dataset_summary()

    print("\nLoading best model for validation...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

    val_loss, val_accuracy = evaluate(val_loader)

    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    print(f"Val Loss : {val_loss:.4f}")
    print(f"Val Acc  : {val_accuracy:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    run_validation()
