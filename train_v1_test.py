from train_v1_common import (
    BEST_MODEL_PATH,
    print_dataset_summary,
    print_environment_info,
    evaluate,
    test_loader,
    model,
    DEVICE,
)
import torch


def run_test():
    print_environment_info()
    print_dataset_summary()

    print("\nLoading best model for test...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

    test_loss, test_accuracy = evaluate(test_loader)

    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)
    print(f"Test Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_accuracy:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    run_test()
