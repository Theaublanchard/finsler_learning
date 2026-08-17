import torch
from pathlib import Path
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Load a dataset and add noise to it. Beware, modify that inplace.")
    parser.add_argument("--dataset_path", 
                        type=str, 
                        required=True, 
                        help="Path to the folder containing the files `train_samples.pth` and `test_samples.pth`.")
    parser.add_argument("--noise_std", 
                        type=float, 
                        required=True,
                        help="Percentage of std of the dataset to use as noise. For example, if the dataset has std=1.0 and you set noise_std=0.1, then the noise will have std=0.1.")
    return parser.parse_args()

def noise_dataset(dataset_path: Path, noise_std: float):
    # Load the dataset
    imgs_train, labels_train = torch.load(dataset_path / "train_samples.pth")
    imgs_test, labels_test = torch.load(dataset_path / "test_samples.pth")

    # Add noise to the training images
    std_data = labels_train.std().item()
    labels_train_noisy = labels_train + noise_std * std_data * torch.randn_like(labels_train)
    labels_test_noisy = labels_test + noise_std * std_data * torch.randn_like(labels_test)

    # Save the noisy dataset
    torch.save((imgs_train, labels_train_noisy), dataset_path / "train_samples.pth")
    torch.save((imgs_test, labels_test_noisy), dataset_path / "test_samples.pth")


if __name__ == "__main__":
    args = parse_args()
    noise_dataset(Path(args.dataset_path), args.noise_std)
