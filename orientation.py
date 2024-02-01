import argparse
import os
import warnings
from datetime import datetime

import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import transforms as custom_transforms
import utils
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

warnings.filterwarnings("ignore")


def get_device(device_str):
    if device_str == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "cpu":
        return torch.device("cpu")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")  # You can modify the GPU index if needed
    else:
        print("Warning: Invalid device option. Using CPU.")
        return torch.device("cpu")


# Define data transforms
data_transforms = {
    "train": transforms.Compose(
        [
            custom_transforms.ToFloat(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            custom_transforms.ToFloat(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


class DirectionResNet18(nn.Module):
    def __init__(self):
        super(DirectionResNet18, self).__init__()
        self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.fc = nn.Linear(
            self.resnet18.fc.out_features, 4
        )  # 4 classes (up, right, down, left)
        self.logsftmx = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc(x)
        x = self.logsftmx(x)
        return x


def train_model(
    model, loss_fn, optimizer, dataloaders, dataset_sizes, num_epochs=10
):
    model.train()
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "train":
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

    return model, train_losses, val_losses


def test_model(model, loss_fn, dataloader, dataset_size):
    model.eval()
    test_loss = 0.0
    test_corrects = 0

    last_batch_images, last_batch_prediciton, last_batch_target = (
        None,
        None,
        None,
    )

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            print()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)

            last_batch_images = inputs.cpu()
            last_batch_prediciton = preds.cpu()
            last_batch_target = labels.cpu()

    test_loss = test_loss / dataset_size
    test_acc = test_corrects.double() / dataset_size

    print(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    return last_batch_images, last_batch_prediciton, last_batch_target


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Pre-Training ResNet on Chest Xrays to find direction of the xray."
        )
    )
    parser.add_argument(
        "mode",
        choices=["train", "test"],
        help='Choose "train" or "test" mode',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for training model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help='Path to the dataset directory (required for "train" mode)',
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu", "mps"],
        default="gpu",
        help="Device to use (cpu, gpu, mps)",
    )
    parser.add_argument(
        "--trained_weights_path",
        type=str,
        help="Path to trained weights for testing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    global device
    device = get_device(args.device)
    print(f"Using: {device}")

    if not args.data_dir:
        print("Data directory is required")

    data = "direction"
    if args.mode == "train":
        utils.split_train_to_val(data, args.data_dir, val_ratio=0.2)

        xray_datasets = {
            x: datasets.DirectionRGB(
                os.path.join(args.data_dir, f"list_{x}.txt"),
                transform=data_transforms[x],
            )
            for x in ["train", "val"]
        }
        dataloaders = {
            x: DataLoader(
                xray_datasets[x], batch_size=64, shuffle=True, num_workers=4
            )
            for x in ["train", "val"]
        }
        dataset_sizes = {x: len(xray_datasets[x]) for x in ["train", "val"]}

        model = DirectionResNet18()
        loss_fn = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        model = model.to(device)
        trained_model, train_losses, val_losses = train_model(
            model,
            loss_fn,
            optimizer,
            dataloaders,
            dataset_sizes,
            num_epochs=10,
        )

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        utils.plot_loss(
            train_losses,
            val_losses,
            args.lr,
            f"{current_datetime}-train-val-plot.png",
        )

        # Save trained weights
        os.makedirs(os.path.dirname("weights/"), exist_ok=True)
        weights_file_name = os.path.join("weights", f"{current_datetime}.pth")
        torch.save(
            trained_model.state_dict(),
            weights_file_name,
        )
        print(f"Trained weights saved to {weights_file_name}")

    elif args.mode == "test":
        xray_test_dataset = datasets.DirectionRGB(
            os.path.join(args.data_dir, "list_test.txt"),
            transform=data_transforms["val"],
        )
        dataloader = DataLoader(
            xray_test_dataset, batch_size=64, shuffle=True, num_workers=4
        )
        test_dataset_size = len(xray_test_dataset)

        model = DirectionResNet18()
        model.to(device)
        loss_fn = nn.NLLLoss()
        if args.trained_weights_path:
            model.load_state_dict(torch.load(args.trained_weights_path))
        else:
            print("No trained weights path passed")

        (
            last_batch_images,
            last_batch_prediciton,
            last_batch_target,
        ) = test_model(model, loss_fn, dataloader, test_dataset_size)

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _, weight_filename = os.path.split(args.trained_weights_path)
        plot_filename = weight_filename.split(".")[0] + ".png"
        utils.plot_images(
            last_batch_images[:8],
            last_batch_prediciton[:8],
            last_batch_target[:8],
            plot_filename,
        )
