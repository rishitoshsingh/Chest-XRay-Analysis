import argparse
import os
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import utils
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)

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


class WarmUpResNet(nn.Module):
    def __init__(
        self, n_classes: int, backbone="resnet18", mode="classification"
    ):
        super(WarmUpResNet, self).__init__()
        self.n_classes = n_classes
        if backbone == "resnet18":
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == "resset50":
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.fc = nn.Linear(self.resnet.fc.out_features, self.n_classes)
        if self.mode == "classification":
            self.logsftmx = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        if self.mode == "classification":
            x = self.logsftmx(x)
        return x


def train_model(
    model, loss_fn, optimizer, dataloaders, dataset_sizes, mode, num_epochs
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
            if mode == "classification":
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
                if mode == "classification":
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            if mode == "classification":
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            else:
                print(f"{phase} Loss: {epoch_loss:.4f}")

            if phase == "train":
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

    return model, train_losses, val_losses


def test_model(model, loss_fn, dataloader, dataset_size, mode):
    model.eval()
    test_loss = 0.0
    test_corrects = 0

    last_batch_images, last_batch_prediciton, last_batch_target = (
        None,
        None,
        None,
    )

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader["val"]):
            print()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            if mode == "classification":
                test_corrects += torch.sum(preds == labels.data)

            last_batch_images = inputs.cpu()
            last_batch_prediciton = preds.cpu()
            last_batch_target = labels.cpu()

    test_loss = test_loss / dataset_size
    if mode == "classification":
        test_acc = test_corrects.double() / dataset_size
        print(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
    else:
        print(f"Test Loss: {test_loss:.4f}")

    return last_batch_images, last_batch_prediciton, last_batch_target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-Tuning ResNet (18/50) on warmup datsets"
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
        "--epochs",
        type=int,
        default=100,
        help="Epochs for training model",
    )
    parser.add_argument(
        "--data",
        choices=["direction", "gender", "age"],
        help="Choose data that you want to use",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help='Path to the dataset directory (required for "train" mode)',
    )
    parser.add_argument(
        "--op_dir",
        type=str,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Experiment name (for creating model and result directory)",
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

    if args.mode == "train":
        if args.exp_name and args.op_dir and args.data:
            exp_directory = os.path.join(args.op_dir, args.data, args.exp_name)
        else:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            exp_directory = os.path.join(
                args.op_dir, args.data, current_datetime
            )

        os.makedirs(exp_directory, exist_ok=True)

        utils.split_train_to_val(args.data, args.data_dir, val_ratio=0.2)

        dataloaders, dataset_sizes = utils.get_dataloaders(
            args.data, args.data_dir, ["train", "val"]
        )

        if args.data == "direction":
            model = WarmUpResNet(4)
            loss_fn = nn.NLLLoss()
        elif args.data == "gender":
            model = WarmUpResNet(2)
            loss_fn = nn.NLLLoss()
        elif args.data == "age":
            model = WarmUpResNet(1, mode="regression")
            loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        model = model.to(device)
        trained_model, train_losses, val_losses = train_model(
            model,
            loss_fn,
            optimizer,
            dataloaders,
            dataset_sizes,
            args.epochs,
            args.mode,
        )

        utils.plot_loss(
            train_losses,
            val_losses,
            args.lr,
            os.path.join(exp_directory, "loss-curve.pth"),
        )

        # Save trained weights
        torch.save(
            trained_model.state_dict(),
            os.path.join(exp_directory, "weights.pth"),
        )
        print("Trained weights saved")

    elif args.mode == "test":
        dataloaders, dataset_sizes = utils.get_dataloaders(
            args.data, args.data_dir, ["test"]
        )

        if args.data == "direction":
            model = WarmUpResNet(4)
            loss_fn = nn.NLLLoss()
        elif args.data == "gender":
            model = WarmUpResNet(2)
            loss_fn = nn.NLLLoss()
        elif args.data == "age":
            model = WarmUpResNet(1, mode="regression")
            loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        model.to(device)

        if args.trained_weights_path:
            model.load_state_dict(torch.load(args.trained_weights_path))
        else:
            print("No trained weights path passed")

        (
            last_batch_images,
            last_batch_prediciton,
            last_batch_target,
        ) = test_model(
            model, loss_fn, dataloaders["test"], dataset_sizes["test"]
        )

        exp_directory, weight_filename = os.path.split(
            args.trained_weights_path
        )
        utils.plot_images(
            last_batch_images[:8],
            last_batch_prediciton[:8],
            last_batch_target[:8],
            os.path.join(exp_directory, "test-results.png"),
        )
