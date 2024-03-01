import argparse
import os
import warnings
from datetime import datetime

import datasets as warmup_datsets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils

# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

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
        self.mode = mode
        if backbone == "resnet18":
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == "resset50":
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.fc1 = nn.Linear(self.resnet.fc.out_features, 512)
        self.fc2 = nn.Linear(512, self.n_classes)
        self.dropout = nn.Dropout(0.5)
        if self.mode == "classification":
            self.logsftmx = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        if self.mode == "classification":
            x = self.logsftmx(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Changed input channels from 1 to 3
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)  # Changed output channels from 1 to 3
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(torch.cat([x1, x1], dim=1))
        return self.sigmoid(x2) 


def train_model(
    model,
    loss_fn,
    optimizer,
    dataloaders,
    dataset_sizes,
    num_epochs,
    mode,
    early_stopping=False,
    patience=3,
):
    model.train()
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    patience_counter = 0

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
                if early_stopping and epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    patience_counter = 0
                elif early_stopping:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(
                            f"Early stopping triggered after {patience} epochs"
                            " without improvement."
                        )
                        return model, train_losses, val_losses

    return model, train_losses, val_losses


def test_model(model, loss_fn, dataloader, dataset_size, mode):
    model.eval()
    test_loss = 0.0
    test_corrects = 0

    last_batch_images, last_batch_prediction, last_batch_target = (
        None,
        None,
        None,
    )

    all_predictions = []
    all_targets = []
    all_outputs = []
    dice_coefficients = []
    iou_scores = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if mode == "segmentation":
                # prediction masks
                preds = torch.argmax(outputs, dim=1)
                output_binary = outputs > 0.5
                target_binary = outputs > 0.5
                
                intersection = (output_binary & target_binary).sum().item()
                union = (output_binary | target_binary).sum().item()
                # print(output)
                dice_coefficient = (2.0 * intersection) / (output_binary.sum().item() + target_binary.sum().item())
                iou_score = intersection / union
                
                dice_coefficients.append(dice_coefficient)
                iou_scores.append(iou_score)
            else:
                _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            if mode == "classification":
                test_corrects += torch.sum(preds == labels.data)
            # elif mode == "regression":
                # Calculate MAE
                # test_loss += torch.sum(torch.abs(outputs - labels)).item()

            all_outputs.append(outputs.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
            last_batch_images = inputs.cpu()
            last_batch_prediction = preds.cpu()
            last_batch_target = labels.cpu()

    test_loss = test_loss / dataset_size
    if mode == "classification":
        test_acc = test_corrects.double() / dataset_size
        print(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

        # Calculate AUC
        auc = roc_auc_score(all_targets, np.array(all_predictions))
        print(f"AUC: {auc:.4f}")
    elif mode == "regression":
        mae = test_loss / dataset_size
        mae = torch.sum(torch.abs(all_outputs - all_targets)).item()
        print(f"Test Loss: {test_loss:.4f} MAE: {mae:.4f}")
    elif mode == "segmentation":

        mean_dice_coefficient = np.mean(dice_coefficients)
        mean_iou_score = np.mean(iou_scores)
        print('Mean Dice Coefficient: {:.4f}'.format(mean_dice_coefficient))
        print('Mean IoU Score: {:.4f}'.format(mean_iou_score))

        all_predictions = torch.from_numpy(np.array(all_predictions))
        all_targets = torch.from_numpy(np.array(all_targets))
        print("all_outputs type:" , type(all_outputs))
        print("all_targets type:" , type(all_targets))
        print("all_predictions type:" , type(all_predictions))
        # dice_coefficient = utils.calculate_dice(torch.from_numpy(np.array(all_predictions)),
        #                                         torch.from_numpy(np.array(all_targets)))
        # iou = utils.calculate_iou(all_predictions,
        #                           all_targets)
        # print(f"Dice Coefficient: {dice_coefficient:.4f}, IoU: {iou:.4f}")

    else:
        print(f"Test Loss: {test_loss:.4f}")

    # target_layers = [model.resnet.layer4[-1]]
    # cam = GradCAM(model=model, target_layers=target_layers)
    # cam_targets = [ClassifierOutputTarget(x) for x in last_batch_target]
    # grayscale_cam = cam(
    #     input_tensor=last_batch_images.detach().clone(), targets=cam_targets
    # )

    # In this example grayscale_cam has only one image in the batch:
    # grayscale_cams = grayscale_cam[0, :]
    # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return (
        last_batch_images,
        last_batch_prediction,
        last_batch_target,
        # grayscale_cams,
    )


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
        "--type",
        default="classification",
        choices=["classification", "regression", "segmentation"],
        help='Choose "classification", "segmentation" or "regression" mode',
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "sgd", "adagrad"],
        help='Choose "classification" or "regression" mode',
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
        choices=["direction", "gender", "age", "segmentation01"],
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

        utils.split_train_to_val(args.data, args.data_dir, val_ratio=0.1)

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
        elif args.data == "segmentation01":
            # model = UNet(3, 1)
            model = UNet()
            loss_fn = nn.BCELoss()

        if args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        elif args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == "adagrad":
            optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

        model = model.to(device)
        trained_model, train_losses, val_losses = train_model(
            model,
            loss_fn,
            optimizer,
            dataloaders,
            dataset_sizes,
            args.epochs,
            args.type,
        )

        utils.plot_loss(
            train_losses,
            val_losses,
            args.lr,
            os.path.join(exp_directory, "loss-curve.png"),
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
        elif args.data == "segmentation01":
            # model = UNet(3, 1)
            model = UNet()
            loss_fn = nn.BCELoss()
        model.to(device)

        if args.trained_weights_path:
            model.load_state_dict(torch.load(args.trained_weights_path))
        else:
            print("No trained weights path passed")

        (
            last_batch_images,
            last_batch_prediciton,
            last_batch_target,
            # grayscale_cams,
        ) = test_model(
            model,
            loss_fn,
            dataloaders["test"],
            dataset_sizes["test"],
            args.type,
        )

        exp_directory, weight_filename = os.path.split(
            args.trained_weights_path
        )
        if args.data == "direction":
            inv_map = warmup_datsets.DirectionRGB.inverse_label_mapping
        elif args.data == "gender":
            inv_map = warmup_datsets.GenderRGB.inverse_label_mapping
        elif args.data == "age" or args.data == "segmentation01":
            inv_map = None

        
        utils.plot_images(
            args.type,
            last_batch_images[:8],
            last_batch_prediciton[:8],
            last_batch_target[:8],
            # grayscale_cams[:8],
            inv_map,
            os.path.join(exp_directory, "test-results.png"),
        )
