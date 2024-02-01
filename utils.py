import os
import random
import shutil

import datasets
import matplotlib.pyplot as plt
import transforms as custom_transforms
from torch.utils.data import DataLoader
from torchvision import transforms


def split_train_to_val(data: str, data_dir: str, val_ratio=0.2):
    # Rename the original file
    if data == "direction" or data == "gender":
        train_list_file_path = os.path.join(data_dir, "list_train.txt")
        val_list_file_path = os.path.join(data_dir, "list_val.txt")
        train_list_org_file_path = os.path.join(data_dir, "list_train_org.txt")
    elif data == "age":
        train_list_file_path = os.path.join(data_dir, "XP", "traindata.csv")
        val_list_file_path = os.path.join(data_dir, "XP", "valdata.csv")
        train_list_org_file_path = os.path.join(
            data_dir, "XP", "trainingdata_org.csv"
        )

    if not os.path.exists(train_list_org_file_path):
        shutil.move(
            train_list_org_file_path.replace("_org", ""),
            train_list_org_file_path,
        )

    # Read the original content
    with open(train_list_org_file_path, "r") as file:
        lines = file.readlines()
        if data == "age":
            lines = lines[1:]

    if data == "gender":
        lines = [line.replace(",", ".", 1) for line in lines]

    # Calculate the number of lines for the test set
    val_size = int(len(lines) * val_ratio)

    # Randomly select lines for the test set
    val_lines = random.sample(lines, val_size)

    # Write the remaining lines back to the original file
    with open(train_list_file_path, "w") as file:
        file.writelines([line for line in lines if line not in val_lines])

    # Write the selected lines to the test file
    with open(val_list_file_path, "w") as file:
        file.writelines(val_lines)


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


def get_dataloaders(
    data: str, data_dir: str, req_dataloaders=["train", "val"]
):
    if data == "direction":
        xray_datasets = {
            x: datasets.DirectionRGB(
                os.path.join(data_dir),
                task=x,
                transform=data_transforms[x]
                if x != "test"
                else data_transforms["val"],
            )
            for x in req_dataloaders
        }
    elif data == "gender":
        xray_datasets = {
            x: datasets.GenderRGB(
                os.path.join(data_dir),
                task=x,
                transform=data_transforms[x]
                if x != "test"
                else data_transforms["val"],
            )
            for x in req_dataloaders
        }
    elif data == "age":
        xray_datasets = {
            x: datasets.AgeRGB(
                os.path.join(data_dir),
                task=x,
                transform=data_transforms[x]
                if x != "test"
                else data_transforms["val"],
            )
            for x in req_dataloaders
        }
    dataloaders = {
        x: DataLoader(
            xray_datasets[x], batch_size=64, shuffle=True, num_workers=4
        )
        for x in req_dataloaders
    }
    dataset_sizes = {x: len(xray_datasets[x]) for x in req_dataloaders}
    return dataloaders, dataset_sizes


def plot_loss(train_losses, val_losses, learning_rate, plot_file_path):
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Learning rate: {learning_rate}")
    plt.legend()

    plt.savefig(os.path.join("plots", plot_file_path))
    # plt.show()


def image_to_arrag(image):
    return image.permute(1, 2, 0).cpu().detach().numpy().astype("int")


def untransform_tensor(image):
    tensor_ = image.squeeze()

    unnormalize_transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        ]
    )
    untransformed_tensor = unnormalize_transform(tensor_)
    return untransformed_tensor


def plot_images(images, predictions, targets, plot_file_path):
    fig, axs = plt.subplots(4, 2, figsize=(8, 16))
    inverse_label_mapping = {0: "up", 1: "right", 2: "down", 3: "left"}
    for i in range(4):
        for j in range(2):
            index = i * 2 + j
            image = image_to_arrag(untransform_tensor(images[index]))
            axs[i, j].imshow(image)
            axs[i, j].axis("off")
            axs[i, j].set_title(
                f"Target: {inverse_label_mapping[targets[index].item()]}"
                "\nPrediction:"
                f" {inverse_label_mapping[predictions[index].item()]}"
            )

    os.makedirs("plots", exist_ok=True)
    plt.savefig(os.path.join("plots", plot_file_path))
    # plt.show()
