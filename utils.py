import glob
import os
import random
import shutil
from functools import reduce
from typing import List, Union

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import transforms as custom_transforms

# import cv2
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageOps

# from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader
from torchvision import transforms


def calculate_iou(outputs, labels):
    intersection = torch.logical_and(outputs, labels).sum()
    union = torch.logical_or(outputs, labels).sum()
    iou = intersection.float() / union.float()
    return iou.item()


def calculate_dice(outputs, labels):
    intersection = torch.logical_and(outputs, labels).sum()
    dice = (2. * intersection) / (outputs.sum() + labels.sum())
    return dice.item()


def split_train_to_val(data: str, data_dir: str, val_ratio=0.2):
    # Rename the original file
    if data == "direction" or data == "gender" or data == "segmentation01" or data == "segmentation02":
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


class ToMultiChannelMasks(object):
    MASKS = {"background": 0, "heart": 85, "body": 170, "lung": 255}
    MASKS_REVERSE = {0: 'background', 85: 'heart', 170: 'body',
                     255: 'lung'}

    def __call__(self, image):
        masks = []
        image_arr = np.array(image)
        for _, value in self.MASKS.items():
            mask = (image_arr == value).astype(np.float32)
            masks.append(mask)
        return np.stack(masks).transpose((1, 2, 0))


class ToMaskClass(object):
    MASKS = {"background": 0, "heart": 85, "body": 170, "lung": 255}
    MASKS_REVERSE = {0: 'background', 85: 'heart', 170: 'body',
                     255: 'lung'}

    def __call__(self, image_tensor):
        image_tensor = image_tensor * 3
        print(image_tensor.unique())
        return image_tensor.int()


class BoundingBoxTransform(object):

    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, bboxes):
        width, height = self.image_size
        normalized_boxes = bboxes.copy()
        normalized_boxes[:, 0::2] /= width
        normalized_boxes[:, 1::2] /= height
        return torch.tensor(normalized_boxes, dtype=torch.float32)


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
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
}
age_data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    ),
}

segmentation_data_transform = {
    "train": transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
}
segmentation_data_y_transform = {
    "train": transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
}
seg_masks_transform = {
    "train": transforms.Compose([
        ToMultiChannelMasks(),
        transforms.ToTensor(),
    ]),
    "val": transforms.Compose([
        ToMultiChannelMasks(),
        transforms.ToTensor(),
    ]),
}
localization_transform = {
    "train": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
    "val": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
}
localization_y_transform = {
    "train": transforms.Compose([
        BoundingBoxTransform((256, 256)),
    ]),
    "val": transforms.Compose([
        BoundingBoxTransform((256, 256)),
    ]),
}
anamoly_data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])


def get_dataloaders(
    data: str, data_dir: Union[str, List[str]], req_dataloaders=["train", "val"], batch_size=4,
    num_workers=4
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
                transform=age_data_transforms[x]
                if x != "test"
                else age_data_transforms["val"],
            )
            for x in req_dataloaders
        }
    elif data == "segmentation01":
        xray_datasets = {
            x: datasets.Segmentation01(
                os.path.join(data_dir),
                task=x,
                transform=segmentation_data_transform[x]
                if x != "test"
                else segmentation_data_transform["val"],
                target_transform=segmentation_data_y_transform[x]
                if x != "test"
                else segmentation_data_y_transform["val"],
            )
            for x in req_dataloaders
        }
    elif data == "segmentation02":
        xray_datasets = {
            x: datasets.Segmentation02(
                os.path.join(data_dir),
                task=x,
                transform=segmentation_data_transform[x]
                if x != "test"
                else segmentation_data_transform["val"],
                target_transform=seg_masks_transform[x]
                if x != "test"
                else seg_masks_transform["val"],
            )
            for x in req_dataloaders
        }
    elif data == "segmentation02loc":
        xray_datasets = {
            x: datasets.Segmentation02Loc(
                os.path.join(data_dir),
                task=x,
                transform=localization_transform[x]
                if x != "test"
                else localization_transform["val"],
                target_transform=localization_y_transform[x]
                if x != "test"
                else localization_y_transform["val"],
            )
            for x in req_dataloaders
        }
    elif data == "anamoly":
        xray_datasets = {
            x: datasets.AutoEncoderIMG(
                os.path.join(data_dir),
                task=x,
                transform=segmentation_data_transform[x]
                if x != "test"
                else segmentation_data_transform["val"],
                target_transform=segmentation_data_transform[x]
                if x != "test"
                else segmentation_data_transform["val"],
            )
            for x in req_dataloaders
        }
    elif data == "anamoly_direction":
        xray_datasets = {
            x: datasets.AnamolyDirectionDataset(
                os.path.join(data_dir[0]),
                os.path.join(data_dir[1]) if data_dir[1] else None,
                task=x,
                transform=anamoly_data_transform,
                target_transform=anamoly_data_transform,
            )
            for x in req_dataloaders
        }
    dataloaders = {
        x: DataLoader(
            xray_datasets[x], batch_size=batch_size, shuffle=True,
            num_workers=num_workers
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

    plt.savefig(plot_file_path)
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


def plot_images(type, images, predictions, targets, inv_label_map,
                plot_file_path):
    
    if type == "segmentation":

        print(images.shape)
        image = images[0]
        mask = predictions[0]
        mask_org = targets[0]
        fig, axes = plt.subplots(1, 3)
        # Plot the first image
        axes[0].imshow(image[0], cmap="gray")
        axes[0].set_title('Image 1')

        # Plot the second image
        print(mask.max(), mask.min())
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title('Image 2')
        print(mask_org[0].max(), mask_org[0].min())
        axes[2].imshow(mask_org[0], cmap="gray")
        axes[2].set_title('Image 3')
        
        # fig = plt.figure(figsize=(16, 4))
        # gs = GridSpec(2, 4, figure=fig)

        # # Add images and masks to the grid
        # for i in range(4):
        #     # Add image to the grid
        #     # image = image_to_arrag(untransform_tensor(images[i]))
        #     image = image_to_arrag(images[i])
        #     print(images[i].max(), images[i].min())
        #     print(predictions[i].max(), predictions[i].min())
        #     mask = predictions[i]
        #     # image = images[i]
        #     # mask = predictions[i]
        #     ax_img = fig.add_subplot(gs[0, i])
        #     # print(image)
        #     # print(mask)
        #     ax_img.imshow(image, cmap="gray")
        #     ax_img.set_title(f'Image {i+1}')
        #     ax_img.axis('off')

        #     # Add mask to the grid
        #     ax_mask = fig.add_subplot(gs[1, i])
        #     ax_mask.imshow(mask, cmap="gray")
        #     ax_mask.set_title(f'Mask {i+1}')
        #     ax_mask.axis('off')

        # fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        # for row_i in range(2):
        #     for j in range(2):
        #         index = row_i * 2 + j
        #         image = image_to_arrag(untransform_tensor(images[index]))
        #         mask = image_to_arrag(predictions[index].unsqueeze(0))
        #         print(row_i, index, row_i, (row_i+index)%2)
        #         print(row_i, index, row_i, (row_i+index)%2+1)
        #         axs[row_i, (row_i+index)%2].imshow(image)
        #         axs[row_i, (row_i+index)%2].axis("off")
        #         axs[row_i, (row_i+index)%2].set_title("Input")
        #         axs[row_i, (row_i+index)%2 +1].imshow(mask)
        #         axs[row_i, (row_i+index)%2 +1].axis("off")
        #         axs[row_i, (row_i+index)%2 +1].set_title("Predicted Mask")

    else:
        fig, axs = plt.subplots(4, 2, figsize=(8, 16))

        for i in range(4):
            for j in range(2):
                index = i * 2 + j
                image = image_to_arrag(untransform_tensor(images[index]))
                axs[i, j].imshow(image)
                axs[i, j].axis("off")
                axs[i, j].set_title(
                    "Target:"
                    f" {inv_label_map[targets[index].item()] if inv_label_map else targets[index].item()}\nPrediction:"
                    f" {inv_label_map[predictions[index].item()] if inv_label_map else targets[index].item()}"
                )

    os.makedirs("plots", exist_ok=True)
    plt.savefig(plot_file_path)
    # plt.show()


# def plot_images(
#     images, predictions, targets, grayscale_cams, inv_label_map, plot_file_path
# ):
#     fig, axs = plt.subplots(8, 2, figsize=(8, 16))

#     for i in range(8):
#         # for j in range(2):
#         # index = i * 2 + j
#         image = image_to_arrag(untransform_tensor(images[i]))
#         axs[i, 0].imshow(image)
#         axs[i, 0].axis("off")
#         axs[i, 0].set_title(
#             "Target:"
#             f" {inv_label_map[targets[i].item()] if inv_label_map else targets[i].item()}\nPrediction:"
#             f" {inv_label_map[predictions[i].item()] if inv_label_map else targets[i].item()}"
#         )
#         axs[i, 1].imshow(
#             show_cam_on_image(
#                 image.astype(float) / 255, grayscale_cams[i], use_rgb=True
#             )
#         )
#         axs[i, 1].axis("off")
#         # axs[i, 0].set_title(
#         #     "Target:"
#         #     f" {inv_label_map[targets[i].item()] if inv_label_map else targets[i].item()}\nPrediction:"
#         #     f" {inv_label_map[predictions[i].item()] if inv_label_map else targets[i].item()}"
#         # )

#     os.makedirs("plots", exist_ok=True)
#     plt.savefig(plot_file_path)
#     # plt.show()

def create_data_file_lists(data_dir):
    # Create lists of files for org_train and org_test
    org_train_files = glob.glob(os.path.join(data_dir, 'org_train', '*'))
    org_test_files = glob.glob(os.path.join(data_dir, 'org_test', '*'))

    # Write lists to files
    org_label_replacements = {".bmp": "_label.png", "org": "label"}
    with open(os.path.join(data_dir, 'list_train.txt'), 'w') as f_train:
        for file_path in org_train_files:
            rel_path = os.path.relpath(file_path, start=data_dir)
            label_rel_path = reduce(lambda s, r: s.replace(*r),
                                    org_label_replacements.items(), rel_path)
            f_train.write(rel_path + "," + label_rel_path + '\n')

    with open(os.path.join(data_dir, 'list_test.txt'), 'w') as f_test:
        for file_path in org_test_files:
            rel_path = os.path.relpath(file_path, start=data_dir)
            label_rel_path = reduce(lambda s, r: s.replace(*r),
                                    org_label_replacements.items(), rel_path)
            f_test.write(rel_path + "," + label_rel_path + '\n')


def to_singlechannel_masks(masks):
    _, max_indices = torch.max(masks, dim=0)
    return max_indices.float()


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