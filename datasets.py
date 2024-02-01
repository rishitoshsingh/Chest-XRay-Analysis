import os

import torch
from pydicom import dcmread
from torch.utils.data import Dataset
from torchvision.io import read_image


class PracticePNGJPG(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        files_list = os.listdir(self.img_dir)
        return len(files_list)

    def __getitem__(self, idx):
        files_list = os.listdir(self.img_dir)
        img_path = os.path.join(self.img_dir, files_list[idx])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image


class PracticeDICOM(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        files_list = os.listdir(self.img_dir)
        return len(files_list)

    def __getitem__(self, idx):
        files_list = os.listdir(self.img_dir)
        img_path = os.path.join(self.img_dir, files_list[idx])
        image = dcmread(img_path).pixel_array
        if self.transform:
            image = self.transform(image)
        return image


class DirectionRGB(Dataset):
    label_mapping = {"up": 0, "right": 1, "down": 2, "left": 3}
    inverse_label_mapping = {0: "up", 1: "right", 2: "down", 3: "left"}

    def __init__(self, image_list_path, transform=None, target_transform=None):
        self.image_list_path = image_list_path
        self.img_path, self.label = [], []
        with open(image_list_path, "r") as file:
            for line in file:
                line = line.strip()
                if line:
                    file_path, label = line.split(",")
                    self.img_path.append(file_path)
                    self.label.append(label)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_path = self.img_path[idx].replace("\\", "/")
        label = self.label[idx]

        img = read_image(
            os.path.join(os.path.dirname(self.image_list_path), img_path)
        )

        if self.transform:
            img = self.transform(img)

        numeric_label = self.label_mapping[label]

        numeric_label = torch.tensor(numeric_label, dtype=torch.long)
        if self.target_transform:
            numeric_label = self.target_transform(numeric_label)

        return img, numeric_label


class GenderRGB(Dataset):
    label_mapping = {"male": 0, "female": 1}
    inverse_label_mapping = {0: "male", 1: "female"}

    def __init__(
        self,
        directory: str,
        task: str,
        transform=None,
        target_transform=None,
    ):
        self.image_list_path = os.path.join(directory, f"list_{task}.txt")
        self.img_path, self.label = [], []
        with open(self.image_list_path, "r") as file:
            for line in file:
                line = line.strip()
                if line:
                    file_path, label = line.split(",")
                    self.img_path.append(file_path)
                    self.label.append(label)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_path = self.img_path[idx].replace("\\", "/")
        label = self.label[idx]

        img = read_image(
            os.path.join(os.path.dirname(self.image_list_path), img_path)
        )

        if self.transform:
            img = self.transform(img)

        numeric_label = self.label_mapping[label]

        numeric_label = torch.tensor(numeric_label, dtype=torch.long)
        if self.target_transform:
            numeric_label = self.target_transform(numeric_label)

        return img, numeric_label


class AgeRGB(Dataset):
    def __init__(
        self,
        directory: str,
        task: str,
        transform=None,
        target_transform=None,
    ):
        self.directory = directory
        self.image_list_path = os.path.join(directory, f"{task}data.csv")
        self.img_name, self.label = [], []
        with open(self.image_list_path, "r") as file:
            for line in file:
                line = line.strip()
                if line:
                    file_path, label = line.split(",")
                    self.img_name.append(file_path)
                    self.label.append(label)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.directory, "XP", "JPGs", self.img_path[idx]
        )
        label = self.label[idx]

        img = read_image(
            os.path.join(os.path.dirname(self.image_list_path), img_path)
        )

        if self.transform:
            img = self.transform(img)

        numeric_label = label

        numeric_label = torch.tensor(numeric_label, dtype=torch.long)
        if self.target_transform:
            numeric_label = self.target_transform(numeric_label)

        return img, numeric_label
