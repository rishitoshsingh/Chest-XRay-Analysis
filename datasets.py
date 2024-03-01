import os
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from pydicom import dcmread
from torch.utils.data import Dataset
from torchvision.io import read_image


class InvertGrayscale(object):
    def __call__(self, img):
        img_inverted_pil = ImageOps.invert(img)
        return img_inverted_pil
    

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

    def __init__(
            self,
            directory,
            task,
            transform=None,
            target_transform=None
    ):
        self.directory = directory
        self.image_list_path = os.path.join(directory, f"list_{task}.txt")
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
                    self.directory, self.img_name[idx]
                ).replace("\\", "/")
        img = Image.open(img_path).convert("L")

        label = self.label[idx]

        if self.transform:
            img = self.transform(img)

        numeric_label = self.label_mapping[label]

        label = torch.zeros(4)
        label[numeric_label] = 1

        if self.target_transform:
            label = self.target_transform(label)

        return img, label


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
        self.directory = directory
        self.image_list_path = os.path.join(directory, f"list_{task}.txt")
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
                    self.directory, self.img_name[idx]
                ).replace("\\", "/")
        img = Image.open(img_path).convert("L")

        label = self.label[idx]

        if self.transform:
            img = self.transform(img)

        numeric_label = self.label_mapping[label]

        numeric_label = torch.tensor(numeric_label, dtype=torch.float)
        if self.target_transform:
            numeric_label = self.target_transform(numeric_label)

        return img, np.array([numeric_label])


class AgeRGB(Dataset):
    def __init__(
        self,
        directory: str,
        task: str,
        transform=None,
        target_transform=None,
    ):
        self.directory = directory
        self.image_list_path = os.path.join(directory, "XP", f"{task}data.csv")
        self.img_name, self.label = [], []
        with open(self.image_list_path, "r") as file:
            for line in file:
                line = line.strip()
                if line and line.split(",")[0] != "filenames":
                    file_path, label = line.split(",")
                    self.img_name.append(file_path)
                    self.label.append(label)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.directory, "XP", "JPGs", self.img_name[idx]
        ).replace("\\", "/")
        label = self.label[idx]

        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)

        numeric_label = int(label)

        numeric_label = torch.tensor(numeric_label, dtype=torch.float)
        if self.target_transform:
            numeric_label = self.target_transform(numeric_label)

        return img, numeric_label


class Segmentation01(Dataset):
    def __init__(
        self,
        directory: str,
        task: str,
        transform=None,
        target_transform=None,
    ):
        self.directory = directory
        self.image_list_path = os.path.join(directory, f"list_{task}.txt")
        self.img_name, self.label_name = [], []
        with open(self.image_list_path, "r") as file:
            for line in file:
                line = line.strip()
                if line and line.split(",")[0] != "filenames":
                    file_path, label_path = line.split(",")
                    self.img_name.append(file_path)
                    self.label_name.append(label_path)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_name)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.directory, self.img_name[idx]
        ).replace("\\", "/")
        label_path = os.path.join(
            self.directory, self.label_name[idx]
        ).replace("\\", "/")

        img = Image.open(img_path).convert("L")
        label = Image.open(label_path).convert("L")
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, label


class Segmentation02(Dataset):
    def __init__(
        self,
        directory: str,
        task: str,
        transform=None,
        target_transform=None,
    ):
        self.directory = directory
        self.image_list_path = os.path.join(directory, f"list_{task}.txt")
        self.img_name, self.label_name = [], []
        with open(self.image_list_path, "r") as file:
            for line in file:
                line = line.strip()
                if line and line.split(",")[0] != "filenames":
                    file_path, label_path = line.split(",")
                    self.img_name.append(file_path.strip())
                    self.label_name.append(label_path.strip())

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_name)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.directory, self.img_name[idx]
        ).replace("\\", "/")
        label_path = os.path.join(
            self.directory, self.label_name[idx]
        ).replace("\\", "/")

        img = Image.open(img_path).convert("L")
        label = Image.open(label_path)
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, label


class Segmentation02Loc(Dataset):
    def __init__(
        self,
        directory: str,
        task: str,
        transform=None,
        target_transform=None,
    ):
        self.directory = directory
        self.image_list_path = os.path.join(directory, f"list_{task}_loc.txt")
        self.img_name, self.y = [], []
        with open(self.image_list_path, "r") as file:
            for line in file:
                row = line.strip().split(",")
                file_path, heart_bb, lungs_bb = row[0], row[2:6], row[7:]
                self.img_name.append(file_path.strip())
                self.y.append([heart_bb, lungs_bb])
        self.y = np.array(self.y, dtype=np.float32)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.directory, self.img_name[idx]
        ).replace("\\", "/")

        img = Image.open(img_path).convert("L")
        label = self.y[idx]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label


class AutoEncoderIMG(Dataset):
    def __init__(
        self,
        directory: str,
        task: str,
        transform=None,
        target_transform=None,
    ):
        self.directory = directory
        self.task = task
        if self.task == "train":
            self.image_list_path = os.path.join(directory, "normal200.txt")
        else:  # val and test
            self.image_list_path = os.path.join(directory, "normal.txt")
        self.img_name = []
        self.test_class = []
        with open(self.image_list_path, "r") as file:
            for line in file:
                line = line.strip()
                if line and line != "filenames":
                    self.img_name.append(line)
        if self.task == "test":
            self.test_class.extend(["normal"]*len(self.img_name))
            with open(os.path.join(directory, "abnormal.txt"), "r") as file:
                for line in file:
                    line = line.strip()
                    if line and line != "filenames":
                        self.img_name.append(line)
            self.test_class.extend(["abnormal"]*(len(self.img_name)-len(self.test_class)))
            # random.shuffle(self.img_name)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.directory, self.img_name[idx]
        ).replace("\\", "/")

        img = Image.open(img_path).convert("L")
        img_reconstructed = img.copy()
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            img_reconstructed = self.target_transform(img_reconstructed)
        if self.task == "test":
            return img, img_reconstructed, self.test_class[idx]
        return img, img_reconstructed


class AnamolyDirectionDataset(Dataset):
    def __init__(
        self,
        autoencoder_img_directory: str,
        direction_img_directory: str,
        task: str,
        transform=None,
        target_transform=None,
    ):
        self.autoencoder_img_directory = autoencoder_img_directory
        self.direction_img_directory = direction_img_directory
        if direction_img_directory != None:
            self.image_file_paths = [
                os.path.join(autoencoder_img_directory, "normal.txt"),
                os.path.join(direction_img_directory, "list_train.txt"),
                os.path.join(direction_img_directory, "list_test.txt"),
                os.path.join(direction_img_directory, "list_val.txt"),
            ]
        else:
            self.image_file_paths = [
                os.path.join(autoencoder_img_directory, "normal.txt"),
            ]
        self.img_paths = []
        self.label = []
        with open(self.image_file_paths[0], "r") as file:
            for line in file:
                line = line.strip()
                if line and line != "filenames":
                    img_path = os.path.join(
                        autoencoder_img_directory, line
                    ).replace("\\", "/")
                    self.img_paths.append(img_path)
        
        self.label.extend(["normal"]*len(self.img_paths))

        if direction_img_directory != None:
            for file_path in self.image_file_paths[1:]:
                with open(file_path, "r") as file:
                    for line in file:
                        line = line.strip()
                        if line:
                            img_name, label = line.split(",")
                            img_name = img_name[1:]
                            img_path = os.path.join(
                                direction_img_directory, img_name
                            ).replace("\\", "/")
                            self.img_paths.append(img_path)
                            # if label == "up":
                            #     label == "normal"
                            self.label.append(label)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        img = Image.open(self.img_paths[idx]).convert("L")
        label = self.label[idx]
        # img_reconstructed = img.copy()
        if label != "normal":
            img = InvertGrayscale()(img)
        if self.transform:
            img = self.transform(img)
        # if self.target_transform:
        #     img_reconstructed = self.target_transform(img_reconstructed)
        return img, label
        # return img, img_reconstructed
