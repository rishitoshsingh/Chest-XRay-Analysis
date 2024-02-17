import argparse

import cv2
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from gender import WarmUpResNet


class GradCAM(WarmUpResNet):
    def __init__(self):
        super(WarmUpResNet, self).__init__()
        self.eval()
        self.feature_maps = None
        self.gradient = None

    def forward(self, x):
        self.feature_maps = None
        self.gradient = None

        def hook_fn(module, input, output):
            self.feature_maps = output.detach()

        def backward_hook_fn(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()

        # Register hook for feature maps
        feature_layer = self.resnet.layer4[
            -1
        ]  # Choose the last layer of the backbone
        hook_handle = feature_layer.register_forward_hook(hook_fn)
        backward_hook_handle = feature_layer.register_backward_hook(
            backward_hook_fn
        )

        # Forward pass
        logits, _ = self.model(x)
        logits[
            :, logits.argmax(dim=1)
        ].backward()  # Backward pass to compute gradients

        # Remove hook
        hook_handle.remove()
        backward_hook_handle.remove()

    def generate_heatmap(self):
        # Global average pooling of gradients
        pooled_gradient = F.adaptive_avg_pool2d(self.gradient, 1)

        # Weight the feature maps by the gradients
        weighted_feature_maps = (self.feature_maps * pooled_gradient).sum(
            dim=1, keepdim=True
        )

        # ReLU operation
        weighted_feature_maps = F.relu(weighted_feature_maps)

        # Upsample the heat map to the input size
        upsampled_heatmap = F.interpolate(
            weighted_feature_maps,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize the heatmap
        heatmap = upsampled_heatmap.squeeze().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap


# Assuming you have an instance of WarmUpResNet called 'model'
# Initialize GradCAM with the model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-Tuning ResNet (18/50) on warmup datsets"
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


if __name__ == "__main__":
    args = parse_args()
    global device
    device = get_device(args.device)
    print(f"Using: {device}")

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
    model.to(device)
    model = GradCAM()
    model.load_state_dict(torch.load(args.trained_weights_path))
    for image in iter(dataloaders["test"]):
        model.forward(image)
        heatmap = model.generate_heatmap()

        # Rescale heatmap to the input image size for visualization
        heatmap_rescaled = cv2.resize(
            heatmap, (image.shape[3], image.shape[2])
        )

        # Apply heatmap on the input image
        heatmap_overlay = cv2.applyColorMap(
            np.uint8(255 * heatmap_rescaled), cv2.COLORMAP_JET
        )

        # Blend the heatmap overlay with the original image
        grad_cam_visualization = cv2.addWeighted(
            np.transpose(image.squeeze().cpu().numpy(), (1, 2, 0)),
            0.5,
            heatmap_overlay,
            0.5,
            0,
        )

# Show or save the Grad-CAM visualization
