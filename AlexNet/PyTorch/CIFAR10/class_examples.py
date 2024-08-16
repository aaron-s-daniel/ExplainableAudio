import os
import warnings
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, Saliency, GuidedBackprop

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Create results directory
results_dir = os.environ.get('RESULTS_DIR', 'quantus_results')
os.makedirs(results_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model (assuming you have this from your previous script)
# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('cifar10_cnn.pth', map_location=device))
model.eval()

# Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1000, shuffle=True, num_workers=2)

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define Captum attribution methods
xai_methods = {
    "Saliency": Saliency(model),
    "IntegratedGradients": IntegratedGradients(model),
    "GuidedBackprop": GuidedBackprop(model)
}

def get_example_and_attribution(target_class):
    for images, labels in testloader:
        class_mask = (labels == target_class)
        if class_mask.sum() > 0:
            image = images[class_mask][0].unsqueeze(0).to(device)
            break
    
    attributions = {}
    for method_name, method in xai_methods.items():
        attribution = method.attribute(image, target=target_class)
        attributions[method_name] = attribution.squeeze().cpu().detach().numpy()
    
    return image.squeeze().cpu().numpy(), attributions

def plot_image_and_attributions(image, attributions, class_name):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axs[0].imshow(np.transpose(image, (1, 2, 0)))
    axs[0].set_title(f"Original Image\nClass: {class_name}")
    axs[0].axis('off')
    
    # Attributions
    for idx, (method_name, attr) in enumerate(attributions.items(), start=1):
        attr_sum = np.sum(attr, axis=0)
        vmin, vmax = attr_sum.min(), attr_sum.max()
        axs[idx].imshow(attr_sum, cmap='seismic', vmin=-vmax, vmax=vmax)
        axs[idx].set_title(f"{method_name}")
        axs[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'attribution_study_{class_name}.png'))
    plt.close()

# Generate and save attribution studies for each class
for class_idx, class_name in enumerate(classes):
    print(f"Generating attribution study for class: {class_name}")
    image, attributions = get_example_and_attribution(class_idx)
    plot_image_and_attributions(image, attributions, class_name)

print(f"\nAttribution studies have been saved to the '{results_dir}' directory.")