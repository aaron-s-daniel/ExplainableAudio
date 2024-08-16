import os
import warnings
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import quantus
from captum.attr import IntegratedGradients, Saliency, GuidedBackprop
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Create results directory
results_dir = os.environ.get('RESULTS_DIR', 'quantus_results')
os.makedirs(results_dir, exist_ok=True)

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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('cifar10_cnn.pth', map_location=device))
model.eval()

# Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Get a batch of test data
x_batch, y_batch = next(iter(testloader))
x_batch, y_batch = x_batch.to(device), y_batch.to(device)

# Define Captum attribution methods
xai_methods = {
    "Saliency": Saliency(model),
    "IntegratedGradients": IntegratedGradients(model),
    "GuidedBackprop": GuidedBackprop(model)
}

# Function to generate explanations
def explainer_wrapper(**kwargs):
    method = kwargs.pop("method")
    inputs = kwargs.pop("inputs")
    targets = kwargs.pop("targets")
    if isinstance(inputs, np.ndarray):
        inputs = torch.tensor(inputs).to(device)
    if isinstance(targets, np.ndarray):
        targets = torch.tensor(targets).long().to(device)
    return xai_methods[method].attribute(inputs, target=targets).cpu().numpy()

# Generate explanations
explanations = {}
for method in xai_methods:
    print(f"Generating explanations for {method}...")
    explanations[method] = explainer_wrapper(method=method, inputs=x_batch, targets=y_batch)

# Custom Sparseness metric
class MultiDimensionalSparseness(quantus.Sparseness):
    def evaluate_instance(self, x: np.ndarray, a: np.ndarray) -> float:
        a = np.abs(a)
        return 1 - (np.sum(a) ** 2) / (np.sum(a ** 2) * a.size)

# Define Quantus metrics
metrics = {
    "Robustness": quantus.AvgSensitivity(
        nr_samples=10,
        lower_bound=0.2,
        norm_numerator=quantus.norm_func.fro_norm,
        norm_denominator=quantus.norm_func.fro_norm,
        perturb_func=quantus.perturb_func.uniform_noise,
        similarity_func=quantus.similarity_func.difference,
        abs=False,
        normalise=False,
        disable_warnings=True,
    ),
    "Faithfulness": quantus.FaithfulnessCorrelation(
        nr_runs=10,
        subset_size=224,
        perturb_baseline="black",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        abs=False,
        normalise=False,
        disable_warnings=True,
    ),
    "Complexity": MultiDimensionalSparseness(
        abs=True,
        normalise=False,
        disable_warnings=True,
    ),
    "Randomisation": quantus.RandomLogit(
        num_classes=10,
        similarity_func=quantus.similarity_func.ssim,
        abs=True,
        normalise=False,
        disable_warnings=True,
    ),
}

# Evaluate explanations
results = {method: {} for method in xai_methods}
for method in xai_methods:
    for metric_name, metric_func in metrics.items():
        print(f"Evaluating {metric_name} of {method} method.")
        scores = metric_func(
            model=model,
            x_batch=x_batch.cpu().numpy(),
            y_batch=y_batch.cpu().numpy(),
            a_batch=explanations[method],
            **{"device": device, "explain_func": explainer_wrapper, "explain_func_kwargs": {"method": method}}
        )
        results[method][metric_name] = np.mean(scores)

# Save results
import json
with open(os.path.join(results_dir, 'quantus_results.json'), 'w') as f:
    json.dump(results, f, indent=4)

# Visualize results
df = pd.DataFrame.from_dict(results, orient='index')
df_normalized = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_normalized_rank = df_normalized.rank()

plt.figure(figsize=(12, 8))
sns.heatmap(df_normalized_rank, annot=True, cmap='YlGnBu', fmt='.0f')
plt.title('Ranking of Explanation Methods across Metrics')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'method_ranking_heatmap.png'))
plt.close()

# Visualize explanations
def plot_explanation(image, explanation, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.transpose(1, 2, 0))
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(explanation.sum(axis=0), cmap='seismic', clim=(-1, 1))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{title.replace(" ", "_")}_explanation.png'))
    plt.close()

for method, attr in explanations.items():
    plot_explanation(x_batch[0].cpu().numpy(), attr[0], method)

print(f"\nResults and plots have been saved to the '{results_dir}' directory.")