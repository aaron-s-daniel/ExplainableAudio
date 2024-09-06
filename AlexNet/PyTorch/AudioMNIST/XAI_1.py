import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, Saliency, GuidedBackprop
import quantus
import seaborn as sns
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Create results directory
results_dir = 'audiomnist_xai_results'
os.makedirs(results_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define AlexNet model (same as in the training script)
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load the trained model
model = AlexNet().to(device)
model.load_state_dict(torch.load('best_audiomnist_alexnet.pth', map_location=device))
model.eval()

# Load a small subset of the test data for XAI experiments
# This assumes you have a saved tensor of preprocessed test data
X_test = torch.load('audiomnist_test_spectrograms.pt')
y_test = torch.load('audiomnist_test_labels.pt')
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# XAI methods
xai_methods = {
    "Saliency": Saliency(model),
    "IntegratedGradients": IntegratedGradients(model),
    "GuidedBackprop": GuidedBackprop(model)
}

# Generate explanations
def explainer_wrapper(**kwargs):
    method = kwargs.pop("method")
    inputs = kwargs.pop("inputs")
    targets = kwargs.pop("targets")
    if isinstance(inputs, np.ndarray):
        inputs = torch.FloatTensor(inputs).to(device)
    if isinstance(targets, np.ndarray):
        targets = torch.LongTensor(targets).to(device)
    return xai_methods[method].attribute(inputs, target=targets).cpu().detach().numpy()

# Get a batch of test data
x_batch, y_batch = next(iter(test_loader))
x_batch, y_batch = x_batch.to(device), y_batch.to(device)

# Generate explanations
explanations = {method: explainer_wrapper(method=method, inputs=x_batch, targets=y_batch) for method in xai_methods}

# Define Quantus metrics
metrics = {
    "Robustness": quantus.AvgSensitivity(
        nr_samples=10,
        perturb_func=quantus.perturb_func.uniform_noise,
        similarity_func=quantus.similarity_func.difference,
        disable_warnings=True,
    ),
    "Faithfulness": quantus.FaithfulnessCorrelation(
        nr_runs=10,
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        disable_warnings=True,
    ),
    "Complexity": quantus.Sparseness(
        disable_warnings=True,
    ),
    "Randomisation": quantus.RandomLogit(
        num_classes=10,
        similarity_func=quantus.similarity_func.ssim,
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
with open(os.path.join(results_dir, 'audiomnist_quantus_results.json'), 'w') as f:
    json.dump(results, f, indent=4)

# Visualize results
df = pd.DataFrame.from_dict(results, orient='index')
df_normalized = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_normalized_rank = df_normalized.rank()

plt.figure(figsize=(12, 8))
sns.heatmap(df_normalized_rank, annot=True, cmap='YlGnBu', fmt='.0f')
plt.title('Ranking of Explanation Methods across Metrics for AudioMNIST')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'audiomnist_method_ranking_heatmap.png'))
plt.close()

# Visualize explanations
def plot_explanation(spectrogram, explanation, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(spectrogram.squeeze(), cmap='viridis')
    plt.title("Original Spectrogram")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(explanation.squeeze(), cmap='seismic', clim=(-1, 1))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'audiomnist_{title.replace(" ", "_")}_explanation.png'))
    plt.close()

for method, attr in explanations.items():
    plot_explanation(x_batch[0].cpu().numpy(), attr[0], method)

print(f"\nResults and plots have been saved to the '{results_dir}' directory.")