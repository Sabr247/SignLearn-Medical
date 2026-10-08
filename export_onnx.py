"""
Export the BiLSTM word recognition model to ONNX format.
Run once locally: python export_onnx.py
The output file word_bilstm.onnx is then committed to the repo.
"""

import torch
import torch.nn as nn
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORD_MODEL_FILE = os.path.join(BASE_DIR, "best_word_bilstm.pt")
WORD_CONFIG_FILE = os.path.join(BASE_DIR, "word_model_config.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "word_bilstm.onnx")


class SignWordBiLSTM(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=150,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, 64, 150) — already flattened sequence
        output, _ = self.lstm(x)
        x = output[:, -1, :]
        x = self.classifier(x)
        return x


with open(WORD_CONFIG_FILE, "r") as f:
    config = json.load(f)

num_classes = config["num_classes"]
print(f"Number of classes: {num_classes}")

model = SignWordBiLSTM(num_classes=num_classes)
checkpoint = torch.load(WORD_MODEL_FILE, map_location="cpu")

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

# The saved model uses the old forward pass that takes (1, 3, 64, 50)
# and permutes/reshapes inside forward(). We need to match that exactly.
# Re-define with original forward for export:

class SignWordBiLSTMOriginal(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=150,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input: (batch, channels=3, frames=64, landmarks=50)
        x = x.permute(0, 2, 1, 3)          # → (batch, 64, 3, 50)
        x = x.reshape(x.size(0), x.size(1), -1)  # → (batch, 64, 150)
        output, _ = self.lstm(x)
        x = output[:, -1, :]
        x = self.classifier(x)
        return x

model = SignWordBiLSTMOriginal(num_classes=num_classes)
model.load_state_dict(state_dict)
model.eval()

# Dummy input matching what the Python code sends: (1, 3, 64, 50)
dummy_input = torch.zeros(1, 3, 64, 50, dtype=torch.float32)

print("Exporting to ONNX...")
# Use dynamo=False to force the legacy exporter which keeps all weights
# embedded in a single .onnx file (no external .data sidecar file).
torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_FILE,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    dynamo=False
)

print(f"Exported to: {OUTPUT_FILE}")
print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")
