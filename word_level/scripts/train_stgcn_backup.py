import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.append(PROJECT_ROOT)

from word_level.models.stgcn_model import STGCNModel


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "word_level", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "word_level", "models")

X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "stgcn_standard_model.pt"
)

NUM_CLASSES = 10
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.001

RANDOM_STATE = 42


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
print("=" * 60)
print("STANDARD ST-GCN TRAINING")
print("=" * 60)

X = np.load(X_PATH)
y = np.load(Y_PATH)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ---------------------------------------------------------
# Stratified train-validation split
# ---------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nDataset split:")
print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")


# ---------------------------------------------------------
# Convert to PyTorch tensors
# ---------------------------------------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)


# ---------------------------------------------------------
# Create DataLoaders
# ---------------------------------------------------------
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ---------------------------------------------------------
# Create model
# ---------------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")

model = STGCNModel(
    num_classes=NUM_CLASSES
).to(device)


# ---------------------------------------------------------
# Standard CrossEntropyLoss
# ---------------------------------------------------------
criterion = nn.CrossEntropyLoss()


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
best_val_accuracy = 0.0
best_model_state = None

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for epoch in range(EPOCHS):

    # -----------------------------
    # Training
    # -----------------------------
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == batch_y
        ).sum().item()

        total += batch_y.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total


    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == batch_y
            ).sum().item()

            val_total += batch_y.size(0)

    val_accuracy = 100 * val_correct / val_total


    # -----------------------------
    # Save best model
    # -----------------------------
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_state = {
            key: value.cpu().clone()
            for key, value in model.state_dict().items()
        }


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {val_accuracy:.2f}%"
    )


# ---------------------------------------------------------
# Save best model
# ---------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

checkpoint = {
    "model_state_dict": best_model_state,
    "num_classes": NUM_CLASSES,
    "input_shape": X.shape[1:],
    "best_val_accuracy": best_val_accuracy,
    "random_state": RANDOM_STATE,
    "stratified_split": True,
    "class_weighted": False
}

torch.save(
    checkpoint,
    MODEL_PATH
)


# ---------------------------------------------------------
# Training summary
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STANDARD ST-GCN TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"\nStandard model saved to:\n"
    f"{MODEL_PATH}"
)