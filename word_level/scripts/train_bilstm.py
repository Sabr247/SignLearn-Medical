import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
))

from bilstm_model import BiLSTMModel

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

X_FILE = os.path.join(DATA_DIR, "X.npy")
Y_FILE = os.path.join(DATA_DIR, "y.npy")

MODEL_FILE = os.path.join(MODEL_DIR, "bilstm_model.pt")

NUM_CLASSES = 10
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.001


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("BiLSTM WORD-LEVEL SIGN LANGUAGE TRAINING")
print("=" * 60)

print("\nLoading dataset...")

X = np.load(X_FILE)
y = np.load(Y_FILE)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# CONVERT TO PYTORCH TENSORS
# ============================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)


# ============================================================
# DATA LOADERS
# ============================================================

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


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating BiLSTM model...")

model = BiLSTMModel(num_classes=NUM_CLASSES)

print(model)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...")
print("=" * 60)

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    # -------------------------
    # Training
    # -------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_accuracy = train_correct / train_total


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total


    # -------------------------
    # Print progress
    # -------------------------

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy * 100:.2f}% "
        f"Val Loss: {val_loss / len(val_loader):.4f} "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )


    # -------------------------
    # Save best model
    # -------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "input_shape": list(X.shape[1:]),
                "best_val_accuracy": best_val_accuracy
            },
            MODEL_FILE
        )

        print(
            f"   ✓ Best model saved "
            f"(Validation Accuracy: {best_val_accuracy * 100:.2f}%)"
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print("Model saved to:")
print(MODEL_FILE)