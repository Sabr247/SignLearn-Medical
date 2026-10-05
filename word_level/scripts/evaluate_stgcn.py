import os
import sys
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# IMPORT ST-GCN MODEL
# ============================================================

sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        ),
        "models"
    )
)

from stgcn_model import STGCNModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

X_FILE = os.path.join(
    DATA_DIR,
    "X.npy"
)

Y_FILE = os.path.join(
    DATA_DIR,
    "y.npy"
)


# IMPORTANT:
# Use the improved model that achieved 58.33%

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "stgcn_improved_model.pt"
)


# ============================================================
# CLASS NAMES
# ============================================================

WORDS = [
    "HELLO",
    "HELP",
    "WATER",
    "FOOD",
    "PAIN",
    "DOCTOR",
    "MEDICINE",
    "HOSPITAL",
    "YES",
    "NO"
]

NUM_CLASSES = len(WORDS)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_sequences(X):
    """
    Apply the same landmark normalization used during
    improved ST-GCN training.

    For every sequence:
    1. Use the wrist as the origin.
    2. Center all landmarks around the wrist.
    3. Scale using the maximum landmark distance.
    """

    X = X.copy().astype(np.float32)

    # Wrist landmark = node 0
    wrist = X[:, :, 0:1, :]

    # Move wrist to origin
    X = X - wrist

    # Calculate distance of every landmark from wrist
    distances = np.linalg.norm(
        X,
        axis=-1
    )

    # Maximum distance for each sequence
    scale = np.max(
        distances,
        axis=(1, 2),
        keepdims=True
    )

    # Avoid division by zero
    scale[scale < 1e-6] = 1.0

    # Normalize
    X = X / scale[:, :, :, None]

    return X.astype(np.float32)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("ST-GCN WORD-LEVEL MODEL EVALUATION")
print("=" * 60)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

X = np.load(X_FILE)
y = np.load(Y_FILE)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# NORMALIZE DATA
# ============================================================

print("\nApplying training normalization...")

X = normalize_sequences(X)

print("Normalization complete.")


# ============================================================
# CREATE SAME VALIDATION SPLIT
# ============================================================

print("\nCreating validation split...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)


# ============================================================
# LOAD TRAINED ST-GCN
# ============================================================

print("\nLoading improved ST-GCN model...")

model = STGCNModel(
    input_channels=3,
    hidden_channels=64,
    num_classes=NUM_CLASSES,
    dropout=0.3
)


checkpoint = torch.load(
    MODEL_FILE,
    map_location=torch.device("cpu"),
    weights_only=False
)


# ============================================================
# HANDLE CHECKPOINT FORMAT
# ============================================================

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()

print("Model loaded successfully!")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\nMaking predictions...")

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)


with torch.no_grad():

    outputs = model(
        X_val_tensor
    )

    predictions = torch.argmax(
        outputs,
        dim=1
    ).numpy()


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_val,
    predictions
)


print("\n" + "=" * 60)
print("OVERALL PERFORMANCE")
print("=" * 60)

print(
    f"Validation Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)


report = classification_report(
    y_val,
    predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=WORDS,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_val,
    predictions,
    labels=list(range(NUM_CLASSES))
)


print("\nRows = Actual")
print("Columns = Predicted\n")


print(
    " " * 12,
    end=""
)

for word in WORDS:

    print(
        f"{word[:5]:>7}",
        end=""
    )

print()


for i, word in enumerate(WORDS):

    print(
        f"{word:<12}",
        end=""
    )

    for value in cm[i]:

        print(
            f"{value:>7}",
            end=""
        )

    print()


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_FILE = os.path.join(
    BASE_DIR,
    "stgcn_evaluation.txt"
)


with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "ST-GCN WORD-LEVEL MODEL EVALUATION\n"
    )

    file.write(
        "=" * 60 + "\n"
    )

    file.write(
        f"Validation Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        report
    )

    file.write(
        "\nCONFUSION MATRIX\n"
    )

    file.write(
        str(cm)
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print("\nResults saved to:")

print(
    RESULTS_FILE
)