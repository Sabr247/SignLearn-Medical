import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.append(PROJECT_ROOT)

from word_level.models.stgcn_model import STGCNModel


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "word_level",
    "data"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "word_level",
    "models"
)

X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "stgcn_standard_model.pt"
)


# ---------------------------------------------------------
# Class names
# ---------------------------------------------------------
CLASS_NAMES = [
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

NUM_CLASSES = 10
RANDOM_STATE = 42


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
print("=" * 60)
print("STANDARD ST-GCN WORD-LEVEL MODEL EVALUATION")
print("=" * 60)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
print("\nLoading dataset...")

X = np.load(X_PATH)
y = np.load(Y_PATH)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ---------------------------------------------------------
# Same stratified split used during training
# ---------------------------------------------------------
print("\nCreating validation split...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")


# ---------------------------------------------------------
# Convert validation data to tensors
# ---------------------------------------------------------
X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.long
)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")


# ---------------------------------------------------------
# Create model
# ---------------------------------------------------------
model = STGCNModel(
    num_classes=NUM_CLASSES
).to(device)


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
print("\nLoading trained standard ST-GCN model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Support checkpoint format
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(checkpoint)

model.eval()

print("Standard ST-GCN model loaded successfully.")


# ---------------------------------------------------------
# Make predictions
# ---------------------------------------------------------
print("\nMaking predictions...")

X_val = X_val.to(device)

with torch.no_grad():

    outputs = model(X_val)

    predictions = outputs.argmax(
        dim=1
    ).cpu().numpy()

actual = y_val.numpy()


# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------
accuracy = (
    np.sum(predictions == actual)
    / len(actual)
) * 100


print("\n" + "=" * 60)
print("OVERALL PERFORMANCE")
print("=" * 60)

print(
    f"Validation Accuracy: {accuracy:.2f}%"
)


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        actual,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    actual,
    predictions,
    labels=list(range(NUM_CLASSES))
)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    f"{'':12}",
    end=""
)

for name in CLASS_NAMES:
    print(
        f"{name[:8]:>10}",
        end=""
    )

print()

for i, row in enumerate(cm):

    print(
        f"{CLASS_NAMES[i]:12}",
        end=""
    )

    for value in row:

        print(
            f"{value:10d}",
            end=""
        )

    print()


# ---------------------------------------------------------
# Prediction details
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("PREDICTION DETAILS")
print("=" * 60)

for actual_label, predicted_label in zip(
    actual,
    predictions
):

    status = (
        "CORRECT"
        if actual_label == predicted_label
        else "WRONG"
    )

    print(
        f"Actual: {CLASS_NAMES[actual_label]:10} "
        f"Predicted: {CLASS_NAMES[predicted_label]:10} "
        f"{status}"
    )


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------
RESULT_PATH = os.path.join(
    PROJECT_ROOT,
    "word_level",
    "stgcn_standard_evaluation.txt"
)

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STANDARD ST-GCN WORD-LEVEL EVALUATION\n"
    )

    f.write("=" * 60 + "\n")

    f.write(
        f"Validation Accuracy: {accuracy:.2f}%\n\n"
    )

    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 60 + "\n")

    f.write(
        classification_report(
            actual,
            predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    f.write("\nCONFUSION MATRIX\n")
    f.write("-" * 60 + "\n")

    for row in cm:
        f.write(
            " ".join(map(str, row)) + "\n"
        )

    f.write("\nPREDICTION DETAILS\n")
    f.write("-" * 60 + "\n")

    for actual_label, predicted_label in zip(
        actual,
        predictions
    ):

        status = (
            "CORRECT"
            if actual_label == predicted_label
            else "WRONG"
        )

        f.write(
            f"Actual: {CLASS_NAMES[actual_label]:10} "
            f"Predicted: {CLASS_NAMES[predicted_label]:10} "
            f"{status}\n"
        )


# ---------------------------------------------------------
# Complete
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(
    f"\nResults saved to:\n{RESULT_PATH}"
)