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

# Allow Python to find the model
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models"
    )
)

from bilstm_model import BiLSTMModel


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

X_FILE = os.path.join(DATA_DIR, "X.npy")
Y_FILE = os.path.join(DATA_DIR, "y.npy")
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "bilstm_model.pt"
)


# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

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


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 60)
print("BiLSTM WORD-LEVEL MODEL EVALUATION")
print("=" * 60)

print("\nLoading dataset...")

X = np.load(X_FILE)
y = np.load(Y_FILE)

print("X shape:", X.shape)
print("y shape:", y.shape)


# --------------------------------------------------
# RECREATE SAME VALIDATION SPLIT
# --------------------------------------------------

print("\nCreating validation split...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("\nLoading trained BiLSTM model...")

model = BiLSTMModel(
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=torch.device("cpu")
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully!")


# --------------------------------------------------
# MAKE PREDICTIONS
# --------------------------------------------------

print("\nMaking predictions...")

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

with torch.no_grad():

    outputs = model(X_val_tensor)

    predictions = torch.argmax(
        outputs,
        dim=1
    ).numpy()


# --------------------------------------------------
# OVERALL ACCURACY
# --------------------------------------------------

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


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

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


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

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

print(" " * 12, end="")

for word in WORDS:
    print(f"{word[:5]:>7}", end="")

print()

for i, word in enumerate(WORDS):

    print(f"{word:<12}", end="")

    for value in cm[i]:
        print(f"{value:>7}", end="")

    print()


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

RESULTS_FILE = os.path.join(
    BASE_DIR,
    "bilstm_evaluation.txt"
)

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "BiLSTM WORD-LEVEL MODEL EVALUATION\n"
    )

    file.write("=" * 60 + "\n")

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


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print("\nResults saved to:")
print(RESULTS_FILE)