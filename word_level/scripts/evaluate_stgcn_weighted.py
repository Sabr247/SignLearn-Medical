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

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_DIR = os.path.join(BASE_DIR, "word_level", "data")
MODEL_DIR = os.path.join(BASE_DIR, "word_level", "models")

X_FILE = os.path.join(DATA_DIR, "X.npy")
Y_FILE = os.path.join(DATA_DIR, "y.npy")

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "stgcn_weighted_model.pt"
)

# Allow Python to find the model file
sys.path.append(MODEL_DIR)

from stgcn_model import STGCNModel


# --------------------------------------------------
# SETTINGS
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
print("WEIGHTED ST-GCN EVALUATION")
print("=" * 60)

X = np.load(X_FILE)
y = np.load(Y_FILE)

print("X shape:", X.shape)
print("y shape:", y.shape)


# --------------------------------------------------
# SAME TRAIN/VALIDATION SPLIT
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nValidation samples:", len(X_val))


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

device = torch.device("cpu")

model = STGCNModel(
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print("Weighted ST-GCN model loaded successfully.")


# --------------------------------------------------
# MAKE PREDICTIONS
# --------------------------------------------------

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
).to(device)

with torch.no_grad():

    outputs = model(X_val_tensor)

    predictions = torch.argmax(
        outputs,
        dim=1
    ).cpu().numpy()


# --------------------------------------------------
# ACCURACY
# --------------------------------------------------

accuracy = accuracy_score(
    y_val,
    predictions
)

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(
    f"\nValidation Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print("\nClassification Report:")
print("-" * 60)

print(
    classification_report(
        y_val,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=WORDS,
        zero_division=0
    )
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_val,
    predictions,
    labels=list(range(NUM_CLASSES))
)

print("\nConfusion Matrix:")
print("-" * 60)

print(
    f"{'':10}",
    " ".join(f"{word[:7]:>8}" for word in WORDS)
)

for i, word in enumerate(WORDS):

    print(
        f"{word:<10}",
        " ".join(
            f"{value:>8}"
            for value in cm[i]
        )
    )


# --------------------------------------------------
# PREDICTION DETAILS
# --------------------------------------------------

print("\nPrediction Details:")
print("-" * 60)

for actual, predicted in zip(
    y_val,
    predictions
):

    actual_word = WORDS[actual]
    predicted_word = WORDS[predicted]

    status = "CORRECT" if actual == predicted else "WRONG"

    print(
        f"Actual: {actual_word:<10} "
        f"Predicted: {predicted_word:<10} "
        f"{status}"
    )


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

RESULT_FILE = os.path.join(
    BASE_DIR,
    "word_level",
    "stgcn_weighted_evaluation.txt"
)

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "WEIGHTED ST-GCN EVALUATION\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        f"Validation Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report:\n"
    )

    file.write(
        classification_report(
            y_val,
            predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=WORDS,
            zero_division=0
        )
    )

    file.write(
        "\nConfusion Matrix:\n"
    )

    file.write(
        str(cm)
    )

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(
    "\nResults saved to:",
    RESULT_FILE
)