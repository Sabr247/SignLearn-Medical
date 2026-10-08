import os
import numpy as np

# Find the project root folder
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_DIR = os.path.join(BASE_DIR, "word_level", "data")

X_FILE = os.path.join(DATA_DIR, "X.npy")
Y_FILE = os.path.join(DATA_DIR, "y.npy")

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

# Load the prepared dataset
X = np.load(X_FILE)
y = np.load(Y_FILE)

print("=" * 50)
print("WORD-LEVEL DATASET ANALYSIS")
print("=" * 50)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nDataset details:")
print("Number of sequences:", len(X))
print("Frames per sequence:", X.shape[1])
print("Landmarks per frame:", X.shape[2])
print("Coordinates per landmark:", X.shape[3])

print("\nClass distribution:")
print("-" * 30)

for label, word in enumerate(WORDS):

    count = np.sum(y == label)

    percentage = (count / len(y)) * 100

    print(
        f"{label}: {word:<10} "
        f"{count:>2} samples "
        f"({percentage:.1f}%)"
    )

print("\n" + "=" * 50)
print("DATASET ANALYSIS COMPLETE")
print("=" * 50)