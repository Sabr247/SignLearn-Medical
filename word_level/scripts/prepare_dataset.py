import os
import numpy as np

# Path to the word-level data folder
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_DIR = os.path.join(BASE_DIR, "word_level", "data")

# Words/classes in our project
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

# Store all sequences and their labels
X = []
y = []

print("=" * 50)
print("PREPARING WORD-LEVEL DATASET")
print("=" * 50)

for label, word in enumerate(WORDS):

    word_folder = os.path.join(DATA_DIR, word)

    if not os.path.exists(word_folder):
        print(f"WARNING: Folder not found: {word}")
        continue

    files = sorted(
        file for file in os.listdir(word_folder)
        if file.endswith(".npy")
    )

    print(f"{word}: {len(files)} samples")

    for file in files:

        file_path = os.path.join(word_folder, file)

        sequence = np.load(file_path)

        # Check that the sequence has the expected shape
        if sequence.shape != (30, 21, 3):
            print(
                f"WARNING: {file} has shape "
                f"{sequence.shape}"
            )
            continue

        X.append(sequence)
        y.append(label)

# Convert lists to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("\n" + "=" * 50)
print("DATASET SUMMARY")
print("=" * 50)

print("Total sequences:", len(X))
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nClass labels:")

for label, word in enumerate(WORDS):
    count = np.sum(y == label)
    print(f"{label}: {word} -> {count} samples")
    # Save the prepared dataset
X_FILE = os.path.join(DATA_DIR, "X.npy")
Y_FILE = os.path.join(DATA_DIR, "y.npy")

np.save(X_FILE, X)
np.save(Y_FILE, y)

print("\nDataset saved successfully!")
print("X saved to:", X_FILE)
print("y saved to:", Y_FILE)