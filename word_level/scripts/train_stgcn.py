import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

sys.path.insert(0, PROJECT_ROOT)

from word_level.models.stgcn_model import STGCNModel


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

SEQUENCE_LENGTH = 30
NUM_NODES = 21
NUM_CHANNELS = 3
NUM_CLASSES = 10

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

PATIENCE = 15

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "word_level",
    "models",
    "stgcn_improved_model.pt"
)

X_PATH = os.path.join(
    PROJECT_ROOT,
    "word_level",
    "data",
    "X.npy"
)

Y_PATH = os.path.join(
    PROJECT_ROOT,
    "word_level",
    "data",
    "y.npy"
)


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


# ============================================================
# LANDMARK NORMALIZATION
# ============================================================

def normalize_landmarks(X):
    """
    Normalize each sequence relative to the wrist.

    X shape:
        (samples, frames, 21, 3)
    """

    X = X.copy()

    # Wrist landmark = landmark 0
    wrist = X[:, :, 0:1, :]

    # Move all landmarks relative to wrist
    X = X - wrist

    # Scale using maximum distance from wrist
    distances = np.linalg.norm(X, axis=-1)

    scale = np.max(distances, axis=(1, 2), keepdims=True)

    scale[scale < 1e-6] = 1.0

    X = X / scale[:, :, :, None]

    return X.astype(np.float32)


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_sequence(sequence):
    """
    Apply small realistic variations to a hand-sign sequence.
    """

    augmented = sequence.copy()

    # Small Gaussian noise
    noise = np.random.normal(
        0,
        0.005,
        augmented.shape
    ).astype(np.float32)

    augmented += noise

    # Small spatial scaling
    scale = np.random.uniform(0.95, 1.05)

    augmented *= scale

    return augmented.astype(np.float32)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("IMPROVED ST-GCN TRAINING")
print("=" * 60)

print("\nLoading dataset...")

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)


# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing hand landmarks...")

X = normalize_landmarks(X)

print("Normalization complete.")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating stratified dataset split...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

print("Training samples:  ", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# AUGMENT TRAINING DATA
# ============================================================

print("\nApplying data augmentation...")

augmented_X = []
augmented_y = []

for sequence, label in zip(X_train, y_train):

    # Keep original
    augmented_X.append(sequence)
    augmented_y.append(label)

    # Add one augmented version
    augmented_X.append(
        augment_sequence(sequence)
    )
    augmented_y.append(label)


X_train = np.array(
    augmented_X,
    dtype=np.float32
)

y_train = np.array(
    augmented_y,
    dtype=np.int64
)

print(
    "Training samples after augmentation:",
    len(X_train)
)


# ============================================================
# CONVERT TO PYTORCH
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.long
)


train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)


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
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing device:", device)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating ST-GCN model...")

model = STGCNModel(
    input_channels=NUM_CHANNELS,
    hidden_channels=64,
    num_classes=NUM_CLASSES,
    dropout=0.3
)

model = model.to(device)

print(model)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING IMPROVED TRAINING")
print("=" * 60)

best_val_accuracy = 0.0
best_model_state = None

epochs_without_improvement = 0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
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

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == batch_y
        ).sum().item()

        total += batch_y.size(0)

    train_loss = running_loss / len(train_loader)

    train_accuracy = (
        100.0 * correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == batch_y
            ).sum().item()

            val_total += batch_y.size(0)

    val_accuracy = (
        100.0 * val_correct / val_total
    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(val_accuracy)

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Acc: {val_accuracy:.2f}% "
        f"LR: {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_model_state,
            MODEL_PATH
        )

        epochs_without_improvement = 0

        print(
            f"   ✓ Best model saved "
            f"({best_val_accuracy:.2f}%)"
        )

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping triggered at "
            f"epoch {epoch + 1}."
        )

        break


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("IMPROVED ST-GCN TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print("\nImproved model saved to:")

print(MODEL_PATH)