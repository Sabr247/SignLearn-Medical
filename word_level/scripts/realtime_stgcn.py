import os
import sys
import cv2
import numpy as np
import torch
import mediapipe as mp

from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "word_level",
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "stgcn_improved_model.pt"
)

HAND_MODEL_FILE = os.path.join(
    BASE_DIR,
    "hand_landmarker.task"
)

sys.path.append(MODEL_DIR)

from stgcn_model import STGCNModel


# ============================================================
# SETTINGS
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

SEQUENCE_LENGTH = 30

# Minimum confidence required
CONFIDENCE_THRESHOLD = 0.50

# Number of consecutive predictions required
# before changing the displayed word
SMOOTHING_WINDOW = 5

# Number of consecutive identical predictions
# required for a stable recognition
REQUIRED_STABLE_PREDICTIONS = 3


# ============================================================
# LANDMARK NORMALIZATION
# ============================================================

def normalize_sequence(sequence):
    """
    Apply the same normalization used during training.

    Steps:
    1. Use wrist landmark as origin.
    2. Center all landmarks around wrist.
    3. Scale using maximum landmark distance.
    """

    sequence = sequence.copy().astype(np.float32)

    # Landmark 0 = wrist
    wrist = sequence[:, 0:1, :]

    # Move wrist to origin
    sequence = sequence - wrist

    # Distance of every landmark from wrist
    distances = np.linalg.norm(
        sequence,
        axis=-1
    )

    # Maximum distance across the entire sequence
    scale = np.max(distances)

    if scale < 1e-6:
        scale = 1.0

    sequence = sequence / scale

    return sequence.astype(np.float32)


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cpu")

print("=" * 60)
print("REAL-TIME ST-GCN WORD RECOGNITION")
print("=" * 60)

print("\nLoading model...")

model = STGCNModel(
    input_channels=3,
    hidden_channels=64,
    num_classes=NUM_CLASSES,
    dropout=0.3
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

# Support both checkpoint formats
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

model.to(device)
model.eval()

print("ST-GCN model loaded successfully.")

print("\nWords:")
print(", ".join(WORDS))

print("\nPress Q to quit.")

print("=" * 60)


# ============================================================
# CHECK HAND LANDMARK MODEL
# ============================================================

if not os.path.exists(HAND_MODEL_FILE):

    print("\nERROR: hand_landmarker.task was not found.")

    print(
        "Expected location:",
        HAND_MODEL_FILE
    )

    raise SystemExit


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL_FILE
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

    landmarker.close()

    raise SystemExit


# ============================================================
# VARIABLES
# ============================================================

sequence = []

prediction_history = deque(
    maxlen=SMOOTHING_WINDOW
)

prediction_text = "Show your hand"

confidence_text = ""

stable_word = None

stable_count = 0

frame_timestamp = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read webcam frame."
        )

        break


    # --------------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MEDIA PIPE IMAGE
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    frame_timestamp += 33


    # --------------------------------------------------------
    # DETECT HAND
    # --------------------------------------------------------

    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )


    landmarks = None


    # --------------------------------------------------------
    # EXTRACT LANDMARKS
    # --------------------------------------------------------

    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        landmarks = []

        for landmark in hand_landmarks:

            landmarks.append([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        landmarks = np.array(
            landmarks,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # DRAW LANDMARKS
        # ----------------------------------------------------

        for landmark in hand_landmarks:

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


        # ----------------------------------------------------
        # HAND CONNECTIONS
        # ----------------------------------------------------

        connections = [

            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),

            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),

            (5, 9),
            (9, 10),
            (10, 11),
            (11, 12),

            (9, 13),
            (13, 14),
            (14, 15),
            (15, 16),

            (13, 17),
            (17, 18),
            (18, 19),
            (19, 20),

            (0, 17)
        ]


        for start, end in connections:

            x1 = int(
                hand_landmarks[start].x
                * frame.shape[1]
            )

            y1 = int(
                hand_landmarks[start].y
                * frame.shape[0]
            )

            x2 = int(
                hand_landmarks[end].x
                * frame.shape[1]
            )

            y2 = int(
                hand_landmarks[end].y
                * frame.shape[0]
            )

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


    # --------------------------------------------------------
    # BUILD SEQUENCE
    # --------------------------------------------------------

    if landmarks is not None:

        sequence.append(
            landmarks
        )

        if len(sequence) > SEQUENCE_LENGTH:

            sequence.pop(0)

    else:

        # No hand detected
        sequence = []

        prediction_history.clear()

        stable_word = None

        stable_count = 0

        prediction_text = "Show your hand"

        confidence_text = ""


    # ========================================================
    # PREDICTION
    # ========================================================

    if len(sequence) == SEQUENCE_LENGTH:

        # Convert to NumPy
        input_data = np.array(
            sequence,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        input_data = normalize_sequence(
            input_data
        )


        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        # ----------------------------------------------------

        input_tensor = torch.tensor(
            input_data,
            dtype=torch.float32
        ).unsqueeze(0)


        input_tensor = input_tensor.to(
            device
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_class = torch.max(
                probabilities,
                dim=1
            )


        confidence_value = confidence.item()

        predicted_index = predicted_class.item()

        predicted_word = WORDS[
            predicted_index
        ]


        # ----------------------------------------------------
        # CONFIDENCE FILTER
        # ----------------------------------------------------

        if confidence_value >= CONFIDENCE_THRESHOLD:

            # Add prediction to history
            prediction_history.append(
                predicted_word
            )

            # Count how many times this prediction
            # has appeared recently
            same_predictions = prediction_history.count(
                predicted_word
            )


            # ------------------------------------------------
            # STABILITY CHECK
            # ------------------------------------------------

            if same_predictions >= REQUIRED_STABLE_PREDICTIONS:

                if stable_word == predicted_word:

                    stable_count += 1

                else:

                    stable_word = predicted_word

                    stable_count = 1


                # Display stable prediction
                prediction_text = stable_word

            confidence_text = (
                f"Confidence: "
                f"{confidence_value * 100:.1f}%"
            )


        else:

            confidence_text = (
                f"Low confidence: "
                f"{confidence_value * 100:.1f}%"
            )

            prediction_text = "Uncertain"


    # ========================================================
    # DISPLAY PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (20, 20),
        (700, 155),
        (0, 0, 0),
        -1
    )


    # --------------------------------------------------------
    # WORD
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"WORD: {prediction_text}",
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    cv2.putText(
        frame,
        confidence_text,
        (40, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if len(sequence) < SEQUENCE_LENGTH:

        status_text = (
            f"Collecting frames: "
            f"{len(sequence)}/{SEQUENCE_LENGTH}"
        )

    else:

        status_text = "Analyzing sign..."


    cv2.putText(
        frame,
        status_text,
        (40, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # --------------------------------------------------------
    # QUIT MESSAGE
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "SignLearn - Word Recognition",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

print(
    "\nReal-time recognition stopped."
)