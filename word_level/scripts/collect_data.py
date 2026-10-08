import cv2
import mediapipe as mp
import numpy as np
import os
import time

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

TASK_FILE = os.path.join(
    BASE_DIR,
    "hand_landmarker.task"
)


# ============================================================
# WORDS
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


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=TASK_FILE
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(frame):

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]

    landmarks = []

    for landmark in hand:

        landmarks.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.array(
        landmarks,
        dtype=np.float32
    )


# ============================================================
# COLLECT ONE SEQUENCE
# ============================================================

def collect_sequence(sequence_length=30):

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "ERROR: Could not open the webcam."
        )

        return None

    sequence = []

    print("\nGet ready...")
    time.sleep(2)

    print("START SIGNING!")

    while len(sequence) < sequence_length:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Could not read webcam frame."
            )

            break

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        landmarks = extract_landmarks(
            frame
        )

        if landmarks is not None:

            sequence.append(
                landmarks
            )

            cv2.putText(
                frame,
                f"Frames: {len(sequence)}/{sequence_length}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "Show your hand",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Collecting Sequence - Press Q to Cancel",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            print(
                "Collection cancelled."
            )

            break

    cap.release()

    cv2.destroyAllWindows()

    if len(sequence) == sequence_length:

        sequence = np.array(
            sequence,
            dtype=np.float32
        )

        print(
            "Sequence collected successfully!"
        )

        print(
            "Sequence shape:",
            sequence.shape
        )

        return sequence

    print(
        "Sequence collection failed."
    )

    return None


# ============================================================
# DISPLAY CURRENT DATASET COUNTS
# ============================================================

print("\n" + "=" * 60)
print("CURRENT DATASET")
print("=" * 60)

DATA_DIR = os.path.join(
    BASE_DIR,
    "word_level",
    "data"
)

for word in WORDS:

    word_folder = os.path.join(
        DATA_DIR,
        word
    )

    if os.path.exists(word_folder):

        count = len([
            file
            for file in os.listdir(word_folder)
            if file.endswith(".npy")
        ])

    else:

        count = 0

    print(
        f"{word:<12} {count:>3} samples"
    )


# ============================================================
# SELECT WORD
# ============================================================

print("\n" + "=" * 60)
print("AVAILABLE WORDS")
print("=" * 60)

for number, word in enumerate(
    WORDS,
    start=1
):

    print(
        f"{number}. {word}"
    )


choice = input(
    "\nEnter the number of the word: "
).strip()


try:

    choice = int(choice)

except ValueError:

    print(
        "ERROR: Please enter a number."
    )

    landmarker.close()
    exit()


if choice < 1 or choice > len(WORDS):

    print(
        "ERROR: Invalid word selection."
    )

    landmarker.close()
    exit()


word = WORDS[
    choice - 1
]


# ============================================================
# NUMBER OF SAMPLES
# ============================================================

try:

    num_samples = int(
        input(
            f"How many additional {word} samples "
            "do you want to collect? "
        )
    )

except ValueError:

    print(
        "ERROR: Please enter a valid number."
    )

    landmarker.close()
    exit()


if num_samples <= 0:

    print(
        "ERROR: Number of samples must be greater than 0."
    )

    landmarker.close()
    exit()


# ============================================================
# CREATE WORD FOLDER
# ============================================================

word_folder = os.path.join(
    DATA_DIR,
    word
)

os.makedirs(
    word_folder,
    exist_ok=True
)


# ============================================================
# FIND EXISTING SAMPLES
# ============================================================

existing_samples = [
    file
    for file in os.listdir(word_folder)
    if file.endswith(".npy")
]


start_number = len(existing_samples) + 1


# ============================================================
# COLLECTION INFORMATION
# ============================================================

print("\n" + "=" * 60)

print(
    f"Word: {word}"
)

print(
    f"Existing samples: {len(existing_samples)}"
)

print(
    f"New samples: {num_samples}"
)

print(
    f"Samples after collection: "
    f"{len(existing_samples) + num_samples}"
)

print("=" * 60)


input(
    "\nPress ENTER when you are ready to begin..."
)


# ============================================================
# COLLECT SAMPLES
# ============================================================

successful = 0


for i in range(num_samples):

    sample_number = start_number + i

    print("\n" + "=" * 60)

    print(
        f"Sample {i + 1}/{num_samples}"
    )

    print(
        f"Word: {word}"
    )

    print("=" * 60)


    print(
        "\nGet ready..."
    )

    time.sleep(2)


    print("3")
    time.sleep(1)

    print("2")
    time.sleep(1)

    print("1")
    time.sleep(1)

    print(
        "START SIGNING!"
    )


    sequence = collect_sequence()


    if sequence is not None:

        file_path = os.path.join(
            word_folder,
            f"sample_{sample_number:03d}.npy"
        )

        np.save(
            file_path,
            sequence
        )

        successful += 1

        print(
            "\nSample saved successfully!"
        )

        print(
            "Word:",
            word
        )

        print(
            "Sample:",
            sample_number
        )

        print(
            "Shape:",
            sequence.shape
        )

        print(
            "Saved to:",
            file_path
        )

    else:

        print(
            "\nSample was not saved."
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

final_count = len(existing_samples) + successful


print("\n" + "=" * 60)

print(
    "COLLECTION COMPLETE"
)

print("=" * 60)

print(
    f"Word: {word}"
)

print(
    f"Successfully added: {successful}"
)

print(
    f"Total {word} samples: {final_count}"
)

print("=" * 60)


landmarker.close()