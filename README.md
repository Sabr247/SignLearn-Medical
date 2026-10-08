<p align="center">
  <img src="assets/SignLearn logo.png" alt="SignLearn AI logo" width="360">
</p>

<h1 align="center">SignLearn 2.0 — AI-Powered Sign Language Learning System</h1>

<p align="center">
  A real-time, educational sign language learning platform combining<br>
  alphabet recognition, word spelling challenges, and word-level sign recognition.<br>
  Powered by Graph Convolutional Networks, BiLSTM, MediaPipe, and Streamlit.
</p>

<p align="center">
  <strong>Educational Project — Sign Language Recognition & Learning</strong>
</p>

---

## 📌 What It Does

---

## 📌 What It Does

**SignLearn 2.0** is an AI-powered sign language learning system designed to help learners practice and improve their American Sign Language (ASL) skills.

The system combines the original **alphabet-level GCN recognition system** with a new **word-level recognition module powered by a Bidirectional LSTM (BiLSTM)**.

SignLearn 2.0 provides five main learning areas:

1. ðŸ”¤ **Alphabet Learning** — Learn and practice ASL alphabet signs.
2. ðŸŽ® **Word Spelling Challenges** — Practice spelling words using individual signs.
3. ðŸ§  **Word-Level Recognition** — Perform a complete sign for an educational word and receive AI feedback.
4. ðŸ“Š **Skill Mastery & Certificate** — Track learning progress and mastery.
5. ðŸ“– **24-Letter Reference Library** — View alphabet sign references while learning.

The original alphabet recognition and word-spelling functionality has been preserved while the new word-level recognition system extends SignLearn into a broader sign language learning platform.

---

# âœ¨ Features

## ðŸ”¤ 1. Alphabet Learning

The original SignLearn GCN system recognizes alphabet signs using hand skeletal landmarks.

Learners can:

* Practice individual ASL letters.
* Receive real-time recognition from the webcam.
* Learn alphabet signs from A to Y.
* View visual references.
* Practice repeatedly through the live camera.

The original GCN alphabet recognition implementation is preserved in SignLearn 2.0.

---

## ðŸŽ® 2. Word Spelling Challenges

Learners can practice spelling words using individual alphabet signs.

The system:

1. Displays a target word.
2. Presents the letters that need to be signed.
3. Uses the alphabet recognition system to recognize each letter.
4. Tracks the learner's progress.
5. Provides feedback during the challenge.

This helps learners move from individual letters toward complete words.

---

## ðŸ§  3. Word-Level Sign Recognition

SignLearn 2.0 introduces a new word-level recognition system powered by a **Bidirectional Long Short-Term Memory (BiLSTM)** model.

Instead of spelling a word letter by letter, the learner performs the complete sign for a target word.

The system:

* Captures the learner through the webcam.
* Detects body and hand landmarks.
* Processes the movement sequence.
* Runs the sequence through the BiLSTM model.
* Predicts the signed word.
* Calculates prediction confidence.
* Compares the prediction with the target word.
* Displays **Correct** or **Try Again** feedback.

The current word-level model supports **20 educational words**.

### ðŸ“š Supported Words

```text
TEACHER
STUDENT
SCHOOL
CLASS
LEARN
STUDY
READ
WRITE
BOOK
PENCIL
PAPER
COMPUTER
HOMEWORK
TEST
ANSWER
UNDERSTAND
LISTEN
LIBRARY
HELP
FRIEND
```

---

## ðŸ“Š 4. Skill Mastery & Certificate

SignLearn provides a learning progress area where learners can review their performance and track their skill development.

The goal is to encourage continuous practice and progression from alphabet recognition to word-level signing.

---

## ðŸ“– 5. 24-Letter Reference Library

The reference library provides visual learning support for the ASL alphabet.

Learners can use it to:

* Review alphabet signs.
* Compare their hand position with references.
* Practice unfamiliar letters.
* Support the alphabet learning activities.

---

# ðŸ§  How It Works

SignLearn 2.0 contains two major recognition pipelines.

### Original Alphabet Recognition

```text
Webcam
   â†“
Hand Landmark Detection
   â†“
Skeletal Representation
   â†“
Graph Convolutional Network
   â†“
Alphabet Prediction
   â†“
Learning Feedback
```

### New Word-Level Recognition

```text
Webcam
   â†“
MediaPipe Holistic
   â†“
Body + Hand Landmarks
   â†“
50-Landmark Representation
   â†“
Normalization & Preprocessing
   â†“
64-Frame Sequence
   â†“
BiLSTM Model
   â†“
Word Prediction
   â†“
Confidence Score
   â†“
Correct / Try Again
```

---

# ðŸ“ Word-Level Landmark Representation

The word-level model does not process the entire video image directly.

Instead, SignLearn extracts important body and hand landmarks from each video frame.

Each frame contains **50 landmarks**:

* 1 neck landmark
* 1 nose landmark
* 6 upper-body landmarks
* 21 left-hand landmarks
* 21 right-hand landmarks

The face is not included in the current word-level model.

Each landmark contains three features:

```text
X coordinate
Y coordinate
Confidence
```

Therefore:

```text
50 landmarks Ã— 3 features = 150 features per frame
```

The BiLSTM receives a sequence of:

```text
64 frames Ã— 150 features
```

This converts the video into a structured movement sequence that can be processed by the neural network.

---

# ðŸ¤– Word-Level AI Model

The word-level recognition system uses a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network.

The BiLSTM was selected after comparing it with an ST-GCN model.

### Model Architecture

```text
Input
64 Frames Ã— 150 Features
        â†“
2-Layer Bidirectional LSTM
        â†“
128 Hidden Units
        â†“
Fully Connected Layer
        â†“
ReLU Activation
        â†“
Dropout
        â†“
20 Word Classes
```

The model learns temporal movement patterns from the sequence of body and hand landmarks.

---

# ðŸ“Š Dataset & Training

The word-level model was trained using the **ASL Citizen dataset**.

A vocabulary of 20 educational words was selected for the project.

### Dataset Processing

```text
Videos processed:      610
Processing failures:     0
Number of words:        20
```

### Dataset Split

```text
Training samples:      289
Validation samples:     78
Testing samples:       243
```

Every selected video was successfully processed into the required landmark representation.

---

# ðŸ“ˆ Model Performance

The final BiLSTM model achieved the following results:

| Metric                   |     Result |
| ------------------------ | ---------: |
| Best Validation Accuracy | **58.97%** |
| Final Training Accuracy  | **77.85%** |
| Test Accuracy            | **51.85%** |

The best validation performance was achieved around **epoch 11**.

These results show that the model was able to learn meaningful patterns from the available word-level sign language data, while also highlighting the need for more data and further improvement.

---

# âš–ï¸ BiLSTM vs ST-GCN

Two approaches were evaluated for word-level recognition.

| Model      | Test Accuracy |
| ---------- | ------------: |
| ST-GCN     |        44.86% |
| **BiLSTM** |    **51.85%** |

The BiLSTM was selected for the final word-level implementation because it achieved better test performance on the available dataset.

This comparison was important because the project did not simply assume that a more complex model would perform better.

The experiment showed that, with the available dataset, the simpler sequence-based BiLSTM produced the stronger result.

---

# ðŸ—ƒï¸ Dataset Processing Pipeline

The selected videos were converted into structured landmark sequences.

```text
ASL Citizen Videos
        â†“
MediaPipe Holistic
        â†“
Landmark Extraction
        â†“
Missing Landmark Handling
        â†“
Normalization
        â†“
Temporal Trimming
        â†“
64-Frame Resampling
        â†“
Training / Validation / Testing
        â†“
BiLSTM
```

This approach allows the model to focus on **movement and body structure** instead of raw image pixels.

---

# ðŸ› ï¸ Technologies Used

| Technology                | Purpose                       |
| ------------------------- | ----------------------------- |
| Python                    | Core programming language     |
| PyTorch                   | Deep learning framework       |
| MediaPipe                 | Landmark detection            |
| OpenCV                    | Computer vision processing    |
| Streamlit                 | Web application               |
| Streamlit WebRTC          | Real-time webcam streaming    |
| NumPy                     | Numerical processing          |
| BiLSTM                    | Word-level recognition        |
| GCN                       | Original alphabet recognition |
| ST-GCN                    | Experimental word-level model |
| ASL Citizen               | Word-level training dataset   |
| GitHub                    | Source code management        |
| Streamlit Community Cloud | Deployment                    |

---

# ðŸ“ Project Structure

```text
SignLearn-GCN/
â”‚
â”œâ”€â”€ assets/
â”‚   â””â”€â”€ SignLearn logo and application assets
â”‚
â”œâ”€â”€ reference_gifs/
â”‚   â””â”€â”€ Word-level sign reference GIFs
â”‚
â”œâ”€â”€ reference_videos/
â”‚   â””â”€â”€ Sign language reference videos
â”‚
â”œâ”€â”€ .vscode/
â”‚   â””â”€â”€ VS Code configuration
â”‚
â”œâ”€â”€ main.py
â”‚   â””â”€â”€ Main Streamlit application
â”‚
â”œâ”€â”€ signlearn_engine.py
â”‚   â””â”€â”€ Original SignLearn recognition engine
â”‚
â”œâ”€â”€ gcn_model.py
â”‚   â””â”€â”€ GCN model implementation
â”‚
â”œâ”€â”€ gcn_checkpoint.pt
â”‚   â””â”€â”€ Original GCN model checkpoint
â”‚
â”œâ”€â”€ gcn_weights.json
â”‚   â””â”€â”€ GCN model weights/configuration
â”‚
â”œâ”€â”€ best_word_bilstm.pt
â”‚   â””â”€â”€ Trained word-level BiLSTM model
â”‚
â”œâ”€â”€ word_model_config.json
â”‚   â””â”€â”€ Word-level model configuration
â”‚
â”œâ”€â”€ hand_landmarker.task
â”‚   â””â”€â”€ MediaPipe hand landmark model
â”‚
â”œâ”€â”€ holistic_landmarker.task
â”‚   â””â”€â”€ MediaPipe holistic landmark model
â”‚
â”œâ”€â”€ label_map.json
â”‚   â””â”€â”€ Alphabet label mapping
â”‚
â”œâ”€â”€ learning_drills.json
â”‚   â””â”€â”€ Learning and challenge configuration
â”‚
â”œâ”€â”€ main_alphabet_backup.py
â”‚   â””â”€â”€ Backup of the original alphabet implementation
â”‚
â”œâ”€â”€ requirements.txt
â”‚   â””â”€â”€ Python dependencies
â”‚
â”œâ”€â”€ packages.txt
â”‚   â””â”€â”€ Deployment packages
â”‚
â”œâ”€â”€ runtime.txt
â”‚   â””â”€â”€ Runtime configuration
â”‚
â”œâ”€â”€ .gitignore
â”‚   â””â”€â”€ Git ignored files
â”‚
â””â”€â”€ README.md
    â””â”€â”€ Project documentation
```

---

# ðŸš€ Getting Started

## Prerequisites

Before running SignLearn 2.0, make sure you have:

* Python 3.10 or newer
* Git
* A working webcam
* Internet connection for installation
* A modern web browser

---

# ðŸ“¥ Installation

Clone the repository:

```bash
git clone https://github.com/Yinusa/SignLearn-GCN.git
```

Move into the project directory:

```bash
cd SignLearn-GCN
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

---

# â–¶ï¸ Running the Application

Start the Streamlit application:

```powershell
streamlit run main.py
```

After Streamlit starts, open the local address shown in the terminal.

Usually:

```text
http://localhost:8501
```

Allow camera access when the browser requests permission.

---

# ðŸŽ“ Recommended Learning Workflow

SignLearn 2.0 is designed to support progressive learning.

```text
        ðŸ”¤
   Learn Alphabet
        â†“
        ðŸŽ®
 Word Spelling Practice
        â†“
        ðŸ§ 
 Word-Level Recognition
        â†“
        ðŸ“Š
 Track Skill Mastery
        â†“
        ðŸ†
    Certification
```

This allows learners to progress from individual signs to complete word recognition.

---

# ðŸ§ª Word-Level Recognition Example

A learner selects a target word such as:

```text
TARGET WORD

HELP
```

The application displays a reference GIF to demonstrate the expected sign.

The learner then performs the sign in front of the webcam.

The AI processes the movement and may display:

```text
TARGET: HELP

AI: HELP

âœ“ CORRECT!

Confidence: 86.4%
```

If the system detects another word:

```text
TARGET: HELP

AI: FRIEND

âœ• TRY AGAIN
```

This creates an interactive learning experience rather than simply displaying an AI prediction.

---

# âš ï¸ Known Limitations

SignLearn 2.0 is an educational and research prototype.

### 1. Limited Word-Level Dataset

The word-level model was trained using a relatively limited dataset compared with large commercial AI systems.

### 2. Recognition Accuracy

The final BiLSTM achieved **51.85% test accuracy**.

Therefore, the system will not recognize every sign correctly.

### 3. Isolated Word Recognition

The current word-level system recognizes isolated words rather than complete continuous sentences.

### 4. Signer Variation

Different people may perform the same sign differently.

More signer diversity is required to improve generalization.

### 5. Facial Information

The current word-level model does not use facial landmarks.

Facial expressions and other non-manual features can be important in sign language.

### 6. Educational Purpose

SignLearn is intended for learning, practice, and research.

It should not replace:

* Professional sign language interpreters
* Sign language teachers
* Healthcare professionals
* Human communication support

The system should not be used as the sole basis for important decisions.

---

# ðŸ”® Future Improvements

Future versions of SignLearn can include:

* Larger word-level datasets
* More educational vocabulary
* More signer diversity
* Signer-independent evaluation
* Continuous sentence recognition
* Facial expression features
* Improved temporal modelling
* Better live prediction smoothing
* More advanced learner analytics
* Mobile application support
* Improved accessibility features
* Testing with real sign language learners
* Collaboration with Deaf and hard-of-hearing communities

---

# ðŸ† Project Achievement

SignLearn 2.0 brings together alphabet-level and word-level sign language learning in one platform.

```text
                    SIGNLEARN 2.0
                         â”‚
             â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
             â”‚                       â”‚
             â†“                       â†“
      ðŸ”¤ ALPHABET LEVEL        ðŸ§  WORD LEVEL
             â”‚                       â”‚
          Original GCN             BiLSTM
             â”‚                       â”‚
         A â†’ Y Signs          20 Educational Words
             â”‚                       â”‚
             â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                         â†“
                Interactive Practice
                         â†“
                 AI-Based Feedback
                         â†“
                  Skill Development
```

The project demonstrates how computer vision, deep learning, and interactive educational design can be combined to create a practical sign language learning environment.

---

# ðŸ™ Acknowledgments

Special appreciation to [National Centre for Artificial Intelligence and Robotics (NCAIR)](https://ncair.nitda.gov.ng/), Abuja, Nigeria, the facilitators, the developers, researchers, and open-source communities behind the technologies used 
---

# 📌 Disclaimer

SignLearn 2.0 is an **educational and research prototype** developed for learning, experimentation, and demonstration purposes.

The system is not intended to replace professional interpreters, qualified educators, healthcare professionals, or human communication.

---

<p align="center">
  <strong>SignLearn 2.0 — Learn. Practice. Sign. Improve.</strong>
</p>


