# Indian Sign Language Smart Communication Tool

A real-time **Indian Sign Language (ISL) recognition system** that converts hand gestures into **text and speech** using Computer Vision and Machine Learning. The project supports both **static alphabet signs (A–Z)** and **dynamic word-level signs (Hello, Thank You)**.

---

## 🎯 Project Features

* 🖐️ Real-time hand tracking using **MediaPipe**
* 🔤 Static sign recognition (A–Z)
* 🎥 Dynamic sign recognition (Hello, Thank You)
* 📝 Sentence builder with space & backspace logic
* 🔊 Online Text-to-Speech (Google TTS)
* 🧠 Trained ML models included using **Git LFS**

---

## 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy, Pandas
* Scikit-learn
* TensorFlow / Keras (for dynamic signs)
* gTTS (Online Text-to-Speech)

---

## 📂 Project Structure

```
sign_lang_smart_comm/
├── dataset/                 # Static alphabet CSV datasets (A–Z)
├── dynamic_dataset/         # Dynamic sign .npy datasets
│   ├── hello/
│   └── thank_you/
├── train_model.py           # Static model training
├── train_dynamic_model.py   # Dynamic model training
├── sentence_builder.py      # Real-time recognition + speech
├── isl_alphabet_model.pkl   # Trained static model
├── dynamic_sign_model.h5    # Trained dynamic model
├── .gitattributes           # Git LFS tracking
├── .gitignore
└── README.md
```

---

## ✋ Static Sign Recognition (A–Z)

### Dataset

* Each alphabet (A–Z) has its own folder
* Data stored as `data.csv`
* Each row contains **126 features** (21 landmarks × 3 × 2 hands)

### Training

* All CSV files are merged
* Labels assigned per alphabet
* Model trained using Scikit-learn
* Saved as:

  ```
  isl_alphabet_model.pkl
  ```

---

## 🎥 Dynamic Sign Recognition (Hello & Thank You)

### Dataset Creation

* Short videos recorded for each word
* MediaPipe extracts landmarks **per frame**
* Each frame → 126 features
* Frames combined into fixed-length sequences
* Saved as `.npy` files

Example shape:

```
(sequence_length, 126)
```

### Training

* `.npy` sequences loaded
* Labels assigned (hello / thank_you)
* Sequence-based model (LSTM)
* Saved as:

  ```
  dynamic_sign_model.h5
  dynamic_sign_model.pkl
  ```

---


https://github.com/user-attachments/assets/52a8b8ed-b645-4b15-99c0-eb7084307ea5


## 🔄 Real-Time Logic

* **Low motion** → Static model (letters)
* **High motion** → Dynamic model (words)
* Static letters form sentences
* Dynamic words are displayed/spoken directly

---

## 🔊 Text-to-Speech

* Uses **online Google Text-to-Speech (gTTS)**
* Press **S** to speak the sentence
* Temporary audio files auto-deleted

---

## ⌨️ Controls

| Key | Action                            |
| --- | --------------------------------- |
| q   | Quit application                  |
| s   | Speak sentence                    |
| b   | Backspace (delete last character) |

---

## 📦 Model Files & Git LFS

Large files are tracked using **Git LFS**:

```
*.pkl
*.h5
```

Make sure Git LFS is installed before cloning:

```
git lfs install
git lfs pull
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python sentence_builder.py
```

---

## 🧠 Outcome

This system enables **real-time ISL to speech translation**, making communication more accessible for the hearing-impaired and muted community.


---

## 👩‍💻 Author

**Hetvi Pandav**
BE – Artificial Intelligence & Machine Learning


https://github.com/user-attachments/assets/0291d2be-7cc2-4521-8bce-1b288516b94a


---

⭐ If you found this project useful, feel free to star the repository!



