# 🎙️ Multimodal Approach to Emotion Analysis using Audio & Text  
*A Deep Learning Approach for Human Emotion Detection from Speech and Language*

---

## 🧠 Overview
This project presents a **robust multimodal emotion recognition system** that analyzes both **audio (speech)** and **text (transcribed content)** to classify human emotions.  
It combines **acoustic**, **linguistic**, and **contextual** features to achieve highly accurate emotion detection, leveraging **late fusion** of two independent neural models — one for audio, one for text.  

The system is implemented with **PyTorch**, **Transformers**, and **Streamlit**, and can perform real-time emotion prediction from microphone input or uploaded audio files.

---

## 🔍 Problem Statement
Traditional emotion recognition systems often rely on a **single modality** — either speech or text — leading to limited accuracy and contextual understanding.  
This project overcomes that limitation by combining both modalities, allowing the model to understand **what** is being said (text) and **how** it’s being said (audio tone and pitch).  

---

## 🎯 Objectives
- Build separate deep learning models for **speech-based** and **text-based** emotion recognition.  
- Integrate both models using **weighted late fusion** for improved classification performance.  
- Develop a **Streamlit-based interface** for real-time emotion prediction and visualization.  
- Achieve **high generalization accuracy** across multiple emotional categories.

---

## Concluded Architecture after experimenting with 6 different architectures such as:
- BERT
- CNN-BiLSTM- Attention Model
- LSTM
- 

## ⚙️ System Architecture
### 1️⃣ **Audio Model (Speech Emotion Recognition)**
- **Architecture:** CNN Based Attention Model
- **Features:** MFCCs, chroma, spectral contrast, zero-crossing rate, and pitch  
- **Frameworks:** Librosa, PyTorch  
- **Accuracy:** ~95%  
- **Output:** Probability distribution across emotion labels  

### 2️⃣ **Text Model (Text Emotion Recognition)**
- **Architecture:** Fine-tuned DistilBERT  
- **Dataset:** GoEmotions (Google)  
- **Frameworks:** Hugging Face Transformers, PyTorch  
- **Accuracy:** ~92%  
- **Output:** Probability distribution across emotion labels  

### 3️⃣ **Fusion Model (Weighted Late Fusion)**
- Combines the probability outputs of both models:  
  \[
  \text{final\_probs} = w_1 \times \text{audio\_probs} + w_2 \times \text{text\_probs}
  \]
- Weights determined by validation performance (e.g., 0.51 for audio, 0.49 for text)
- Final prediction is made based on maximum combined probability.

---

## 📊 Results Summary
| Model Type | Architecture | Accuracy | Key Feature |
|-------------|---------------|-----------|--------------|
| Audio Model | CNN | **95%** | Captures tone, intensity, rhythm |
| Text Model  | DistilBERT (Fine-tuned) | **92%** | Understands context and sentiment |
| Fusion Model | Weighted Late Fusion | **94–96%** | Combines best of both worlds |

---

## 🎬 Live Demo (Streamlit)
The Streamlit interface provides:  
✅ Real-time microphone input  
✅ Live waveform & spectrogram display  
✅ Transcription-based emotion inference  
✅ Visualization of confidence scores using bar graphs  


🧰 Technologies & Libraries

Python 3.10+
PyTorch – model building & training
Transformers (Hugging Face) – DistilBERT
Librosa – feature extraction (MFCCs, pitch, etc.)
Scikit-learn – data preprocessing, metrics
Matplotlib / Seaborn – visualization
Streamlit – interactive web app


🧠 Emotion Categories

The model predicts among the following emotions:

Label	Emotion
0	Angry
1	Disgust
2	Fear
3	Happy
4	Neutral
5	Sad
6	Surprise
