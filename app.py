import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import parselmouth
import speech_recognition as sr
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os
import torch
from tensorflow.keras.models import model_from_json
import pickle
from pydub import AudioSegment

# Import the st_audiorecorder component
try:
    from audio_recorder_streamlit import audio_recorder
except ImportError:
    st.error("Please install st_audiorecorder: pip install streamlit-audiorecorder")

# -----------------------------
# Set file paths (update to your model directory)
# -----------------------------
AUDIO_MODEL_JSON = r"C:/Users/sajid/Desktop/SAJID Personal/VCET/Final Year/FINAL YEAR REPORT/revision/96percentAudioModel/96percentCNN_model.json"
AUDIO_MODEL_WEIGHTS = r"C:/Users/sajid/Desktop/SAJID Personal/VCET/Final Year/FINAL YEAR REPORT/revision/96percentAudioModel/best_model1_weights.keras"
SCALER_PATH = r"C:/Users/sajid/Desktop/SAJID Personal/VCET/Final Year/FINAL YEAR REPORT/revision/96percentAudioModel/scaler2.pickle"
ENCODER2_PATH = r"C:/Users/sajid/Desktop/SAJID Personal/VCET/Final Year/FINAL YEAR REPORT/revision/96percentAudioModel/encoder2.pickle"
TEXT_MODEL_DIR = r"C:/Users/sajid/Desktop/SAJID Personal/VCET/Final Year/FINAL YEAR REPORT/revision/GUI/87percentDistilBert3kModel"

# -----------------------------
# Audio feature extraction functions
# -----------------------------
def zcr(data, frame_length, hop_length):
    z = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(z)
    
def rmse(data, frame_length=2048, hop_length=512):
    r = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(r)
    
def mfcc(data, sr, frame_length=2048, hop_length=512, flatten:bool=True):
    m = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(m.T) if not flatten else np.ravel(m.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                       ))
    return result

def get_predict_feat(path, expected_dim=2376):
    data, sr = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(data)
    result = np.array(res)
    current_dim = result.shape[0]
    if current_dim < expected_dim:
        pad_width = expected_dim - current_dim
        result = np.pad(result, (0, pad_width), mode='constant')
    elif current_dim > expected_dim:
        result = result[:expected_dim]
    result = np.reshape(result, newshape=(1, expected_dim))
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2)
    return final_result

# -----------------------------
# Google Speech-to-Text Function
# -----------------------------
def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio_data)
        return transcript
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

# -----------------------------
# Load Models, Scaler, and Encoder2
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_audio_model():
    with open(AUDIO_MODEL_JSON, "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(AUDIO_MODEL_WEIGHTS)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    st.write("Audio model loaded.")
    return model

@st.cache_resource(show_spinner=False)
def load_scaler():
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    st.write("Scaler loaded.")
    return scaler

@st.cache_resource(show_spinner=False)
def load_encoder():
    with open(ENCODER2_PATH, 'rb') as f:
        encoder = pickle.load(f)
    st.write("Encoder loaded.")
    return encoder

@st.cache_resource(show_spinner=False)
def load_text_model_and_tokenizer():
    model = DistilBertForSequenceClassification.from_pretrained(TEXT_MODEL_DIR)
    tokenizer = DistilBertTokenizerFast.from_pretrained(TEXT_MODEL_DIR)
    st.write("Text model and tokenizer loaded.")
    return model, tokenizer

audio_model = load_audio_model()
scaler2 = load_scaler()
encoder2 = load_encoder()
text_model, tokenizer = load_text_model_and_tokenizer()

# -----------------------------
# Helper function to decode predictions using encoder2
# -----------------------------
def decode_prediction(probs):
    return encoder2.inverse_transform(probs.reshape(1, -1))[0][0]

# -----------------------------
# Prediction Functions
# -----------------------------
def predict_audio(audio_path):
    feat = get_predict_feat(audio_path)  # shape: (1, expected_dim, 1)
    audio_probs = audio_model.predict(feat)[0]
    audio_decoded = decode_prediction(audio_probs)
    return audio_probs, audio_decoded

def predict_text(transcript):
    inputs = tokenizer(transcript, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    outputs = text_model(**inputs)
    text_probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]
    text_decoded = decode_prediction(text_probs)
    return text_probs, text_decoded

def fusion_decision(audio_probs, text_probs, threshold=0.90, weight=0.8):
    if audio_probs is None or text_probs is None:
        return None
    if np.max(audio_probs) >= threshold:
        final_index = np.argmax(audio_probs)
    else:
        final_probs = weight * audio_probs + (1 - weight) * text_probs
        final_index = np.argmax(final_probs)
    one_hot = np.zeros_like(audio_probs)
    one_hot[final_index] = 1
    return encoder2.inverse_transform(one_hot.reshape(1, -1))[0][0]

# -----------------------------
# Streamlit User Interface
# -----------------------------
st.title("Multimodal Emotion Recognition Demo")
st.markdown(
    "This app uses a CNN-based audio model (96% accuracy) as the primary predictor and a DistilBERT-based text model (93% accuracy) as support. "
    "The audio is converted to text via Google Speech-to-Text, and then the outputs are fused using an encoder to decode the final emotion."
)

# Choose between uploading or recording audio
audio_source = st.radio("Choose audio input method:", ["Upload Audio", "Record Audio"])

temp_audio_path = "temp_audio.wav"
if audio_source == "Upload Audio":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension != "wav":
            # Convert non-wav files to wav using pydub
            audio_segment = AudioSegment.from_file(uploaded_file, format=file_extension)
            audio_segment.export(temp_audio_path, format="wav")
        else:
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.audio(uploaded_file, format="audio/wav")
elif audio_source == "Record Audio":
    st.info("Click the button below to record your audio.")
    audio_bytes = audio_recorder("Click to record")
    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)

# Only run prediction if a temporary audio file exists and user clicks the predict button
if os.path.exists(temp_audio_path):
    if st.button("Predict"):
        st.info("Converting audio to text...")
        transcript = audio_to_text(temp_audio_path)
        if transcript:
            st.write("**Transcript:**", transcript)
        else:
            st.write("Transcript could not be obtained. You can manually enter one below.")
    
        transcript = st.text_area("Edit transcript (if needed):", transcript)
    
        st.info("Predicting emotion using audio model...")
        audio_probs, audio_pred = predict_audio(temp_audio_path)
        if audio_probs is not None:
            st.write("**Audio Model Prediction:**", audio_pred)
            st.write("Confidence:", np.max(audio_probs))
        else:
            st.error("Audio model prediction failed.")
    
        if transcript:
            st.info("Predicting emotion using text model...")
            text_probs, text_pred = predict_text(transcript)
            st.write("**Text Model Prediction:**", text_pred)
            st.write("Confidence:", np.max(text_probs))
        else:
            text_probs, text_pred = None, None
    
        st.info("Fusing predictions...")
        final_pred = fusion_decision(audio_probs, text_probs, threshold=0.90, weight=0.8)
        if final_pred is not None:
            st.write("### **Final Prediction:**", final_pred)
        else:
            st.error("Fusion decision could not be made.")
    
    # Remove temporary file after processing
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
