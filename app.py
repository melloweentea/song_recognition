import streamlit as st
import tempfile
import os
import librosa
import numpy as np
import pickle
import hashlib
from scipy.ndimage import maximum_filter, binary_erosion, generate_binary_structure

FAN_VALUE = 15
MAX_TIME_DELTA = 200
DB_FILE = 'fingerprints.pkl'

def fingerprint(audio, sr):
    S = np.abs(librosa.stft(audio))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    struct = generate_binary_structure(2, 1)
    local_max = maximum_filter(S_db, footprint=struct) == S_db
    background = (S_db == 0)
    detected_peaks = local_max ^ binary_erosion(background)

    peaks = np.argwhere(detected_peaks)
    hashes = []

    for i in range(len(peaks)):
        for j in range(1, FAN_VALUE):
            if i + j < len(peaks):
                freq1, time1 = peaks[i]
                freq2, time2 = peaks[i + j]
                delta_t = time2 - time1
                if 0 < delta_t <= MAX_TIME_DELTA:
                    hash_input = f"{freq1}|{freq2}|{delta_t}"
                    hash_val = hashlib.sha1(hash_input.encode()).hexdigest()[:20]
                    hashes.append((hash_val, time1))
    return hashes

def load_database(db_file=DB_FILE):
    if os.path.exists(db_file):
        with open(db_file, 'rb') as f:
            return pickle.load(f)
    return {}

def match_sample(sample_hashes, database):
    match_counts = {}
    for song, hashes in database.items():
        hash_set = set(h[0] for h in hashes)
        match_count = sum(1 for h, _ in sample_hashes if h in hash_set)
        match_counts[song] = match_count

    if match_counts:
        best_match = max(match_counts, key=match_counts.get)
        return best_match, match_counts[best_match]
    return None, 0

st.title("🎵 Song Matcher Prototype")
st.write("Hello! Welcome to this song matcher prototype. This webapp has been trained on Tatsuro Yamashita's 1982 album \"For You\". Due to memory limits, only the first 20s of each song are processed in the database. Please record a short clip up to 20s of the beginning of the song for this to work properly.")
st.markdown(''' ### The tracklist for this album is as follows:
    1. Sparkle
    2. Music Book
    3. Morning Glory
    4. Futari
    5. Loveland, Island
    6. Love Talkin' (Honey It's You)
    7. Hey Reporter!
    8. Your Eyes 
''')
st.write("Note: Tatsuro Yamashita's songs are not available on conventional music streaming platforms, but you can find them on YouTube. Here is the link to \"[Sparkle](https://youtu.be/pqobRu9aR3M?si=V4jT-XLEO5_cIP_5)\", my favorite song on the album for you to test out.")

db = load_database()

audio_clip = st.audio_input("Record 10-20s of the song you want to match")

if audio_clip:
    audio, sr = librosa.load(audio_clip, sr=None, mono=True)
    sample_hashes = fingerprint(audio, sr)

    match, score = match_sample(sample_hashes, db)
    if match:
        st.success(f"🎶 Matched with: {match} (Score: {score})")
    else:
        st.warning("No match found.")

