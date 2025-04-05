import streamlit as st
import librosa
import audio_utils
import tempfile

st.title("🎵 Song Matcher Prototype")

# Load database
db = audio_utils.load_database()

# Upload audio
uploaded = st.file_uploader("Upload a short audio clip", type=["wav", "mp3", "flac"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded.read())
        tmp_path = tmp_file.name

    audio, sr = librosa.load(tmp_path, sr=None, mono=True)
    sample_hashes = audio_utils.fingerprint(audio, sr)

    match, score = audio_utils.match_sample(sample_hashes, db)
    if match:
        st.success(f"🎶 Matched with: {match} (Score: {score})")
    else:
        st.warning("No match found.")

# Optional: Button to rebuild database
if st.button("Rebuild song database"):
    with st.spinner("Building..."):
        db = audio_utils.build_database("database")
        st.success("Database rebuilt!")
