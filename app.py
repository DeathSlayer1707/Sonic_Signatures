import streamlit as st
import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from scipy.ndimage import maximum_filter

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Sonic Signatures",
    page_icon="🎵",
    layout="wide"
)

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

N_FFT = 2048
HOP_LENGTH = 512
TOP_PEAKS = 250

# --------------------------------------------------
# LOAD DATABASE
# --------------------------------------------------

@st.cache_resource
def load_database():

    with open("database.pkl", "rb") as f:
        return pickle.load(f)

database = load_database()

# --------------------------------------------------
# PEAK DETECTION
# --------------------------------------------------

def get_peaks(y):

    S = np.abs(
        librosa.stft(
            y,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    freqs = librosa.fft_frequencies(
        sr=22050,
        n_fft=N_FFT
    )

    valid_rows = freqs <= 3500

    S = S[valid_rows, :]

    # remove very low frequencies
    S[:15, :] = 0

    peak_mask = maximum_filter(
        S,
        size=(20, 20)
    ) == S

    freq_idx, time_idx = np.where(peak_mask)

    strengths = S[freq_idx, time_idx]

    strongest = np.argsort(strengths)[-TOP_PEAKS:]

    freq_idx = freq_idx[strongest]
    time_idx = time_idx[strongest]

    return S, freq_idx, time_idx

# --------------------------------------------------
# SONG IDENTIFICATION
# --------------------------------------------------

def identify_song(y):

    S, freq_idx, time_idx = get_peaks(y)

    votes = Counter()

    offsets = []

    for i in range(len(time_idx)):

        for j in range(
            i + 1,
            min(i + 6, len(time_idx))
        ):

            f1 = int(freq_idx[i])
            f2 = int(freq_idx[j])

            dt = int(time_idx[j] - time_idx[i])

            if dt <= 0:
                continue

            h = (f1, f2, dt)

            if h not in database:
                continue

            query_time = time_idx[i]

            for song_name, db_time in database[h]:

                offset = db_time - query_time

                votes[(song_name, offset)] += 1

                offsets.append(offset)

    if len(votes) == 0:
        return None, S, freq_idx, time_idx, offsets

    (song_name, best_offset), score = votes.most_common(1)[0]

    return song_name, S, freq_idx, time_idx, offsets

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🎵 Sonic Signatures")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Select Mode",
    [
        "Single Clip",
        "Batch Mode"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("EE200 Project")
st.sidebar.write("Audio Fingerprinting")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎵 Sonic Signatures")
st.caption(
    "Shazam-style Music Identification using Spectrogram Fingerprinting"
)

# --------------------------------------------------
# SINGLE CLIP MODE
# --------------------------------------------------

if mode == "Single Clip":

    uploaded_file = st.file_uploader(
        "Upload Query Song",
        type=["mp3", "wav"]
    )

    if uploaded_file is not None:

        st.audio(uploaded_file)

        y, sr = librosa.load(
            uploaded_file,
            sr=22050,
            mono=True
        )

        prediction, S, freq_idx, time_idx, offsets = identify_song(y)

        if prediction is None:

            st.error("No matching song found.")

        else:

            st.success(
                f"🎵 Predicted Song: {prediction}"
            )

            S_db = librosa.amplitude_to_db(
                S,
                ref=np.max
            )

            # --------------------------------------
            # SIDE BY SIDE PLOTS
            # --------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Spectrogram")

                fig1, ax1 = plt.subplots(
                    figsize=(8, 4)
                )

                ax1.imshow(
                    S_db,
                    origin="lower",
                    aspect="auto",
                    cmap="magma",
                    vmin=-80,
                    vmax=0
                )

                ax1.set_xlabel("Time Frame")
                ax1.set_ylabel("Frequency Bin")

                st.pyplot(fig1)

            with col2:

                st.subheader("Constellation Map")

                fig2, ax2 = plt.subplots(
                    figsize=(8, 4)
                )

                ax2.imshow(
                    S_db,
                    origin="lower",
                    aspect="auto",
                    cmap="magma",
                    vmin=-80,
                    vmax=0
                )

                ax2.scatter(
                    time_idx,
                    freq_idx,
                    s=60,
                    facecolors="none",
                    edgecolors="cyan",
                    linewidths=1.5
                )

                ax2.set_xlabel("Time Frame")
                ax2.set_ylabel("Frequency Bin")

                st.pyplot(fig2)

            # --------------------------------------
            # HISTOGRAM
            # --------------------------------------

            st.subheader("Offset Histogram")

            fig3, ax3 = plt.subplots(
                figsize=(10, 4)
            )

            ax3.hist(
                offsets,
                bins=50
            )

            ax3.set_xlabel("Offset")
            ax3.set_ylabel("Count")
            ax3.set_title("Hash Match Offset Distribution")

            st.pyplot(fig3)

            # --------------------------------------
            # EXPLANATION BOXES
            # --------------------------------------

            with st.expander("What is a Spectrogram?"):

                st.write(
                    """
                    A spectrogram shows how frequency content
                    changes over time. Bright regions indicate
                    stronger frequency components.
                    """
                )

            with st.expander("What is a Constellation Map?"):

                st.write(
                    """
                    The strongest local peaks extracted from the
                    spectrogram. These peaks are used to generate
                    fingerprints for song matching.
                    """
                )

# --------------------------------------------------
# BATCH MODE
# --------------------------------------------------

elif mode == "Batch Mode":

    st.header("Batch Prediction")

    uploaded_files = st.file_uploader(
        "Upload Query Clips",
        type=["mp3", "wav"],
        accept_multiple_files=True
    )

    if uploaded_files:

        results = []

        for file in uploaded_files:

            y, sr = librosa.load(
                file,
                sr=22050,
                mono=True
            )

            prediction, _, _, _, _ = identify_song(y)

            results.append(
                [
                    file.name,
                    prediction
                ]
            )

        import pandas as pd

        df = pd.DataFrame(
            results,
            columns=[
                "filename",
                "prediction"
            ]
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download results.csv",
            data=csv,
            file_name="results.csv",
            mime="text/csv"
        )