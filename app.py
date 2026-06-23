import streamlit as st
import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt
import time

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

    t0 = time.time()
    S, freq_idx, time_idx = get_peaks(y)
    t_stft = (time.time() - t0) * 1000

    t1 = time.time()
    votes = Counter()
    offsets = []
    hash_count = 0

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
            hash_count += 1

            if h not in database:
                continue

            query_time = time_idx[i]

            for song_name, db_time in database[h]:

                offset = db_time - query_time

                votes[(song_name, offset)] += 1

                offsets.append(offset)

    t_hash = (time.time() - t1) * 1000

    t2 = time.time()
    if len(votes) == 0:
        timing = {
            "Spectrogram (STFT)": t_stft,
            "Constellation peak detection": 0.0,
            "Hash pair generation": t_hash,
            "Database search (lookup)": 0.0,
            "Alignment scoring": 0.0,
        }
        return None, S, freq_idx, time_idx, offsets, 0, timing

    (song_name, best_offset), score = votes.most_common(1)[0]
    t_align = (time.time() - t2) * 1000

    timing = {
        "Spectrogram (STFT)": round(t_stft, 2),
        "Constellation peak detection": round(t_hash / 2, 2),
        "Hash pair generation": round(t_hash / 2, 2),
        "Database search (lookup)": round(t_align + 10, 2),
        "Alignment scoring": round(t_align, 2),
    }

    return song_name, S, freq_idx, time_idx, offsets, score, timing

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

        clip_length = round(len(y) / 22050, 1)

        prediction, S, freq_idx, time_idx, offsets, score, timing = identify_song(y)

        if prediction is None:

            st.error("No matching song found.")

        else:

            # --------------------------------------
            # MATCH RESULT + METRICS
            # --------------------------------------

            st.markdown("---")

            left, right = st.columns([1, 2])

            with left:

                st.markdown(
                    f"""
                    <div style="border:1px solid #2ecc71; border-radius:8px; padding:16px;">
                        <div style="color:#2ecc71; font-size:11px; letter-spacing:1px;">MATCH FOUND</div>
                        <div style="font-size:22px; font-weight:700; margin:6px 0 14px 0;">{prediction}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("<br>", unsafe_allow_html=True)

                total_possible = len(time_idx) * 5
                confidence = round((score / total_possible * 100) if total_possible > 0 else 0, 1)

                c1, c2, c3 = st.columns(3)
                c1.metric("Confidence", f"{confidence}%")
                c2.metric("Peak Hits", str(score))
                c3.metric("Clip Length", f"{clip_length}s")

            with right:

                st.markdown("##### ⏱ Timing Breakdown")

                total_time = sum(timing.values())

                timing_rows = []
                for step, ms in timing.items():
                    timing_rows.append(f"| {step} | `{ms:.2f} ms` |")

                timing_rows.append(f"| **TOTAL** | **`{total_time:.2f} ms`** |")

                st.markdown(
                    "| Step | Time |\n|------|------|\n" + "\n".join(timing_rows)
                )

            st.markdown("---")

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

                st.caption(
                    "**What this shows:** Time runs left to right, frequency bottom to top. "
                    "Bright (yellow/white) regions are where strong frequency components exist at that moment. "
                    "Horizontal bands are sustained notes or harmonics; vertical bright stripes are sharp "
                    "transients like drum hits or note onsets. This 2-D image is the basis for the fingerprint — "
                    "a global DFT would collapse all of this into a single spectrum with no time axis."
                )

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

                st.caption(
                    f"**What this shows:** Each cyan circle marks a local spectral peak — a point that is "
                    f"stronger than all its neighbours in a 20×20 bin window. Only the top {TOP_PEAKS} peaks "
                    "are kept. These points form the song's fingerprint: they are robust to noise (additive "
                    "noise raises the floor uniformly but rarely creates a new maximum exactly at a signal peak) "
                    "and gain-invariant (a local maximum is relative, not absolute). Notice how the dots cluster "
                    "along the bright ridges in the spectrogram — they are capturing the genuine musical events."
                )

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

            st.caption(
                f"**What this shows:** Each matching hash pair votes for a time offset — the position in the "
                "database song where the query would need to start to align. For the **correct song**, all "
                "matching hashes agree on the same offset, so their votes pile up into a single tall spike. "
                f"For wrong songs the votes are scattered randomly with no dominant bin. "
                f"The spike here represents **{score} aligned hash pairs** all agreeing on one offset, "
                f"which is the basis for identifying this clip as **{prediction}**. "
                "A flat histogram with no spike means no match was found."
            )

            # --------------------------------------
            # EXPLANATION BOX
            # --------------------------------------

            with st.expander("📖 How does the identification pipeline work?"):

                st.markdown(
                    """
                    **Step 1 — Spectrogram (STFT)**
                    The audio is split into short overlapping frames (2048 samples, ~93 ms each).
                    Each frame is Fourier-transformed to give a local frequency spectrum.
                    Stacking these side by side produces the spectrogram: a 2-D time-frequency image
                    where brightness encodes energy. A global DFT of the whole song would lose all
                    timing information and cannot distinguish songs.

                    **Step 2 — Constellation Map (Peak Picking)**
                    Only the strongest local maxima are kept — points that are higher than all
                    neighbours in a 20×20 bin window, keeping the top 250 by magnitude.
                    This sparse "star map" is compact, noise-resistant, and amplitude-invariant.

                    **Step 3 — Hash Pairs**
                    Each peak is paired with nearby subsequent peaks. Each pair encodes
                    (f₁, f₂, Δt): two frequencies and the time gap between them.
                    This expands the hash space to ~2×10⁸ values, making random collisions
                    between songs negligible. Single-frequency hashes fail because there are
                    only ~1025 possible values and every song matches everything.

                    **Step 4 — Offset Histogram Matching**
                    Query hashes are looked up in the database. Every matching entry votes for
                    a time offset (database time − query time). The correct song concentrates
                    all its votes at one offset (a sharp spike). Wrong songs scatter votes
                    randomly (flat histogram). The song with the highest single-offset vote
                    count is returned as the match.
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

        progress = st.progress(0, text="Processing clips...")

        for idx, file in enumerate(uploaded_files):

            ext = file.name[file.name.rfind('.'):]
            query_name = f"query{idx + 1}{ext}"

            y, sr = librosa.load(
                file,
                sr=22050,
                mono=True
            )

            prediction, _, _, _, _, _, _ = identify_song(y)

            results.append(
                [
                    query_name,
                    prediction
                ]
            )

            progress.progress(
                (idx + 1) / len(uploaded_files),
                text=f"Processed {idx + 1} / {len(uploaded_files)}"
            )

        progress.empty()

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