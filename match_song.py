import matplotlib
matplotlib.use("Agg")

import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from scipy.ndimage import maximum_filter

# ==================================================
# PARAMETERS
# ==================================================

N_FFT = 2048
HOP_LENGTH = 512

PEAK_NEIGHBOURHOOD = 20

TOP_PEAKS = 250

TARGET_ZONE = 5

# ==================================================
# LOAD DATABASE
# ==================================================

with open("database.pkl", "rb") as f:
    database = pickle.load(f)

# ==================================================
# PEAK DETECTION
# ==================================================

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

    # remove strongest bass region
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
# ==================================================
# HASH CREATION
# ==================================================

def create_hashes(freq_idx, time_idx):

    hashes = []

    for i in range(len(time_idx)):

        for j in range(
            i + 1,
            min(i + TARGET_ZONE + 1,
                len(time_idx))
        ):

            f1 = int(freq_idx[i])
            f2 = int(freq_idx[j])

            dt = int(time_idx[j] - time_idx[i])

            if dt > 0:

                hashes.append(
                    (
                        (f1, f2, dt),
                        int(time_idx[i])
                    )
                )

    return hashes

# ==================================================
# SONG IDENTIFICATION
# ==================================================

def identify_song(query_file):

    y, sr = librosa.load(
        query_file,
        sr=22050,
        mono=True
    )

    S, freq_idx, time_idx = get_peaks(y)

    hashes = create_hashes(
        freq_idx,
        time_idx
    )

    votes = Counter()

    offsets = []

    for h, query_time in hashes:

        if h not in database:
            continue

        for song_name, db_time in database[h]:

            offset = db_time - query_time

            votes[(song_name, offset)] += 1

            offsets.append(offset)

    if len(votes) == 0:
        return None, S, freq_idx, time_idx, hashes, offsets

    best_match = votes.most_common(1)[0]

    predicted_song = best_match[0][0]

    return (
        predicted_song,
        S,
        freq_idx,
        time_idx,
        hashes,
        offsets
    )

# ==================================================
# MAIN
# ==================================================

query_file = input(
    "Enter query file path (example: songs/Hey Jude.mp3): "
)

(
    song_name,
    S,
    freq_idx,
    time_idx,
    hashes,
    offsets
) = identify_song(query_file)

print("\n====================================")
print("Predicted Song :", song_name)
print("Peaks Found    :", len(time_idx))
print("Hashes Created :", len(hashes))
print("====================================")

# ==================================================
# dB SPECTROGRAM
# ==================================================

S_db = librosa.amplitude_to_db(
    S,
    ref=np.max
)

# ==================================================
# FIGURE 1
# Spectrogram
# ==================================================

plt.figure(figsize=(12,5))

plt.imshow(
    S_db,
    origin="lower",
    aspect="auto",
    cmap="magma",
    vmin= -80,
    vmax=0
)

plt.colorbar(label="Magnitude (dB)")

plt.title("Spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency Bin")

plt.tight_layout()

plt.savefig(
    "1_spectrogram.png",
    dpi=300
)

plt.close()

# ==================================================
# FIGURE 2
# Spectrogram + Constellation
# ==================================================

plt.figure(figsize=(12,5))

plt.imshow(
    S_db,
    origin="lower",
    aspect="auto",
    cmap="magma",
    vmin= -80,
    vmax=0
)

plt.scatter(
    time_idx,
    freq_idx,
    s=60,
    marker='o',
    facecolors='none',
    edgecolors='cyan',
    linewidths=1.5
)

plt.colorbar(label="Magnitude (dB)")

plt.title("Spectrogram with Constellation Peaks")
plt.xlabel("Time Frame")
plt.ylabel("Frequency (Hz)")

plt.tight_layout()

plt.savefig(
    "2_constellation_overlay.png",
    dpi=300
)

plt.close()
# ==================================================
# FIGURE 3
# Constellation Map
# ==================================================

plt.figure(figsize=(12,5))

plt.scatter(
    time_idx,
    freq_idx,
    color="blue",
    s=25
)

plt.title("Constellation Map")
plt.xlabel("Time Frame")
plt.ylabel("Frequency (Hz)")

plt.tight_layout()

plt.savefig(
    "3_constellation_map.png",
    dpi=300
)

plt.close()
# ==================================================
# FIGURE 4
# Hash Pair Connections
# ==================================================

plt.figure(figsize=(12,5))

plt.scatter(
    time_idx,
    freq_idx,
    color="blue",
    s=15
)

num_lines = min(
    100,
    len(time_idx)
)

for i in range(num_lines):

    for j in range(
        i + 1,
        min(
            i + TARGET_ZONE + 1,
            num_lines
        )
    ):

        plt.plot(
            [time_idx[i], time_idx[j]],
            [freq_idx[i], freq_idx[j]],
            color="red",
            alpha=0.2
        )

plt.title("Hash Pair Connections")
plt.xlabel("Time Frame")
plt.ylabel("Frequency Bin")

plt.tight_layout()

plt.savefig(
    "4_hash_pairs.png",
    dpi=300
)

plt.close()

# ==================================================
# FIGURE 5
# Offset Histogram
# ==================================================

plt.figure(figsize=(10,5))

plt.hist(
    offsets,
    bins=100
)

plt.title("Offset Histogram")
plt.xlabel("Offset")
plt.ylabel("Votes")

plt.tight_layout()

plt.savefig(
    "5_offset_histogram.png",
    dpi=300
)

plt.close()

print("\nFigures saved:")
print("1_spectrogram.png")
print("2_constellation_overlay.png")
print("3_constellation_map.png")
print("4_hash_pairs.png")
print("5_offset_histogram.png")