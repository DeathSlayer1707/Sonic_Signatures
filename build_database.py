import os
import pickle
import librosa
import numpy as np

from tqdm import tqdm
from scipy.ndimage import maximum_filter

# -----------------------------
# PARAMETERS
# -----------------------------

SONG_FOLDER = "songs"

N_FFT = 2048
HOP_LENGTH = 512

PEAK_NEIGHBOURHOOD = 20
TOP_PEAKS = 500

TARGET_ZONE = 5

# -----------------------------
# FIND PEAKS
# -----------------------------

def get_peaks(y, sr):

    S = np.abs(
        librosa.stft(
            y,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    peak_mask = maximum_filter(
        S,
        size=PEAK_NEIGHBOURHOOD
    ) == S

    freq_idx, time_idx = np.where(peak_mask)

    strengths = S[freq_idx, time_idx]

    strongest = np.argsort(strengths)[-TOP_PEAKS:]

    freq_idx = freq_idx[strongest]
    time_idx = time_idx[strongest]

    return freq_idx, time_idx

# -----------------------------
# CREATE HASHES
# -----------------------------

def create_hashes(freq_idx, time_idx):

    hashes = []

    num_peaks = len(time_idx)

    for i in range(num_peaks):

        for j in range(i + 1,
                       min(i + TARGET_ZONE + 1,
                           num_peaks)):

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

# -----------------------------
# BUILD DATABASE
# -----------------------------

database = {}

songs = [
    f for f in os.listdir(SONG_FOLDER)
    if f.endswith(".mp3")
]

print("Songs found:", len(songs))

for song in tqdm(songs):

    song_path = os.path.join(
        SONG_FOLDER,
        song
    )

    y, sr = librosa.load(
        song_path,
        sr=22050,
        mono=True
    )

    freq_idx, time_idx = get_peaks(y, sr)

    hashes = create_hashes(
        freq_idx,
        time_idx
    )

    song_name = os.path.splitext(song)[0]

    for h, anchor_time in hashes:

        if h not in database:
            database[h] = []

        database[h].append(
            (
                song_name,
                anchor_time
            )
        )

print("Total unique hashes:",
      len(database))

with open(
    "database.pkl",
    "wb"
) as f:

    pickle.dump(
        database,
        f
    )

print("database.pkl created")