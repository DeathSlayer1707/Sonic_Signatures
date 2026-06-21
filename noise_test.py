import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from scipy.ndimage import maximum_filter

# -----------------------------------
# PARAMETERS
# -----------------------------------

N_FFT = 2048
HOP_LENGTH = 512
TOP_PEAKS = 250

# -----------------------------------
# LOAD DATABASE
# -----------------------------------

with open("database.pkl", "rb") as f:
    database = pickle.load(f)

# -----------------------------------
# PEAK DETECTION
# -----------------------------------

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

    S[:15, :] = 0

    peak_mask = maximum_filter(
        S,
        size=(20,20)
    ) == S

    freq_idx, time_idx = np.where(peak_mask)

    strengths = S[freq_idx, time_idx]

    strongest = np.argsort(strengths)[-TOP_PEAKS:]

    freq_idx = freq_idx[strongest]
    time_idx = time_idx[strongest]

    return freq_idx, time_idx

# -----------------------------------
# HASH MATCHER
# -----------------------------------

def identify_song(y):

    freq_idx, time_idx = get_peaks(y)

    votes = Counter()

    for i in range(len(time_idx)):

        for j in range(i+1, min(i+6, len(time_idx))):

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

    if len(votes) == 0:
        return None, 0

    (song_name, offset), score = votes.most_common(1)[0]

    return song_name, score

# -----------------------------------
# LOAD SONG
# -----------------------------------

query_file = input("Enter query song: ")

y, sr = librosa.load(
    query_file,
    sr=22050,
    mono=True
)

# -----------------------------------
# NOISE LEVELS
# -----------------------------------

noise_levels = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

scores = []
predictions = []

print("\nNoise Experiment\n")

for noise in noise_levels:

    noisy_signal = y + noise * np.random.randn(len(y))

    pred, score = identify_song(noisy_signal)

    predictions.append(pred)
    scores.append(score)

    print(
        f"Noise={noise:.2f}  "
        f"Prediction={pred}  "
        f"Score={score}"
    )

# -----------------------------------
# PLOT
# -----------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    noise_levels,
    scores,
    marker='o'
)

plt.xlabel("Noise Level")
plt.ylabel("Matching Score")
plt.title("Noise Robustness Test")

plt.grid(True)

plt.savefig(
    "noise_robustness.png",
    dpi=300
)

plt.close()

print("\nSaved: noise_robustness.png")