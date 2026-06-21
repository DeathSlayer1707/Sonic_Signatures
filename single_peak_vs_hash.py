import pickle
import librosa
import numpy as np

from collections import Counter
from scipy.ndimage import maximum_filter

# -----------------------------------------
# PARAMETERS
# -----------------------------------------

N_FFT = 2048
HOP_LENGTH = 512

TOP_PEAKS = 250

# -----------------------------------------
# LOAD DATABASE
# -----------------------------------------

with open("database.pkl", "rb") as f:
    database = pickle.load(f)

# -----------------------------------------
# PEAK DETECTION
# -----------------------------------------

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

# -----------------------------------------
# SINGLE PEAK MATCHING
# -----------------------------------------

def single_peak_match(query_file):

    y, sr = librosa.load(
        query_file,
        sr=22050,
        mono=True
    )

    freq_idx, time_idx = get_peaks(y)

    votes = Counter()

    for f in freq_idx:

        for h in database.keys():

            if h[0] == f:
                for song_name, _ in database[h]:
                    votes[song_name] += 1

    if len(votes) == 0:
        return None, 0

    best_song, score = votes.most_common(1)[0]

    return best_song, score

# -----------------------------------------
# HASH MATCHING
# -----------------------------------------

def hash_match(query_file):

    y, sr = librosa.load(
        query_file,
        sr=22050,
        mono=True
    )

    freq_idx, time_idx = get_peaks(y)

    votes = Counter()

    for i in range(len(time_idx)):

        for j in range(
            i+1,
            min(i+6, len(time_idx))
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

    if len(votes) == 0:
        return None, 0

    (song_name, offset), score = votes.most_common(1)[0]

    return song_name, offset, score

# -----------------------------------------
# MAIN
# -----------------------------------------

query_file = input(
    "Enter query file path: "
)

song1, score1 = single_peak_match(
    query_file
)

song2, offset2, score2 = hash_match(
    query_file
)

print("\n========== RESULTS ==========")

print("\nSingle Peak Matching")
print("Prediction :", song1)
print("Score      :", score1)

print("\nHash Pair Matching")
print("Prediction :", song2)
print("Offset     :", offset2)
print("Score      :", score2)

print("\n=============================")