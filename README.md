#  Sonic Signatures: Music Fingerprinting and Identification System

## Overview

Sonic Signatures is a Shazam-inspired music identification system developed as part of the EE200 Signals, Systems and Networks course project.

The system identifies a song from a short audio clip by generating a unique audio fingerprint based on spectral peaks extracted from a spectrogram. Instead of directly comparing audio waveforms, the algorithm converts audio into a compact set of frequency-time fingerprints and matches them against a pre-built song database.

The project demonstrates practical applications of:

- Fourier Transform (DFT/STFT)
- Spectrogram Analysis
- Time-Frequency Signal Processing
- Peak Detection
- Audio Fingerprinting
- Hash-Based Matching
- Streamlit Deployment

---

## Features

### Single Clip Identification

- Upload an audio clip (.mp3 or .wav)
- Generate spectrogram
- Extract constellation map
- Perform fingerprint matching
- Display predicted song
- Visualize offset histogram used for matching

### Batch Processing Mode

- Upload multiple audio clips
- Identify all clips automatically
- Generate `results.csv` in the required format

### Visualization

- Spectrogram
- Constellation Map
- Offset Histogram
- Audio Playback

---

## Methodology

### 1. Spectrogram Generation

The audio signal is converted into the time-frequency domain using the Short-Time Fourier Transform (STFT).

This produces a spectrogram representing how frequency content evolves over time.

### 2. Peak Detection

Local maxima are extracted from the spectrogram.

Only the strongest peaks are retained to create a sparse representation of the audio.

These peaks form the Constellation Map.

### 3. Fingerprint Generation

Nearby peaks are paired to generate compact hashes of the form:

```
(f1, f2, Δt)
```

where:

- `f1` = frequency of anchor peak
- `f2` = frequency of target peak
- `Δt` = time difference between peaks

These hashes act as the audio fingerprint.

### 4. Database Construction

Fingerprints from all reference songs are indexed and stored in a database.

Each hash stores:

- Song name
- Time location

### 5. Song Identification

Fingerprints are generated for the query clip and compared with the database.

Matches vote using:

```
(song, offset)
```

where offset represents temporal alignment between query and database fingerprints.

The song receiving the strongest consistent vote is selected as the prediction.

---

## Project Structure

```text
Sonic_Signatures/
│
├── app.py
├── database.pkl
├── requirements.txt
│
├── songs/
│   ├── song1.mp3
│   ├── song2.mp3
│   └── ...
│
├── reports/
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Sonic_Signatures.git
cd Sonic_Signatures
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your browser.


## Experimental Observations

### Single Peaks vs Paired Hashes

Single spectral peaks are not sufficiently distinctive and often produce false matches.

Paired hashes incorporating frequency relationships and temporal information provide significantly more reliable identification.

### Noise Robustness

The system remains accurate under moderate noise levels because dominant spectral peaks remain detectable.

As noise increases, matching performance gradually degrades.

### Pitch Shift Sensitivity

Small pitch shifts significantly affect performance because the fingerprinting scheme relies on absolute frequency locations.

Even though humans can recognize the song, shifted frequencies produce different hashes and reduce matching accuracy.

---

## Technologies Used

- Python
- NumPy
- SciPy
- Librosa
- Matplotlib
- Streamlit
- Pandas

---

## Course Information

**Course:** EE200 – Signals, Systems and Networks

**Project Title:** Sonic Signatures: Magical Mystery Tune

---

## Author

**Vrushabh D**  
Indian Institute of Technology Kanpur
