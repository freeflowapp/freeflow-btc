"""BTC chord recognition (major/minor model) as BTC-ISMIR19's test.py runs it.

Features and inference follow upstream test.py and utils/mir_eval_modules.py:audio_file_to_features exactly;
only the audio decoding differs (ffmpeg to a temporary wav, then librosa.load at 22050 Hz mono).
torch and librosa are imported lazily so the tests need numpy only.
"""
import os
import subprocess
import tempfile

import numpy as np

from .segments import IDX_TO_CHORD, frames_to_segments

SOURCE = "jayg996/BTC-ISMIR19@2682317"
MODEL = "btc-majmin"
WEIGHTS_LICENSE = "MIT (Copyright (c) 2019 Jonggwon Park; the repository's licence, no separate weights licence stated)"
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_model.pt")

# run_config.yaml, major/minor model.
SONG_HZ = 22050
INST_LEN = 10.0
N_BINS = 144
BINS_PER_OCTAVE = 24
HOP_LENGTH = 2048
MODEL_CONFIG = {
    "feature_size": 144,
    "timestep": 108,
    "num_chords": 25,
    "input_dropout": 0.2,
    "layer_dropout": 0.2,
    "attention_dropout": 0.2,
    "relu_dropout": 0.2,
    "num_layers": 8,
    "num_heads": 4,
    "hidden_size": 128,
    "total_key_depth": 128,
    "total_value_depth": 128,
    "filter_size": 128,
    "loss": "ce",
    "probs_out": False,
}
TIMESTEP = MODEL_CONFIG["timestep"]
FRAME_SECONDS = INST_LEN / TIMESTEP  # 10/108 s per frame


def load(path):
    """Decode any file ffmpeg reads to 22050 Hz mono float32."""
    import librosa

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-vn", wav], check=True)
        wave, _ = librosa.load(wav, sr=SONG_HZ, mono=True)
    return wave


class Recognizer:
    def __init__(self, device=None):
        import torch

        from .model import BTC_model

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(WEIGHTS, map_location=self.device, weights_only=False)
        self.mean, self.std = ck["mean"], ck["std"]
        self.model = BTC_model(config=MODEL_CONFIG).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    @staticmethod
    def features(wave):
        """log(|CQT| + 1e-6) in 10 s pieces, as audio_file_to_features does; frames x bins."""
        import librosa

        step = int(SONG_HZ * INST_LEN)
        cqt = lambda x: librosa.cqt(x, sr=SONG_HZ, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP_LENGTH)
        parts, cur = [], 0
        while len(wave) > cur + step:
            parts.append(cqt(wave[cur:cur + step]))
            cur += step
        parts.append(cqt(wave[cur:]))
        feature = np.concatenate(parts, axis=1)
        return np.log(np.abs(feature) + 1e-6).T

    def frame_labels(self, wave):
        torch = self.torch
        feature = (self.features(wave) - self.mean) / self.std
        frames = feature.shape[0]
        num_pad = TIMESTEP - (frames % TIMESTEP)
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        labels = []
        with torch.no_grad():
            x = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)
            for t in range(feature.shape[0] // TIMESTEP):
                out, _ = self.model.self_attn_layers(x[:, TIMESTEP * t:TIMESTEP * (t + 1), :])
                prediction, _ = self.model.output_layer(out)
                labels += prediction.squeeze().reshape(-1).tolist()
        return labels[:frames]

    def recognize(self, path):
        wave = load(path)
        return {
            "model": MODEL,
            "source": SOURCE,
            "weights_license": WEIGHTS_LICENSE,
            "duration": round(len(wave) / SONG_HZ, 3),
            "frame_seconds": FRAME_SECONDS,
            "segments": frames_to_segments(self.frame_labels(wave), FRAME_SECONDS),
        }
