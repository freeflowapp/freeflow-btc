"""Cog entry point: audio in, BTC major/minor chord segments out (JSON string)."""

import json

from cog import BasePredictor, Input, Path

import btc


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.recognizer = btc.Recognizer()

    def predict(self, audio: Path = Input(description="Audio file (any format ffmpeg reads)")) -> str:
        return json.dumps(self.recognizer.recognize(str(audio)))
