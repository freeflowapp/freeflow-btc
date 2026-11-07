import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc import FRAME_SECONDS  # noqa: E402
from btc.segments import IDX_TO_CHORD, frames_to_segments  # noqa: E402


def test_runs():
    assert frames_to_segments([0, 0, 1, 1, 1, 24], 1.0) == [
        {"start": 0.0, "end": 2.0, "chord": "C"},
        {"start": 2.0, "end": 5.0, "chord": "C:min"},
        {"start": 5.0, "end": 6.0, "chord": "N"},
    ]


def test_empty():
    assert frames_to_segments([], FRAME_SECONDS) == []


def test_one_frame():
    assert frames_to_segments([8], FRAME_SECONDS) == [{"start": 0.0, "end": 0.093, "chord": "E"}]


def test_rounding_10_over_108():
    segs = frames_to_segments([0] * 5 + [2] * 103, FRAME_SECONDS)
    assert segs == [{"start": 0.0, "end": 0.463, "chord": "C"}, {"start": 0.463, "end": 10.0, "chord": "C#"}]
    assert FRAME_SECONDS == 10 / 108


def test_vocabulary():
    assert len(IDX_TO_CHORD) == 25
    assert IDX_TO_CHORD[-1] == "N"
    assert IDX_TO_CHORD[:4] == ["C", "C:min", "C#", "C#:min"]
    assert all(c.endswith(":min") for c in IDX_TO_CHORD[1:24:2])
