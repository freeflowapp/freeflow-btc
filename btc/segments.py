"""Frame labels to chord segments. Pure; needs nothing beyond the standard library."""

# utils/mir_eval_modules.py idx2chord in BTC-ISMIR19 at 2682317, in order: the major/minor model's 25 classes.
IDX_TO_CHORD = ['C', 'C:min', 'C#', 'C#:min', 'D', 'D:min', 'D#', 'D#:min', 'E', 'E:min', 'F', 'F:min', 'F#',
                'F#:min', 'G', 'G:min', 'G#', 'G#:min', 'A', 'A:min', 'A#', 'A#:min', 'B', 'B:min', 'N']


def frames_to_segments(labels, frame_seconds, names=IDX_TO_CHORD):
    """Runs of equal labels as [{"start", "end", "chord"}], seconds rounded to 3 decimals.
    Frame k starts at k * frame_seconds; the last run ends after the last frame."""
    out = []
    labels = list(labels)
    start = 0
    for k in range(1, len(labels) + 1):
        if k == len(labels) or labels[k] != labels[start]:
            out.append({"start": round(start * frame_seconds, 3), "end": round(k * frame_seconds, 3),
                        "chord": names[labels[start]]})
            start = k
    return out
