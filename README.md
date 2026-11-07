# btc

Chord recognition from audio, packaged with [Cog](https://cog.run) for
[Replicate](https://replicate.com): the major/minor model of
[BTC](https://github.com/jayg996/BTC-ISMIR19) (Bi-directional Transformer for
Chord recognition, Park et al., ISMIR 2019).

You give it an audio file. It returns where the chord changes and what each
chord is, one of the 24 major and minor triads or no chord.

## Input

| Input | Meaning |
|---|---|
| `audio` | Any file ffmpeg can decode. |

## Output

A JSON string:

```json
{
  "model": "btc-majmin",
  "source": "jayg996/BTC-ISMIR19@2682317",
  "weights_license": "MIT (Copyright (c) 2019 Jonggwon Park; ...)",
  "duration": 241.3,
  "frame_seconds": 0.09259259259259259,
  "segments": [
    {"start": 0.0, "end": 0.463, "chord": "N"},
    {"start": 0.463, "end": 12.685, "chord": "E"},
    {"start": 12.685, "end": 15.926, "chord": "C#:min"}
  ]
}
```

`chord` is one of the 24 labels in sharps, `"C"`, `"C:min"`, `"C#"`,
`"C#:min"` … `"B"`, `"B:min"`, or `"N"` for no chord. Times are seconds into
the file, rounded to the millisecond. `frame_seconds` is 10/108: the model
labels every frame of that length, and a segment is a run of frames with the
same label.

## How it runs

Features and inference follow upstream `test.py` exactly: the audio is
resampled to 22050 Hz mono, the constant-Q transform (144 bins, 24 per octave,
hop 2048) is taken in 10-second pieces and joined, then `log(|CQT| + 1e-6)`,
normalised with the checkpoint's mean and std, cut into 108-frame windows and
labelled by the model one window at a time. The one difference is the audio
decoding: ffmpeg decodes the file to a temporary wav first, so any format
ffmpeg reads works. It runs on CPU.

## Licence

`btc/model.py`, `btc/transformer_modules.py` and the weights `btc/btc_model.pt`
are copied from BTC-ISMIR19 at commit 2682317, which is MIT licensed
(Copyright (c) 2019 Jonggwon Park). The weights ship in that repository under
the same licence file; it states no separate licence for them. The one change
to the copied code is that `np.float` (removed in NumPy 1.24) is now `float`;
`btc/model.py` also has its imports adjusted and its `__main__` demo removed.
Everything else here is MIT licensed, Copyright (c) 2026 Freeflow. See
`LICENSE`.

## Development

```
pip install numpy pytest && python -m pytest tests     # no model needed
pip install -r requirements.txt && python cli.py song.mp3 --out song.json
```

## Publishing

`.github/workflows/push.yml` runs the tests, builds the image and pushes a new
version to `r8.im/$REPLICATE_MODEL` on every change to the model code on
`main` (or by hand from the Actions tab). It needs:

- repository variable `REPLICATE_MODEL`, for example `owner/chords`
- repository secret `REPLICATE_CLI_AUTH_TOKEN`, the Cog CLI token from
  https://replicate.com/auth/token (an ordinary API token is refused by `cog login`)
