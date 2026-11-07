"""Run the recogniser locally without Cog: python cli.py AUDIOFILE [--device D] [--out FILE]."""

import argparse
import json
import time

import btc

p = argparse.ArgumentParser()
p.add_argument("audio")
p.add_argument("--device", default=None)
p.add_argument("--out", default="-")
a = p.parse_args()

t0 = time.time()
result = btc.Recognizer(a.device).recognize(a.audio)
result["seconds_taken"] = round(time.time() - t0, 1)
text = json.dumps(result, indent=1)
if a.out == "-":
    print(text)
else:
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
