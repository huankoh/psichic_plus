import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

mode = sys.argv[1]
workspace = os.environ.get("PSICHIC_AFVS_WORKSPACE")
if workspace:
    Path(workspace, "child-artifact").write_text("created by child")
if mode == "descendant":
    subprocess.Popen([sys.executable, __file__, "before-read", sys.argv[2]])
    while True:
        time.sleep(1)
if mode == "before-read":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(str(os.getpid()))
    while True:
        time.sleep(1)
for line in sys.stdin:
    values = json.loads(line)
    if mode in {"initialize", "predict"}:
        if mode == "predict":
            print("prediction started", file=sys.stderr, flush=True)
        while True:
            time.sleep(1)
    if mode == "partial":
        os.write(1, b'[{"position":')
        while True:
            time.sleep(1)
    if mode == "eof":
        sys.exit(0)
    if mode == "normalize-timeout":
        print(
            json.dumps(
                [
                    {
                        "position": 0,
                        "stage": "normalization",
                        "reason": "Standardization timeout",
                    }
                ]
                + [{"position": i, "affinity": 6.0} for i in range(1, len(values))]
            ),
            flush=True,
        )
    elif mode == "bad-position":
        print(json.dumps([{"position": 0, "affinity": 7.0}] * len(values)), flush=True)
    elif mode == "nan":
        print('[{"position":0,"affinity":NaN}]', flush=True)
    else:
        print(
            json.dumps(
                [
                    {"position": position, "affinity": float(position + 5)}
                    for position in reversed(range(len(values)))
                ]
            ),
            flush=True,
        )
