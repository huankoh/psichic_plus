from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from applications.scorer import Failure
from . import BOLTZ_VERSION
from .models import BoltzScore, EngineRun


PROTECTED_OPTIONS = {"--out_dir", "--model", "--override", "--use_msa_server", "--help", "--"}
SECRET_OPTIONS = {"--msa_server_password", "--api_key_value"}


def recorded_command(command):
    result, redact_next = [], False
    for item in command:
        key = item.split("=", 1)[0]
        if redact_next:
            result.append("<redacted>")
            redact_next = False
        elif key in SECRET_OPTIONS:
            result.append(key + "=<redacted>" if "=" in item else key)
            redact_next = "=" not in item
        else:
            result.append(item)
    return result


def parse_affinity(path, candidate):
    with Path(path).open() as handle:
        native = json.load(handle)
    if not isinstance(native, dict):
        raise ValueError("Affinity JSON must be an object")
    json.dumps(native, allow_nan=False)
    for key in ("affinity_pred_value", "affinity_probability_binary"):
        value = native.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Missing or invalid native {key}")
    if not 0 <= native["affinity_probability_binary"] <= 1:
        raise ValueError("Native binder probability is outside [0, 1]")
    return BoltzScore(
        native["affinity_probability_binary"], native["affinity_pred_value"],
        native, str(Path(path).parent), candidate.smiles,
    )


class BoltzRunner:
    def __init__(self, python, target, directory, *, options=(), timeout=None):
        self.python = str(Path(python).expanduser().absolute())
        self.executable = str(Path(self.python).parent / "boltz")
        self.target = target
        self.directory = Path(directory).absolute()
        self.options = tuple(options)
        self.timeout = timeout
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Boltz timeout must be positive and finite")
        for option in options:
            if option.split("=", 1)[0] in PROTECTED_OPTIONS:
                raise ValueError(f"Adapter owns {option.split('=', 1)[0]}; use workflow settings instead")
        self.environment = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME"):
            self.environment.pop(key, None)
        library_paths = self.environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        self.environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            path for path in library_paths if path and Path(path) != Path(sys.prefix) / "lib"
        )
        self.metadata = {}

    def check(self):
        if not Path(self.executable).is_file():
            raise ValueError(f"Boltz CLI is missing next to the supplied Python: {self.executable}")
        code = (
            "import importlib.metadata as m,json,sys; "
            "print(json.dumps({'boltz':m.version('boltz'),'python':sys.version,"
            "'executable':sys.executable,'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"
        )
        try:
            probe = subprocess.run([self.python, "-c", code], env=self.environment, capture_output=True, text=True, check=True, timeout=60)
            self.metadata = json.loads(probe.stdout)
            if self.metadata["boltz"] != BOLTZ_VERSION:
                raise ValueError(f"Boltz {BOLTZ_VERSION} is required; found {self.metadata['boltz']}")
            help_result = subprocess.run([self.executable, "predict", "--help"], env=self.environment, capture_output=True, text=True, check=True, timeout=120)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot run the isolated Boltz environment: {error}") from error
        supported = set(re.findall(r"(?m)^\s+(--[\w-]+)", help_result.stdout))
        for token in self.options:
            if token.startswith("--") and token.split("=", 1)[0] not in supported:
                raise ValueError(f"Unsupported Boltz {BOLTZ_VERSION} option: {token}")
        self.metadata.update(model="boltz2", msa_mode=self.target.msa_mode, native_help=help_result.stdout)
        return self.metadata

    def __call__(self, candidates):
        inputs = self.directory / "inputs"
        inputs.mkdir()
        native_output = self.directory / "native"
        native_output.mkdir()
        prepared, assets = self.target.prepare(self.directory)
        manifest = []
        outcomes = {}
        for candidate in candidates:
            path = inputs / f"{candidate.key}.yaml"
            if not candidate.smiles.strip():
                outcomes[candidate.key] = Failure(candidate.key, "boltz_input", "Empty SMILES")
                continue
            self.target.render(prepared, candidate, path)
            manifest.append({**asdict(candidate), "key": candidate.key, "input": str(path)})
        (self.directory / "input_manifest.json").write_text(json.dumps(manifest, indent=2))
        command = [self.executable, "predict", str(inputs), "--out_dir", str(native_output), "--model", "boltz2"]
        if self.target.msa_mode == "server":
            command.append("--use_msa_server")
        command.extend(self.options)
        metadata = {**self.metadata, "command": recorded_command(command), "assets": assets}
        metadata.pop("native_help", None)
        error = None
        returncode = None
        with (self.directory / "stdout.log").open("w") as stdout, (self.directory / "stderr.log").open("w") as stderr:
            if manifest:
                try:
                    with subprocess.Popen(command, stdout=stdout, stderr=stderr, env=self.environment, start_new_session=True) as process:
                        try:
                            returncode = process.wait(timeout=self.timeout)
                        except (subprocess.TimeoutExpired, KeyboardInterrupt):
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            raise
                    if returncode:
                        error = f"Boltz exited with status {returncode}; see boltz/stderr.log and stdout.log"
                except subprocess.TimeoutExpired:
                    error = f"Boltz prediction exceeded the timeout of {self.timeout} seconds"
                except OSError as failure:
                    error = f"Boltz execution failed: {failure}"
        predictions = native_output / "boltz_results_inputs" / "predictions"
        for candidate in candidates:
            if candidate.key in outcomes:
                continue
            path = predictions / candidate.key / f"affinity_{candidate.key}.json"
            try:
                outcomes[candidate.key] = parse_affinity(path, candidate)
            except (OSError, ValueError, TypeError) as failure:
                reason = f"{error + '; ' if error else ''}{failure}"
                outcomes[candidate.key] = Failure(candidate.key, "boltz_output", reason)
        cache = self.environment.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
        checkpoints = {}
        for index, option in enumerate(self.options):
            name, separator, value = option.partition("=")
            if name in ("--cache", "--checkpoint", "--affinity_checkpoint"):
                value = value if separator else (self.options[index + 1] if index + 1 < len(self.options) else "")
                if name == "--cache":
                    cache = value
                else:
                    checkpoints[name] = value
        checkpoint_paths = [Path(cache).expanduser() / name for name in ("boltz2_conf.ckpt", "boltz2_aff.ckpt")]
        checkpoint_paths.extend(Path(value).expanduser() for value in checkpoints.values())
        weights = []
        for path in dict.fromkeys(checkpoint_paths):
            if path.is_file():
                with path.open("rb") as handle:
                    weights.append({"path": str(path.absolute()), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()})
        metadata.update(returncode=returncode, error=error, output_directory=str(native_output),
                        cache=str(Path(cache).expanduser().absolute()), weights=weights, working_directory=os.getcwd())
        (self.directory / "execution.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
        return EngineRun(outcomes, metadata, error)
