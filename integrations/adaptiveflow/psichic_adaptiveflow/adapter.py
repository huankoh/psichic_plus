"""Native collection identity, worker ownership, and bounded model execution."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import selectors
import signal
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass

MAX_FRAME = 8 * 1024 * 1024


@dataclass(frozen=True)
class LigandKey:
    collection_key: str
    ligand_key: str


@dataclass(frozen=True)
class ScoreInput:
    key: LigandKey
    replica: int
    path: Path
    smiles: str | None = None


@dataclass(frozen=True)
class Affinity:
    value: float

    def __post_init__(self):
        if not math.isfinite(self.value):
            raise ValueError("Affinity must be finite")


@dataclass(frozen=True)
class Failure:
    reason: str
    stage: str = "prediction"


@dataclass(frozen=True)
class Target:
    protein_fasta: Path
    model_dir: Path
    device: str

    @classmethod
    def read(cls, config_path):
        config_path = Path(config_path)
        config = {}
        for line in config_path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            key, sep, value = line.partition("=")
            if not sep or key.strip() not in {"protein_fasta", "model_dir", "device"}:
                raise ValueError(f"Unsupported PSICHIC+ config line: {line}")
            if key.strip() in config:
                raise ValueError(f"Duplicate PSICHIC+ config key: {key}")
            config[key.strip()] = value.strip()
        fasta = Path(config["protein_fasta"])
        if not fasta.is_absolute():
            fasta = config_path.parent / fasta
        fasta = fasta.resolve()
        model = Path(config["model_dir"])
        if not model.is_absolute():
            raise ValueError("model_dir must be an absolute worker-visible path")
        device = config.get("device", "cpu")
        if device not in {"cpu", "cuda"} and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("device must be cpu, cuda, or cuda:<index>")
        if not fasta.is_file():
            raise ValueError(f"Missing protein_fasta: {fasta}")
        lines = [line.strip() for line in fasta.read_text().splitlines() if line.strip()]
        if not lines or not lines[0].startswith(">") or sum(line.startswith(">") for line in lines) != 1 or len(lines) < 2:
            raise ValueError("protein_fasta must contain exactly one nonempty FASTA record")
        for name in ("config.json", "degree.pt", "model.pt"):
            if not (model / name).is_file():
                raise ValueError(f"Missing model file on this machine: {model / name}")
        return cls(fasta, model, device)


def validate_preparation(config):
    for index, method in enumerate(config['docking_scenario_methods']):
        if method != 'psichic_plus':
            continue
        if config['ligand_library_format'] != 'smi':
            raise ValueError("psichic_plus requires native per-ligand .smi archives")
        for key in ('run_atom_check', 'run_fast_check', 'dynamic_tranche_filtering'):
            if int(config.get(key, 0)):
                raise ValueError(f"psichic_plus requires {key}=0 for SMILES inputs")
        for key in ('threads_per_docking', 'program_timeout'):
            if int(config[key]) < 1:
                raise ValueError(f"{key} must be positive")
        if int(config['docking_scenario_batchsizes'][index]) < 1 or int(config['docking_scenario_replicas'][index]) < 1:
            raise ValueError("PSICHIC+ batch size and replica count must be positive")
        Target.read(Path(config['docking_scenario_basefolder']) / config['docking_scenario_names'][index] / 'config.txt')


def unpack_collection(item):
    """Reject ambiguous native member IDs before extraction can overwrite them."""
    root = Path(item['temp_dir'])
    number = item['collection']['collection_number']
    ligands = {}
    try:
        with tarfile.open(item['local_path']) as archive:
            members = []
            for member in archive.getmembers():
                if member.isdir():
                    continue
                path = Path(member.name)
                if len(path.parts) != 2 or path.parts[0] != number or not member.isfile():
                    raise ValueError(f"Unsupported archive member: {member.name}")
                if path.name == '.listing':
                    members.append(member)
                    continue
                key = path.name.split('.')[0]
                if not key or path.suffix != '.smi':
                    raise ValueError(f"Expected per-ligand .smi member: {member.name}")
                if key in ligands:
                    raise ValueError(f"Ambiguous duplicate ligand ID in archive: {key}")
                ligands[key] = {'path': str(root / path), 'base_collection_key': item['collection_key'], 'collection_key': item['collection_key']}
                members.append(member)
            if not ligands:
                raise ValueError("Archive has no .smi ligands")
            if item['collection']['mode'] == 'prescreen_mode':
                listing = [member for member in members if Path(member.name).name == '.listing']
                if len(listing) != 1:
                    raise ValueError("Sparse archive requires one .listing")
                selected = {}
                seen = set()
                for line in archive.extractfile(listing[0]).read().decode().splitlines():
                    collection, ligand, index = line.split(',')
                    if ligand in seen:
                        raise ValueError(f"Ambiguous duplicate ligand ID in .listing: {ligand}")
                    seen.add(ligand)
                    if ligand not in ligands:
                        raise ValueError(f"Missing .listing ligand: {ligand}")
                    if int(index) < 0:
                        raise ValueError("Negative sparse index")
                    if int(index) < item['collection']['prescreen_ligands_per_tranche']:
                        selected[ligand] = dict(ligands[ligand], collection_key=collection)
                ligands = selected
            elif item['collection']['mode'] == 'named':
                ligands = {key: ligands[key] for key in item['collection']['ligands']}
            for member in members:
                destination = root / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open('wb') as output:
                    shutil.copyfileobj(source, output)
    except Exception as error:
        reason = f"Invalid native SMILES archive: {error}"
        if not ligands:
            ligands[''] = {'path': '', 'base_collection_key': item['collection_key'], 'collection_key': item['collection_key']}
        for ligand in ligands.values():
            ligand['failure'] = reason
    return ligands


class Scorer:
    def __init__(self, target: Target, threads: int, command=None):
        self.target = target
        self.threads = threads
        self.command = command
        self.process = None
        self.disabled = False
        self.generation = 0

    def _start(self):
        if self.disabled:
            raise RuntimeError("Previous inference child could not be reaped; respawn disabled")
        env = dict(os.environ)
        env['PSICHIC_AFVS_PARENT_PID'] = str(os.getpid())
        env['OMP_NUM_THREADS'] = str(self.threads)
        env['MKL_NUM_THREADS'] = str(self.threads)
        env['OPENBLAS_NUM_THREADS'] = str(self.threads)
        if self.target.device == 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = ''
        command = self.command or [sys.executable, '-m', 'psichic_adaptiveflow._child', str(self.target.protein_fasta), str(self.target.model_dir), self.target.device, str(self.threads)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, bufsize=0, start_new_session=True, env=env)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.generation += 1

    def close(self):
        process = self.process
        if process is None:
            return
        for stream in (process.stdin, process.stdout):
            stream.close()
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except OSError as error:
                if process.poll() is None and not isinstance(error, ProcessLookupError):
                    self.disabled = True
                    raise RuntimeError(f"Cannot terminate inference child {process.pid}: {error}; respawn disabled") from error
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is not None:
            self.process = None
            return
        self.disabled = True
        raise RuntimeError(f"Inference child {process.pid} could not be reaped; respawn disabled")

    def score(self, inputs: list[ScoreInput], timeout: float) -> list[Affinity | Failure]:
        deadline = time.monotonic() + timeout
        try:
            payload = json.dumps([entry.smiles for entry in inputs], allow_nan=False).encode() + b'\n'
            if len(payload) > MAX_FRAME:
                raise ValueError("PSICHIC+ batch exceeds the 8 MiB request limit; reduce native batch size")
            if self.process is None:
                self._start()
            process = self.process
            received = bytearray()
            sent = 0
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdin, selectors.EVENT_WRITE)
                selector.register(process.stdout, selectors.EVENT_READ)
                while b'\n' not in received:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("PSICHIC+ batch timeout including initialization and transport")
                    events = selector.select(remaining)
                    for key, _ in events:
                        try:
                            if key.fileobj is process.stdin:
                                sent += os.write(process.stdin.fileno(), payload[sent:sent + 65536])
                                if sent == len(payload):
                                    selector.unregister(process.stdin)
                            else:
                                data = os.read(process.stdout.fileno(), 65536)
                                if not data:
                                    raise RuntimeError("PSICHIC+ child closed its response stream")
                                received.extend(data)
                                if len(received) > MAX_FRAME:
                                    raise ValueError("PSICHIC+ response exceeds the 8 MiB limit")
                        except BlockingIOError:
                            continue
            if sent != len(payload) or not received.endswith(b'\n') or received.count(b'\n') != 1:
                raise ValueError("PSICHIC+ child returned an invalid response frame")
            rows = json.loads(received)
            if not isinstance(rows, list) or len(rows) != len(inputs):
                raise ValueError("PSICHIC+ child returned the wrong result count")
            results = [None] * len(inputs)
            for row in rows:
                position = row['position']
                if type(position) is not int or not 0 <= position < len(inputs) or results[position] is not None:
                    raise ValueError("PSICHIC+ child returned invalid result positions")
                if 'affinity' in row:
                    if type(row['affinity']) not in (int, float):
                        raise ValueError("PSICHIC+ child returned a nonnumeric affinity")
                    results[position] = Affinity(float(row['affinity']))
                else:
                    results[position] = Failure(str(row['reason']), str(row['stage']))
            if time.monotonic() > deadline:
                raise TimeoutError('PSICHIC+ batch timeout while decoding response')
            return results
        except Exception as error:
            reason = str(error)
            try:
                self.close()
            except RuntimeError as cleanup_error:
                reason += f"; {cleanup_error}"
            return [Failure(reason, 'supervision') for _ in inputs]


def complete_batch(batch, scenario, scorers, setup, clean, temp_dir):
    items = batch['items']
    ready = []
    inputs = []
    for item in items:
        item['log'].update(scenario_key=item['scenario_key'], replica_index=item['replica_index'], stage='input')
        try:
            setup(item, 'single', temp_dir)
            if item['attrs'].get('psichic_input_failure'):
                raise ValueError(item['attrs'].pop('psichic_input_failure'))
            entry = ScoreInput(LigandKey(item['collection_key'], item['ligand_key']), item['replica_index'], Path(item['ligand_path']))
            if entry.path.stat().st_size > MAX_FRAME:
                raise ValueError("Per-ligand SMILES file exceeds 8 MiB")
            lines = entry.path.read_text().splitlines()
            if len(lines) != 1 or not lines[0].strip():
                raise ValueError("Expected one nonempty SMILES line per ligand file")
            smiles = lines[0].split()[0]
            item['attrs']['smi'] = smiles
            inputs.append(ScoreInput(entry.key, entry.replica, entry.path, smiles))
            ready.append(item)
        except Exception as error:
            item['log']['reason'] = str(error)
    if ready:
        try:
            if scenario['key'] not in scorers:
                scorers[scenario['key']] = Scorer(Target.read(scenario['config']), ready[0]['threads_per_docking'])
            scorer = scorers[scenario['key']]
            results = scorer.score(inputs, ready[0]['timeout'])
            for item, result in zip(ready, results):
                item['log']['child_generation'] = scorer.generation
                if isinstance(result, Affinity):
                    item['attrs']['psichic_affinity'] = result.value
                    item['score'] = -result.value
                    item['status'] = 'success'
                    item['log']['stage'] = 'prediction'
                else:
                    item['log'].update(reason=result.reason, stage=result.stage)
        except Exception as error:
            for item in ready:
                item['log'].update(reason=str(error), stage='configuration')
    for item in items:
        if 'tmp_run_dir' in item:
            try:
                clean(item)
            except OSError as error:
                item['status'] = 'failed'
                item['log'].update(stage='cleanup', reason=str(error))
        try:
            Path(item['log_path']).write_text(json.dumps(dict(item['log'], status=item['status'], psichic_affinity=item['attrs'].get('psichic_affinity')), allow_nan=False) + '\n')
        except OSError as error:
            item['status'] = 'failed'
            item['log'].update(stage='logging', reason=str(error))
    batch.update(type='docking_complete', execution_type='single')
