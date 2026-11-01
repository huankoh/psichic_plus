"""Private, persistent inference process. The worker bounds both pipe directions."""

import os
import sys

# Native libraries can write fd 1 directly, so reserve the response FD first.
protocol = os.fdopen(os.dup(1), 'wb', buffering=0)
os.dup2(2, 1)
sys.stdout = sys.stderr

import ctypes
import json
from pathlib import Path
import signal
import tempfile
import threading
import time

from .adapter import MAX_FRAME


def watch_parent():
    expected_parent = os.environ.get('PSICHIC_AFVS_PARENT_PID')
    parent = int(expected_parent) if expected_parent is not None else os.getppid()
    if os.getppid() != parent or (expected_parent is None and parent == 1):
        os._exit(1)
    if sys.platform == 'linux':
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0:
            raise RuntimeError('Could not enable parent death signal')
        if os.getppid() != parent:
            os._exit(1)
    else:
        def watch():
            while os.getppid() == parent:
                time.sleep(0.2)
            os._exit(1)
        threading.Thread(target=watch, daemon=True).start()


def main():
    watch_parent()
    fasta_path, model_dir, device_name, threads = sys.argv[1:]
    if device_name == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(Path(__file__).with_name('_screening')))
    import torch
    import pandas as pd
    import inference as screening

    torch.set_num_threads(int(threads))
    device = torch.device('cuda:0' if device_name == 'cuda' else device_name)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = None
    protein_dict = None
    fasta = screening.read_single_fasta(Path(fasta_path))
    with tempfile.TemporaryDirectory(prefix='psichic-afvs-') as workspace:
        output = Path(workspace)
        while True:
            data = sys.stdin.buffer.readline(MAX_FRAME + 1)
            if not data:
                return
            if len(data) > MAX_FRAME or not data.endswith(b'\n'):
                raise ValueError('Invalid or oversized request frame')
            smiles = json.loads(data)
            if not isinstance(smiles, list) or not smiles or any(not isinstance(value, str) for value in smiles):
                raise ValueError('Expected a nonempty bounded list of SMILES')
            frame = pd.DataFrame({'ID': [str(index) for index in range(len(smiles))], 'Ligand': smiles})
            frame = screening.prepare_input_frame(frame, 'Ligand', fasta, None)
            if protein_dict is None:
                protein_dict, _, *_ = screening.build_protein_state(frame, None, False)
                model = screening.PSICHICPlusModel.from_pretrained(
                    Path(model_dir), device=device, motifpool_dense_strategy='pyg_to_dense',
                    protein_mincut_strategy='pyg_dense_mincut', protein_dense_cache_strategy='none',
                )
                print('PSICHIC+ initialized target and model once', file=sys.stderr, flush=True)
            frame = screening.initialize_output_columns(frame)
            loader, *_ = screening.build_loader(frame, protein_dict, None, output, output / 'ligands', len(frame), 0, False, False, device_name, False, 'input_order', False)
            result, *_ = screening.run_screening(frame, model, loader, protein_dict, loader.dataset.ligands, output, device, False, True, 'fp32', False, False, 'thread', False)
            result, _ = screening.apply_invalid_ligand_policy(result, 'emit_null')
            rows = []
            for position, row in result.iterrows():
                if row['inference_status'] == 'success':
                    rows.append({'position': int(position), 'affinity': float(row['predicted_binding_affinity'])})
                else:
                    rows.append({'position': int(position), 'stage': str(row['failure_stage']), 'reason': str(row['failure_reason'])})
            response = json.dumps(rows, allow_nan=False).encode() + b'\n'
            if len(response) > MAX_FRAME:
                raise ValueError('Oversized response frame')
            sent = 0
            while sent < len(response):
                sent += protocol.write(response[sent:])
            print(f'PSICHIC+ completed batch of {len(rows)}', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
