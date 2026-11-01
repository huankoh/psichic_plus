"""Install a worker-local PSICHIC+ hook in a separately installed AFVS worker."""

import argparse
import importlib
import json
import logging
import math
import multiprocessing as mp
from multiprocessing.queues import Queue
import os
from pathlib import Path
import sys
import time

from predictor import Predictor


class PortableQueue(Queue):
    """Keep upstream backpressure usable where sem_getvalue is unavailable."""

    def __init__(self):
        context = mp.get_context()
        super().__init__(ctx=context)
        self.count = context.Value("i", 0)

    def __getstate__(self):
        return super().__getstate__(), self.count

    def __setstate__(self, state):
        queue_state, self.count = state
        super().__setstate__(queue_state)

    def put(self, *args, **kwargs):
        with self.count.get_lock():
            self.count.value += 1
        try:
            return super().put(*args, **kwargs)
        except BaseException:
            with self.count.get_lock():
                self.count.value -= 1
            raise

    def get(self, *args, **kwargs):
        result = super().get(*args, **kwargs)
        with self.count.get_lock():
            self.count.value -= 1
        return result

    def qsize(self):
        return self.count.value


_predictor = None
_config_path = None


def psichic_batch(summary_queue, scenario, items, temp_dir):
    global _predictor, _config_path
    if scenario["program"] != "psichic_plus":
        return original_batch(summary_queue, scenario, items, temp_dir)
    if not items:
        return

    start = time.perf_counter()
    records = [json.loads(Path(item["ligand_path"]).read_text()) for item in items]
    try:
        if _predictor is None:
            _config_path = items[0]["config_path"]
            config = json.loads(Path(_config_path).read_text())
            _predictor = Predictor(config, Path(temp_dir) / "psichic_plus")
            print(
                f"PSICHIC+ model and target initialized in docking child {os.getpid()}",
                flush=True,
            )
        if any(item["config_path"] != _config_path for item in items):
            raise ValueError("Use one PSICHIC+ scenario per worker")
        predictions = _predictor.predict(records)
    except Exception as error:
        logging.exception("PSICHIC+ batch failed")
        predictions = [
            {
                "inference_status": "failed",
                "failure_reason": f"{type(error).__name__}: {error}",
            }
            for _ in items
        ]

    seconds = time.perf_counter() - start
    for item, record, prediction in zip(items, records, predictions, strict=True):
        affinity = prediction.get("predicted_binding_affinity")
        success = prediction["inference_status"] == "success" and math.isfinite(
            float(affinity)
        )
        reason = ""
        if not success:
            reason = (
                "Nonfinite predicted affinity"
                if prediction["inference_status"] == "success"
                else str(prediction["failure_reason"])
            )
        result = {
            **record,
            "status": "success" if success else "failed",
            "predicted_binding_affinity": float(affinity) if success else None,
            "score": -float(affinity) if success else None,
            "reason": reason,
            "seconds": seconds / len(items),
            "worker_pid": os.getpid(),
            "batch_number": _predictor.batch_number if _predictor else 0,
        }
        item["status"] = result["status"]
        item["seconds"] = result["seconds"]
        item["output_path"] = str(Path(item["output_dir"]) / "output")
        item["log_path"] = str(Path(item["output_dir"]) / "stdout")
        item["log"] = {
            "base_collection_key": item["base_collection_key"],
            "collection_key": item["collection_key"],
            "ligand_key": item["ligand_key"],
            "reason": reason,
        }
        if success:
            item["score"] = result["score"]
        else:
            item.pop("score", None)
        Path(item["output_path"]).write_text(json.dumps(result, allow_nan=False) + "\n")
        Path(item["log_path"]).write_text(
            json.dumps({"status": result["status"], "reason": reason}) + "\n"
        )

    summary_queue.put(
        {
            "type": "docking_complete",
            "execution_type": "single",
            "program": "psichic_plus",
            "scenario_key": scenario["key"],
            "items": items,
        }
    )


def install_hook(path):
    global original_batch
    sys.path.insert(0, str(Path(path) / "tools" / "templates"))
    module = importlib.import_module("afvs_run")
    if module.docking_process_batch is not psichic_batch:
        original_batch = module.docking_process_batch
        module.docking_process_batch = psichic_batch
    module.Queue = PortableQueue
    return module


# Spawn imports this module before unpickling the upstream process targets.
if os.environ.get("PSICHIC_AFVS_UPSTREAM"):
    install_hook(os.environ["PSICHIC_AFVS_UPSTREAM"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--workunit", type=Path, required=True)
    parser.add_argument("--subjob", default="0")
    parser.add_argument(
        "--start-method", choices=mp.get_all_start_methods(), default="spawn"
    )
    parser.add_argument("--temp-dir", type=Path, required=True)
    args = parser.parse_args()
    mp.set_start_method(args.start_method)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "PSICHIC_AFVS_UPSTREAM": str(args.upstream.resolve()),
            "AFVS_WORKUNIT": args.workunit.name.split(".")[0],
            "AFVS_WORKUNIT_SUBJOB": args.subjob,
            "AFVS_JOB_STORAGE_MODE": "sharedfs",
            "AFVS_CONFIG_JOB_TGZ": str(args.workunit.resolve()),
            "AFVS_VCPUS": "1",
            "AFVS_TMP_PATH": str(args.temp_dir.resolve()),
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_DEFAULT_REGION": "us-east-1",
        }
    )
    install_hook(args.upstream.resolve()).main()


if __name__ == "__main__":
    main()
