import hashlib
import inspect
import json
from pathlib import Path
import sys


def source_provenance(predictor):
    inference = Path(predictor.helpers.__file__).resolve()
    screening = inference.parent
    paths = set(screening.rglob("*.py"))
    paths.add(inference)
    paths.add(Path(inspect.getsourcefile(type(predictor))).resolve())
    loaded = {"inference": str(inference)}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and (
            path.is_relative_to(screening)
            or name.split(".")[0] in ("featurizers", "psichic_model", "psichic_ligand")
        ):
            loaded[name] = str(path)
            paths.add(path)
    files = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(paths)]
    return {"screening_directory": str(screening), "loaded_modules": loaded, "source_files": files,
            "source_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
