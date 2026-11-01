# Native extension

Rebuild in the activated conda environment from the repository root.

```bash
pip install --no-build-isolation --no-deps ./psichic-ligand
python -m psichic_ligand verify
```

The extension is compiled at installation and loaded without runtime compilation.
