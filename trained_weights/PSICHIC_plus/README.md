# psichic_plus Checkpoint Directory

This public alpha / pre-release intentionally does not include pretrained
psichic_plus weight files.

Expected local-only files for inference:

- `config.json`: architecture metadata, tracked in this repository.
- `degree.pt`: degree-scaler checkpoint artifact, not tracked.
- `model.pt`: pretrained model checkpoint, not tracked.

Place compatible local checkpoint files here, or pass another checkpoint
directory with `--model_dir`.
