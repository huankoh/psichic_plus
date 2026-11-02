# Score molecules with PSICHIC+

The scorer loads a protein target and PSICHIC+ model once, then scores batches of molecule IDs and SMILES.
Run from the repository root in the existing PSICHIC+ environment.

```python
from applications.scorer.predictor import Predictor

predictor = Predictor(
    fasta="/inputs/target.fasta",
    model_dir="/weights/PSICHIC_plus",
    device="cpu",
    cache="/outputs/cache",
    batch_size=32,
)
results = predictor.score([("aspirin", "CC(=O)Oc1ccccc1C(=O)O"), ("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O")])
for result in results:
    print(result)
```

Successful results contain the original ID and predicted affinity.
Failures contain the original ID, stage, and reason.
Larger raw affinity values indicate stronger predicted binding.
The AdaptiveFlow integration uses its own process-managed scorer to honor native worker timeouts.
