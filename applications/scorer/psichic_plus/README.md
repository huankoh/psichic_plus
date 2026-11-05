# Score molecules with PSICHIC+

The scorer loads a protein target and PSICHIC+ model once, then scores batches of molecule IDs and SMILES.
Run from the repository root in the existing PSICHIC+ environment.

```python
from applications.scorer.psichic_plus.predictor import Predictor

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
The legacy `applications.scorer.predictor` module remains an alias for existing callers.

## Distribution output

`Predictor.score(inputs)` defaults to affinity output, as does `output="affinity"`.
The keyword-only `output` also accepts `"distribution"` or the corresponding `applications.scorer.Output` enum value.
An invalid selector raises `ValueError` before input preparation or inference.
The constructor signature and SyntheMol's float callback are unchanged.

```python
from applications.scorer import BetaDistribution, DistributionFailure, Success

results = predictor.score([("aspirin", "CC(=O)Oc1ccccc1C(=O)O")], output="distribution")
for result in results:
	if isinstance(result, Success):
		print(result.token, result.raw_affinity, result.adaptive_score)
		if isinstance(result.distribution, BetaDistribution):
			print(result.distribution.alpha, result.distribution.beta, result.distribution.concentration)
		elif isinstance(result.distribution, DistributionFailure):
			print(result.distribution.reason, result.distribution.raw_alpha, result.distribution.raw_beta)
```

`Success` retains `token`, `raw_affinity`, and `adaptive_score`, with `adaptive_score == -raw_affinity`.
The reporting name `affinity` maps to `raw_affinity`.
Its `distribution` is one of these frozen records from `applications.scorer`.

| Type | `state` | Fields |
|---|---|---|
| `DistributionNotRequested` | `not-requested` | No distribution values. |
| `BetaDistribution` | `valid` | Original `alpha` and `beta`, plus computed `concentration`. |
| `DistributionFailure` | `distribution-failure` | `reason`, `raw_alpha`, and `raw_beta` diagnostics. |

`Success(token, raw_affinity, adaptive_score)` defaults to `DistributionNotRequested()`.
Affinity-only requests ignore missing or invalid distribution fields.
Distribution requests read native `reg_alpha`, `reg_beta`, and `predicted_binding_affinity` from the same inference frame without another model pass.
Alpha and beta must be finite and positive, and their sum must be finite.
`BetaDistribution` computes concentration as `alpha + beta` during construction.
Callers cannot set it independently.

The beta distribution describes normalized activity on [0, 1], with affinity scaled as `12 * alpha / (alpha + beta)`.
It describes the model's predicted conditional distribution, not measured uncertainty or model-parameter epistemic uncertainty.
The scorer checks native affinity against this formula with `math.isclose(rel_tol=1e-6, abs_tol=1e-6)`.
The relative tolerance allows several float32 rounding steps in the sum, ratio, and scaling.
The absolute tolerance covers values near zero.
This comparison uses values from the same pass and does not claim agreement across separate stochastic predictions or environments.
The scorer always retains the original native affinity rather than replacing it with the recomputed value.

An invalid distribution preserves valid affinity in `Success` and attaches `DistributionFailure`.
Diagnostics store parameter representations as strings, including nonfinite values and `None` for absent parameters, so they remain JSON-safe.
A nonfinite affinity or native inference failure still returns `Failure`.
Distribution output currently covers this Python API.
AdaptiveFlow transport and saved-result contracts are separate work.

## Verification

`python -m pytest applications/scorer/psichic_plus/tests applications/scorer/tests -q` runs the focused tests from the repository root.
Set `PSICHIC_TEST_MODEL_DIR` to authorized weights to enable the native-frame checks on CPU and on CUDA when available.
These checks compare every returned value with its actual inference frame and verify that each valid entry is inferred once.
Without that setting, the native checks skip and the remaining tests use controlled inference output without model weights.
