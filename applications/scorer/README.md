# Scoring applications

[PSICHIC+](psichic_plus/README.md) provides the existing predictor used by SyntheMol.
[Boltz-2](boltz2/README.md) adds optional native scoring and four ranking workflows.
Boltz runs in its own environment and is not required by PSICHIC+, SyntheMol, or AdaptiveFlow.

The canonical PSICHIC+ import is `applications.scorer.psichic_plus.predictor.Predictor`.
The legacy `applications.scorer.predictor` module remains an alias with the same constructor and result types.
Shared `Success` and `Failure` types remain in `applications.scorer`.
AdaptiveFlow retains its separate process-managed adapter and native interface.
