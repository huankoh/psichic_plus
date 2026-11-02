# SyntheMol source patch

The patch extends SyntheMol revision `60302339148339355fc3e568ce200ed06207f99c` with the `psichic_plus` individual scorer.
The original launcher, catalogue loader, reaction matching, search, score composition, and writers remain upstream code.
No upstream correctness fixes are included.
The installed distribution retains its original MIT license and copyright notices.
The installer checks the exact revision, the patch SHA-256, and `git apply --check` before building a temporary source copy.
The installer excludes bundled catalogue resources from the wheel.

SyntheMol is copyright (c) 2025 Kyle Swanson and distributed under the MIT license.
See the [pinned upstream license](https://github.com/swansonk14/SyntheMol/blob/60302339148339355fc3e568ce200ed06207f99c/LICENSE).
