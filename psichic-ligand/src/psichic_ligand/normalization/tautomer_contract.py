"""Versioned tautomer contract for RDKit 2024.03.3 Enumerate+PickCanonical @100.

This is a contract change versus the previous production law
(``fixedpoint-v1``, native ``maxTransforms=1000``). It is not an RDKit 2026
upgrade. ``maxTautomers`` stays ``2**32-1``.
"""
from __future__ import annotations

PREPROCESSING_POLICY = "rdkit-2024.03.3-enumpick-max100-v1"
TAUTOMER_ENGINE = "copied-2024.03.3-enumerate-pickcanonical"
RDKIT_RELEASE = "2024.03.3"
TAUTOMER_MAX_TRANSFORMS = 100
TAUTOMER_MAX_TAUTOMERS = 2**32 - 1
REMOVE_SP3_STEREO = True
REMOVE_BOND_STEREO = True
REMOVE_ISOTOPIC_HS = True
REASSIGN_STEREO = True
LEGACY_ORACLE_POLICY = "fixedpoint-v1"
LEGACY_ORACLE_MAX_TRANSFORMS = 1000
