"""Local tautomer extension: 2024.03.3 truncated search with cheaper copies."""

from ..._native import load_native

_ext = load_native(__name__ + ".tautomer_ext")
pick_canonical_tautomer = _ext.pick_canonical_tautomer
normalize_once = _ext.normalize_once
papyrus_rest = _ext.papyrus_rest
SOURCE_REVISION = "Release_2024_03_3"
PATCH_NOTE = (
    "cached-findSSSR + kekule dup skip + sanit without conj/hybrid + "
    "copy-elision of known kekule dups + stereo-free aromatic-invariant "
    "sanit skip + pre-copy stereo-free aromatic-invariant skip + "
    "local scoreTautomer with one symmetrized SSSR + "
    "kekuleKey VF2 presence-bit skip; "
    "maxTransforms=100; SMILES map keys unchanged; "
    "native Papyrus salts/uncharge/export around pickle-free pick"
)
