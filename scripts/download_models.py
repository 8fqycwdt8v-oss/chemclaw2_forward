"""Download / warm up model weights for installed predictors.

For HuggingFace-hosted models (ReactionT5), `transformers.AutoModel.from_pretrained`
will pull weights on first use; running this script once after install primes
the cache so the first server request isn't slow.

For models that require manual checkpoint download (Molecular Transformer,
T5Chem, MEGAN, GraphRXN, Parrot, ASKCOS), this script prints the download URL
and the expected destination path.
"""

from __future__ import annotations

import sys

from chemclaw2_forward.config import get_settings


MANUAL_DOWNLOADS = {
    "molecular_transformer": (
        "MIT_mixed_augm_model_average.pt",
        "https://github.com/pschwllr/MolecularTransformer/releases",
        "MOLECULAR_TRANSFORMER_MODEL_PATH",
    ),
    "t5chem": (
        "USPTO_500_MT.tar.bz2 (extract `product/` subdir)",
        "https://yzhang.hpc.nyu.edu/T5Chem/",
        "T5CHEM_MODEL_PATH",
    ),
    "megan": (
        "uspto_50k pretrained checkpoint",
        "https://github.com/molecule-one/megan/releases",
        "MEGAN_MODEL_PATH",
    ),
    "graphrxn": (
        "trained model.pt",
        "https://github.com/jidushanbojue/GraphRXN",
        "GRAPHRXN_MODEL_PATH",
    ),
    "parrot": (
        "uspto_condition checkpoint",
        "https://github.com/wangxr0526/Parrot",
        "PARROT_MODEL_PATH",
    ),
    "reagents_mt": (
        "reagents model.pt",
        "https://github.com/Academich/reagents",
        "REAGENTS_MT_MODEL_PATH",
    ),
    "two_stage_dnn": (
        "Chen & Li 2024 supplementary checkpoints",
        "https://doi.org/10.1186/s13321-024-00805-4 (supp. info)",
        "TWO_STAGE_DNN_MODEL_PATH",
    ),
    "askcos_condition": (
        "ASKCOS NN context recommender weights",
        "https://gitlab.com/mlpds_mit/askcosv2",
        "ASKCOS_CONTEXT_MODEL_PATH",
    ),
}


def warm_huggingface() -> None:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        print("[skip] transformers not installed; can't warm ReactionT5.")
        return
    print("[reaction_t5_v2] downloading sagawa/ReactionT5v2-forward ...")
    AutoTokenizer.from_pretrained("sagawa/ReactionT5v2-forward")
    AutoModelForSeq2SeqLM.from_pretrained("sagawa/ReactionT5v2-forward")
    print("[reaction_t5_v2] done.")


def print_manual_instructions() -> None:
    settings = get_settings()
    print("\nManual checkpoint downloads required for these predictors:")
    print(f"(default cache dir: {settings.model_cache_dir})\n")
    for name, (artefact, url, env_var) in MANUAL_DOWNLOADS.items():
        print(f"  {name}:")
        print(f"    artefact:    {artefact}")
        print(f"    download:    {url}")
        print(f"    env var:     {env_var}")
        print()


def main() -> int:
    warm_huggingface()
    print_manual_instructions()
    return 0


if __name__ == "__main__":
    sys.exit(main())
