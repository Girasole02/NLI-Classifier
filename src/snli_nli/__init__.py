"""SNLI natural language inference package.

Import submodules explicitly so a baseline-only workflow does not load PyTorch:

    from snli_nli.data import load_snli_frames, clean_split
    from snli_nli.baseline import train_and_evaluate_baseline
"""

from snli_nli.constants import LABEL_MAP, LABEL_NAMES, VALID_LABELS

__version__ = "1.0.0"
__all__ = ["LABEL_MAP", "LABEL_NAMES", "VALID_LABELS", "__version__"]
