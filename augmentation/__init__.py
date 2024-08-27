from .diff_mix import *
from .real_mix import *
from .ti_mix import *

AUGMENT_METHODS = {
    "ti-mix": TextualInversionMixup,
    "ti_aug": TextualInversionMixup,
    "real-aug": DreamboothLoraMixup,
    "real-mix": DreamboothLoraMixup,
    "real-gen": RealGeneration,
    "diff-mix": DreamboothLoraMixup,
    "diff-aug": DreamboothLoraMixup,
    "diff-gen": DreamboothLoraGeneration,
}
