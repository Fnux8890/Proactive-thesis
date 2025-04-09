# DataIngestion/python_pipeline/pipeline_pkg/cleaners/__init__.py
# Make cleaner classes accessible
from .base_cleaner import BaseCleaner
from .knudjepsen_cleaners import (
    KnudjepsenMultiHeaderCsvCleaner,
    KnudjepsenExtraCsvCleaner,
    KnudjepsenBelysningCsvCleaner,
    KnudjepsenSourceUnitCleaner
)
from .aarslev_cleaners import (
    AarslevMortenCsvCleaner,
    AarslevSimpleCsvCleaner,
    AarslevCelleCsvCleaner,
    AarslevUuidHeaderCsvCleaner,
    AarslevStreamListJsonCleaner,
    AarslevStreamDictJsonCleaner,
)

# Optional: Define a factory function or dictionary to get cleaner instances
# based on the name string from the format spec.

CLEANER_REGISTRY = {
    "KnudjepsenMultiHeaderCsvCleaner": KnudjepsenMultiHeaderCsvCleaner,
    "KnudjepsenExtraCsvCleaner": KnudjepsenExtraCsvCleaner,
    "KnudjepsenBelysningCsvCleaner": KnudjepsenBelysningCsvCleaner,
    "KnudjepsenSourceUnitCleaner": KnudjepsenSourceUnitCleaner,
    "AarslevMortenCsvCleaner": AarslevMortenCsvCleaner,
    "AarslevSimpleCsvCleaner": AarslevSimpleCsvCleaner,
    "AarslevCelleCsvCleaner": AarslevCelleCsvCleaner,
    "AarslevUuidHeaderCsvCleaner": AarslevUuidHeaderCsvCleaner,
    "AarslevStreamListJsonCleaner": AarslevStreamListJsonCleaner,
    "AarslevStreamDictJsonCleaner": AarslevStreamDictJsonCleaner,
}

def get_cleaner(cleaner_name: str) -> BaseCleaner:
    """Instantiates and returns a cleaner based on its name."""
    cleaner_class = CLEANER_REGISTRY.get(cleaner_name)
    if cleaner_class:
        return cleaner_class()
    else:
        raise ValueError(f"Unknown cleaner name: {cleaner_name}")

__all__ = [
    'BaseCleaner',
    'KnudjepsenMultiHeaderCsvCleaner',
    'KnudjepsenExtraCsvCleaner',
    'KnudjepsenBelysningCsvCleaner',
    'KnudjepsenSourceUnitCleaner',
    'AarslevMortenCsvCleaner',
    'AarslevSimpleCsvCleaner',
    'AarslevCelleCsvCleaner',
    'AarslevUuidHeaderCsvCleaner',
    'AarslevStreamListJsonCleaner',
    'AarslevStreamDictJsonCleaner',
    'get_cleaner',
    'CLEANER_REGISTRY'
] 