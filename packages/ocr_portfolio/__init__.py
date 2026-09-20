"""OCR portfolio helpers: controlled variants and adapter contracts."""

from .monetary_recognizer import (
    MonetaryRead,
    MonetaryRecognizeResult,
    apply_charge_line_resolution,
    is_ruling_tick_charge,
    monetary_variants_extended,
    prefer_charge_ink_amount,
    recognize_monetary_crop,
    recover_dollars_from_split_raw,
    resolve_service_charge,
    shape_dollars_ruling_amount,
    shape_monetary,
    split_charge_at_vertical_ruling,
)
from .monetary_variants import CropVariant, iter_variant_ids, monetary_crop_variants

__all__ = [
    "CropVariant",
    "MonetaryRead",
    "MonetaryRecognizeResult",
    "apply_charge_line_resolution",
    "is_ruling_tick_charge",
    "iter_variant_ids",
    "monetary_crop_variants",
    "monetary_variants_extended",
    "prefer_charge_ink_amount",
    "recognize_monetary_crop",
    "recover_dollars_from_split_raw",
    "resolve_service_charge",
    "shape_dollars_ruling_amount",
    "shape_monetary",
    "split_charge_at_vertical_ruling",
]
