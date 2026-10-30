from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meldai_v2.router import v2_router
    from meldai_v2.services.prescription_service import PrescriptionServiceV2
    from meldai_v2 import req_dtos, res_dtos, dtos

__all__ = [
    "v2_router",
    "PrescriptionServiceV2",
    "req_dtos",
    "res_dtos",
    "dtos",
]


def __getattr__(name: str):
    if name == "v2_router":
        from meldai_v2.router import v2_router
        return v2_router
    if name == "PrescriptionServiceV2":
        from meldai_v2.services.prescription_service import PrescriptionServiceV2
        return PrescriptionServiceV2
    if name in ("req_dtos", "res_dtos", "dtos"):
        import importlib
        return importlib.import_module(f"meldai_v2.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

