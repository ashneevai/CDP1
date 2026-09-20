import sys
from types import ModuleType

from PIL import Image

from evaluation.phase8_2_performance import _semantic_output
from evaluation.phase8_3_finalize import calculate_costs
from workers.page_detection.text_extraction import RapidOCRTextExtractor


def test_rapidocr_thread_budget_is_passed_without_semantic_wrapper_change(monkeypatch):
    captured = {}

    class Backend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __call__(self, _image):
            return ([([[1, 2], [11, 2], [11, 8], [1, 8]], "A123", .9)], [0, 0, 0])

    backend_module = ModuleType("rapidocr_onnxruntime")
    backend_module.RapidOCR = Backend
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", backend_module)
    extractor = RapidOCRTextExtractor(intra_op_num_threads=2, inter_op_num_threads=1)

    values = extractor.extract_region(Image.new("RGB", (100, 50), "white"), 0, 0, 50, 20)

    assert captured == {"intra_op_num_threads": 2, "inter_op_num_threads": 1}
    assert [(item.text, item.confidence) for item in values] == [("A123", .9)]


def test_default_and_bounded_adapters_have_identical_output_with_same_backend():
    backend = lambda _image: (
        [([[1, 2], [11, 2], [11, 8], [1, 8]], "A123", .9)], [0, 0, 0]
    )
    image = Image.new("RGB", (100, 50), "white")
    default = RapidOCRTextExtractor(backend=backend).extract_region(image, 0, 0, 50, 20)
    bounded = RapidOCRTextExtractor(
        backend=backend, intra_op_num_threads=2, inter_op_num_threads=1,
    ).extract_region(image, 0, 0, 50, 20)
    assert default == bounded


def test_equivalence_fingerprint_excludes_generated_identity_and_time_only():
    left = {"field_id": "random-1", "produced_at": "time-1", "raw_value": "A123",
            "candidates": [{"evidence_id": "random-2", "raw_text": "A123"}]}
    right = {"field_id": "random-3", "produced_at": "time-2", "raw_value": "A123",
             "candidates": [{"evidence_id": "random-4", "raw_text": "A123"}]}

    assert _semantic_output(left) == _semantic_output(right)
    right["raw_value"] = "B456"
    assert _semantic_output(left) != _semantic_output(right)


def test_measured_machine_and_hitl_cost_formulas_are_deterministic():
    config = {
        "worker_node_hourly_cost_usd": .20,
        "cpu_cost_per_core_hour_usd": .05,
        "memory_cost_per_gb_hour_usd": .006,
        "reviewer_hourly_cost_usd": 25,
        "average_review_seconds_per_field": 5,
        "claim_open_close_overhead_seconds": 30,
        "pages_per_document": 3,
        "cloud_ai_cost_per_page_usd": 0,
        "shared_infra_cost_per_page_usd": .0001,
        "scenario_field_hitl_rates": [.05],
        "scenario_review_seconds": [3],
    }
    perf = {"pages_per_hour": 400, "wall_seconds": 900, "pages": 100,
            "cpu_seconds_per_page": 4, "memory_peak_gb": 1}
    hitl = {"review_fields_per_page": 2, "eligible_fields": 1000}
    claims = {"claim_hitl_rate": .5, "claim_stp_rate": .5}

    first = calculate_costs(config, perf, hitl, claims)
    second = calculate_costs(config, perf, hitl, claims)

    assert first == second
    assert first["throughput_based_machine_cost_per_page_usd"] == .20 / 400
    assert first["review_cost_per_field_usd"] == 25 * 5 / 3600
    assert first["hitl_field_cost_per_page_usd"] == 2 * 25 * 5 / 3600
    assert first["cloud_processing_cost_per_page_usd"] == 0
