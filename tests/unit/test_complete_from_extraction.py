import json
from hashlib import sha256

import pytest

from scripts.complete_from_extraction import decide, evidence_from_decision, run


def extraction():
    return {'type':'ExtractionResult','status':'ASSEMBLED',
            'document':{'document_id':'synthetic-claim'},'page':{'page_number':1},
            'field_results':[{'field_name':'patient_last','ocr':{'candidates':[]},
                'ranked_candidate':None,'alternatives':[],'candidate_validations':[],
                'validation':None,'normalized_value':None,'status':'NO_VALUE'}],
            'errors':[],'warnings':[],'telemetry':{}}


def test_existing_rules_require_review_and_missing_fields():
    result=decide(extraction(),'CMS1500')
    assert result['claim_status']=='FIELD_REVIEW_REQUIRED'
    assert result['review_required'] is True
    assert result['missing_fields']
    assert 'patient_last' in result['missing_observed_fields']
    assert result['claim_decision']['stp_eligible'] is False
    assert result['claim_decision']['runtime_profile_id']!='UNBOUND'


def test_saved_decision_consumed_and_source_preserved(tmp_path):
    source=tmp_path/'ExtractionResult.json'
    source.write_text(json.dumps(extraction()))
    original=source.read_bytes()
    output=tmp_path/'result'
    final=run(source,output,'CMS1500')
    saved=output/'DecisionResult.json'
    assert final['review_required'] is True
    assert final['field_results']==extraction()['field_results']
    assert final['evidence']['decision_sha256']==sha256(saved.read_bytes()).hexdigest()
    assert evidence_from_decision(saved)==final
    assert source.read_bytes()==original
    assert json.loads((output/'completion_telemetry.json').read_text())['input_unchanged']


def test_failed_decision_never_produces_final(tmp_path):
    source=tmp_path/'input.json'
    source.write_text(json.dumps({'type':'ExtractionResult','status':'FAILED'}))
    with pytest.raises(ValueError):run(source,tmp_path/'out','CMS1500')
    assert not (tmp_path/'out/FinalClaim.json').exists()
    events=json.loads((tmp_path/'out/completion_telemetry.json').read_text())['events']
    assert len(events)==1 and events[0]['stage']=='decision' and events[0]['status']=='FAILED'


def test_duplicates_rejected():
    value=extraction();value['field_results']*=2
    with pytest.raises(ValueError,match='unique'):decide(value,'CMS1500')


def test_evidence_rejects_wrong_document(tmp_path):
    source=tmp_path/'input.json';source.write_text(json.dumps(extraction()))
    run(source,tmp_path/'out','CMS1500')
    path=tmp_path/'out/DecisionResult.json';value=json.loads(path.read_text())
    value['document']['document_id']='different';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='identity'):evidence_from_decision(path)


def test_line_sum_replaces_invalid_box28_ocr_candidate():
    """Garbage box-28 OCR must not block LINE_TOTALS derived candidate injection."""
    payload = extraction()
    payload["service_lines"] = [
        {
            "line_number": 1,
            "charges": "498.00",
            "charge_amount": "498.00",
            "candidates": [
                {"value": "498.00", "engine": "paddleocr"},
                {"value": "498.00", "engine": "rapidocr"},
            ],
        },
        {
            "line_number": 2,
            "charges": "66.00",
            "charge_amount": "66.00",
            "candidates": [
                {"value": "66.00", "engine": "paddleocr"},
                {"value": "66.00", "engine": "rapidocr"},
            ],
        },
    ]
    ocr_candidate = {
        "value": "2.22",
        "raw_value": ".2.22.",
        "engine": "tesseract_digits",
        "model_name": "unknown",
        "model_version": "unknown",
        "preprocessing_variant": "charge_digit_whitelist_fast",
        "raw_confidence": 0.7,
        "calibrated_confidence": None,
        "bounding_box": {
            "x0": 0,
            "y0": 0,
            "x1": 10,
            "y1": 10,
            "image_width": 100,
            "image_height": 100,
        },
        "latency_ms": 0.0,
        "validation_results": [],
        "evidence_reference": None,
        "estimated_cost_usd": 0.0,
        "actual_cost_usd": None,
        "preprocessing_version": "test",
        "registration_confidence": None,
        "image_quality_score": None,
        "provenance": None,
    }
    validation = {
        "field_id": "total_charge",
        "candidate_id": "total_charge:0",
        "status": "NO_VALUE",
        "reason": ["NO_RAW_VALUE", "confidence 0.70 below required 0.95"],
        "normalized_value": None,
        "validator": ["test"],
        "telemetry_reference": "t",
    }
    payload["field_results"] = [
        {
            "field_name": "total_charge",
            "ocr": {"candidates": [ocr_candidate]},
            "ranked_candidate": {
                "field_id": "total_charge",
                "candidate_id": "total_charge:0",
                "winner": "total_charge:0",
                "is_winner": True,
                "alternatives": [],
                "confidence": 0.7,
                "ranking_score": 0.1,
                "ranking_reason": ["WINNER_REQUIRES_REVIEW_DETERMINISTIC_INVALID"],
                "provider": "tesseract_digits",
                "telemetry_reference": "t",
                "ocr_candidate": ocr_candidate,
            },
            "alternatives": [],
            "candidate_validations": [validation],
            "validation": validation,
            "normalized_value": None,
            "status": "NO_VALUE",
        }
    ]
    result = decide(payload, "CMS1500")
    charge = next(d for d in result["field_decisions"] if d["field_name"] == "total_charge")
    assert charge["selected_value"] == "564.00"
    assert "LINE_TOTALS_RECONCILED" in charge["reason_codes"]
    assert "LINE_TOTALS_UNCORROBORATED" in charge["reason_codes"]
    assert "LINE_TOTALS_CORROBORATED" not in charge["reason_codes"]


def test_uncorroborated_line_sum_does_not_auto_accept_charge():
    """Single-engine line OCR without DI must not E6-AUTO total_charge."""
    payload = extraction()
    payload["service_lines"] = [
        {"line_number": 1, "charges": "424.00", "charge_amount": "424.00"},
    ]
    ocr_candidate = {
        "value": "",
        "raw_value": "",
        "engine": "paddleocr",
        "model_name": "unknown",
        "model_version": "unknown",
        "preprocessing_variant": "charge_digit_whitelist_fast",
        "raw_confidence": 0.7,
        "calibrated_confidence": None,
        "bounding_box": {
            "x0": 0,
            "y0": 0,
            "x1": 10,
            "y1": 10,
            "image_width": 100,
            "image_height": 100,
        },
        "latency_ms": 0.0,
        "validation_results": [],
        "evidence_reference": None,
        "estimated_cost_usd": 0.0,
        "actual_cost_usd": None,
        "preprocessing_version": "test",
        "registration_confidence": None,
        "image_quality_score": None,
        "provenance": None,
    }
    validation = {
        "field_id": "total_charge",
        "candidate_id": "total_charge:0",
        "status": "NO_VALUE",
        "reason": ["NO_RAW_VALUE"],
        "normalized_value": None,
        "validator": ["test"],
        "telemetry_reference": "t",
    }
    payload["field_results"] = [
        {
            "field_name": "total_charge",
            "ocr": {"candidates": [ocr_candidate]},
            "ranked_candidate": {
                "field_id": "total_charge",
                "candidate_id": "total_charge:0",
                "winner": "total_charge:0",
                "is_winner": True,
                "alternatives": [],
                "confidence": 0.7,
                "ranking_score": 0.1,
                "ranking_reason": ["WINNER_REQUIRES_REVIEW_DETERMINISTIC_INVALID"],
                "provider": "paddleocr",
                "telemetry_reference": "t",
                "ocr_candidate": ocr_candidate,
            },
            "alternatives": [],
            "candidate_validations": [validation],
            "validation": validation,
            "normalized_value": None,
            "status": "NO_VALUE",
        }
    ]
    result = decide(payload, "CMS1500")
    charge = next(d for d in result["field_decisions"] if d["field_name"] == "total_charge")
    assert charge["disposition"] != "AUTO_ACCEPTED"
    assert "LINE_TOTALS_CORROBORATED" not in charge["reason_codes"]
    assert "LINE_TOTALS_UNCORROBORATED" in charge["reason_codes"]
    assert result["review_required"] is True
