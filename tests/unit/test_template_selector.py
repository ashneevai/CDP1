from dataclasses import asdict

from PIL import Image, ImageDraw

from app import register_classified_document
from packages.domain.enums import ClaimFormType
from packages.templates.registry import TemplateRegistry
from workers.page_detection.template_selector import TemplateSelector
from workers.page_detection.text_extraction import TextLine


def registry_without_references():
    source = TemplateRegistry.load_from_directory(canonical_dir=None)
    return TemplateRegistry([
        template.model_copy(update={"reference_image_path": None})
        for family in ClaimFormType for template in source.all_for_form_type(family)
    ])


def anchor_lines(template):
    return [TextLine(anchor.phrase, 0, 0, 1700, 2200, 1.0)
            for anchor in template.anchor_definitions]


def test_explicit_type_precedes_anchors():
    registry = registry_without_references()
    ub = registry.latest_for_form_type(ClaimFormType.UB04)
    result = TemplateSelector(registry).select(
        [None], text_lines={1: anchor_lines(ub)}, document_type="CMS1500",
    )
    assert result.template_id == "cms1500"
    assert result.reason == "EXPLICIT_DOCUMENT_TYPE"
    assert result.confidence == 1.0


def test_unknown_explicit_type_never_falls_back():
    result = TemplateSelector(registry_without_references()).select(
        [None], text_lines={}, document_type="invented",
    )
    assert result.template_id is None
    assert result.reason == "UNKNOWN_DOCUMENT_TYPE"


def test_anchor_match_selects_actual_page_and_template():
    registry = registry_without_references()
    cms = registry.latest_for_form_type(ClaimFormType.CMS1500)
    result = TemplateSelector(registry).select(
        [None, None], text_lines={2: anchor_lines(cms)},
    )
    assert result.template_id == cms.template_id
    assert result.page_number == 2
    # Dual-loaded templates share anchors; selection must use the active
    # release pin rather than silently switching to lexicographic latest.
    assert result.template_version == "02-12"
    assert result.reason == "ANCHOR_MATCH"


def test_multiple_qualified_pages_are_ambiguous():
    registry = registry_without_references()
    cms = registry.latest_for_form_type(ClaimFormType.CMS1500)
    result = TemplateSelector(registry).select(
        [None, None], text_lines={1: anchor_lines(cms), 2: anchor_lines(cms)},
    )
    assert result.reason == "AMBIGUOUS_TEMPLATE"
    assert result.template_id is None
    assert len(result.candidate_templates) >= 2
    # Rejected selection cannot consume the routing object or load a template.
    registration = register_classified_document([], None, None, result)
    assert registration["status"] == "UNAVAILABLE"
    assert registration["reason"] == "AMBIGUOUS_TEMPLATE"


def test_multiple_template_matches_are_ambiguous():
    registry = registry_without_references()
    lines = [line for family in (ClaimFormType.CMS1500, ClaimFormType.UB04)
             for line in anchor_lines(registry.latest_for_form_type(family))]
    result = TemplateSelector(registry).select([None], text_lines={1: lines})
    assert result.reason == "AMBIGUOUS_TEMPLATE"
    assert result.template_id is None


def test_missing_evidence_rejects_even_with_explicit_multipage_type():
    result = TemplateSelector(registry_without_references()).select(
        [None, None], text_lines={}, document_type="CMS1500",
    )
    assert result.reason == "NO_TEMPLATE_ABOVE_THRESHOLD"
    assert result.template_id is None
    assert all("REFERENCE_TEMPLATE_IMAGE_UNAVAILABLE" in item["reasons"]
               for item in result.candidate_templates)


def test_real_feature_matching_uses_existing_template(tmp_path):
    source = registry_without_references()
    cms = source.latest_for_form_type(ClaimFormType.CMS1500)
    cms = cms.model_copy(update={"reference_image_path": "reference.png"})
    registry = TemplateRegistry()
    registry.register(cms, source_dir=tmp_path)
    with Image.new("L", (300, 400), 255) as image:
        draw = ImageDraw.Draw(image)
        for y in range(30, 370, 30):
            draw.line((20, y, 280, y), fill=0, width=2)
        for x in range(20, 281, 40):
            draw.line((x, 30, x, 360), fill=0, width=2)
        image.save(tmp_path / "reference.png")
        result = TemplateSelector(registry).select([image], text_lines={})
    assert result.template_id == "cms1500"
    assert result.reason == "FEATURE_MATCH"
    assert result.confidence > 0.99
    assert asdict(result)["candidate_templates"][0]["scores"]["features"] > 0.99


def test_blank_reference_does_not_pass_registration_fallback(tmp_path):
    source = registry_without_references()
    cms = source.latest_for_form_type(ClaimFormType.CMS1500)
    cms = cms.model_copy(update={"reference_image_path": "blank.png"})
    registry = TemplateRegistry()
    registry.register(cms, source_dir=tmp_path)
    with Image.new("L", (300, 400), 255) as image:
        image.save(tmp_path / "blank.png")
        result = TemplateSelector(registry).select([image], text_lines={})
    assert result.template_id is None
    assert result.reason == "NO_TEMPLATE_ABOVE_THRESHOLD"
    assert "registration" in result.candidate_templates[0]["scores"]
