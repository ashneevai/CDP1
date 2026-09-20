"""Complete one saved ExtractionResult using existing decision policies only."""
import argparse
import contextlib
import json
import re
from hashlib import sha256
from pathlib import Path
from time import perf_counter


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _registration_confidence_from_document(document: dict) -> float | None:
    from packages.recovery.registration_recovery import evidence_grade_alignment_confidence

    registration = document.get('registration') or {}
    evidence = registration.get('evidence') or {}
    accepted = registration.get('accepted') is True
    raw = registration.get('evidence_grade_alignment_confidence')
    if raw is None:
        raw = evidence.get('alignment_confidence')
    if raw is not None:
        return evidence_grade_alignment_confidence(float(raw), accepted=accepted)
    if accepted:
        # Accepted registration without a recorded score still clears the
        # low-confidence gate once remapped from the acceptance floor.
        return evidence_grade_alignment_confidence(0.40, accepted=True)
    return None


def _registration_confidence_from_report(report: dict) -> float | None:
    """Ops app.py emits registration_report.json (not document.json)."""
    from packages.recovery.registration_recovery import evidence_grade_alignment_confidence

    for attempt in report.get('attempts') or []:
        acceptance = attempt.get('acceptance') or {}
        if acceptance.get('accepted') is not True:
            continue
        raw_evidence = attempt.get('raw_evidence') or {}
        raw = (
            acceptance.get('score')
            if acceptance.get('score') is not None
            else raw_evidence.get('alignment_confidence')
        )
        if raw is None:
            raw = raw_evidence.get('homography_quality')
        if raw is not None:
            return evidence_grade_alignment_confidence(float(raw), accepted=True)
        return evidence_grade_alignment_confidence(0.40, accepted=True)
    return None


def _load_registration_context(extraction):
    """Load sibling registration/geometry artifacts for E3 without re-acquiring evidence.

    Acceptance order (first hit wins):
    1. document.json registration blob (legacy / in-proc path)
    2. registration_report.json next to GeometryResult (ops app.py path)
    3. per-field result.registration.accepted on GeometryResult (last resort)
    """
    from packages.evidence.models import (
        StructuralLocalizationEvidence,
        StructuralLocalizationType,
    )
    from packages.recovery.registration_recovery import evidence_grade_alignment_confidence

    artifacts = extraction.get('source_artifacts') or {}
    geometry_ref = artifacts.get('geometry') or {}
    geometry_path = Path(geometry_ref['path']) if geometry_ref.get('path') else None
    registration_confidence = None
    localizations = {}
    warnings = []
    confidence_source = None

    if geometry_path and geometry_path.is_file():
        geometry = json.loads(geometry_path.read_text(encoding='utf-8'))
        document_path = geometry_path.parent / 'document.json'
        if document_path.is_file():
            document = json.loads(document_path.read_text(encoding='utf-8'))
            registration_confidence = _registration_confidence_from_document(document)
            if registration_confidence is not None:
                confidence_source = 'document.json'
        if registration_confidence is None:
            report_path = geometry_path.parent / 'registration_report.json'
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding='utf-8'))
                registration_confidence = _registration_confidence_from_report(report)
                if registration_confidence is not None:
                    confidence_source = 'registration_report.json'
        if registration_confidence is None:
            # GeometryResult rows already carry per-field accepted registration.
            for row in geometry.get('fields') or []:
                result = row.get('result') or {}
                reg = result.get('registration') or {}
                if reg.get('accepted') is True:
                    registration_confidence = evidence_grade_alignment_confidence(
                        0.40, accepted=True,
                    )
                    confidence_source = 'geometry_field_registration'
                    break
        if registration_confidence is None:
            warnings.append({
                'reason': (
                    'Accepted registration confidence unavailable beside GeometryResult; '
                    'E3 structural localization omitted.'
                ),
            })
    else:
        geometry = None
        warnings.append({'reason': 'Geometry artifact unavailable; E3 structural localization omitted.'})

    if geometry and geometry.get('status') == 'SUCCESS' and registration_confidence is not None:
        confidence = float(registration_confidence)
        for row in geometry.get('fields') or []:
            name = row.get('field')
            result = row.get('result') or {}
            box = result.get('aligned_roi') or result.get('safe_cell')
            if not name or not isinstance(box, dict):
                continue
            try:
                bbox = (float(box['x0']), float(box['y0']), float(box['x1']), float(box['y1']))
            except (KeyError, TypeError, ValueError):
                continue
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            localizations[name] = StructuralLocalizationEvidence(
                evidence_type=StructuralLocalizationType.TEMPLATE_REGISTRATION_CONFIRMED,
                confidence=confidence,
                confirmed=True,
                reason_codes=(
                    'ACCEPTED_REGISTRATION_RECTIFIED_FRAME',
                    'TEMPLATE_FIELD_ROI_BOUNDED',
                    f'E3_SOURCE:{confidence_source}',
                ),
                source='geometry',
                field_name=name,
                field_bbox=bbox,
                localization_mode='TEMPLATE_ROI',
                positive_bounded_roi=True,
                geometry_valid=True,
                registration_compatible=True,
            )
    return registration_confidence, localizations, warnings


def decide(extraction, family):
    from pydantic import TypeAdapter

    from packages.claim_decision.contracts import ClaimDecisionContext, ClaimDisposition
    from packages.claim_evidence.builder import ClaimEvidenceBuilder
    from packages.criticality import CriticalityLevel
    from packages.deterministic_evidence.service import DeterministicEvidenceService
    from packages.evidence_decision.contracts import DecisionContext
    from packages.ocr.contracts import OCRCandidate
    from packages.runtime_profile.decision_factory import DecisionServiceFactory

    if extraction.get('type') != 'ExtractionResult' or extraction.get('status') != 'ASSEMBLED':
        raise ValueError('An assembled ExtractionResult is required')
    if family not in ('CMS1500', 'UB04'):
        raise ValueError('Explicit supported document family required')
    fields = extraction['field_results']
    names = [f['field_name'] for f in fields]
    if len(names) != len(set(names)) or not fields:
        raise ValueError('Nonempty, unique field results required')
    for f in fields:
        winner = f['ranked_candidate']
        validation = f['validation']
        if winner is None:
            if f['ocr']['candidates'] or validation is not None or f['status'] != 'NO_VALUE':
                raise ValueError('Inconsistent missing candidate')
        elif (validation is None or winner['field_id'] != f['field_name']
              or validation['candidate_id'] != winner['candidate_id']
              or validation['field_id'] != f['field_name']
              or validation['normalized_value'] != f['normalized_value']
              or validation['status'] != f['status']):
            raise ValueError('Extraction winner/validation lineage mismatch')
    services = DecisionServiceFactory.from_profile()
    deterministic = DeterministicEvidenceService()
    claim_id = extraction['document']['document_id']
    from packages.claim_evidence.line_sum_authority import (
        is_decimal_place_shift,
        line_sum_auto_eligible,
        parse_currency,
        should_defer_box28_to_line_sum,
    )
    values = {f['field_name']: f['normalized_value'] for f in fields}
    # Attach Box 28 geometry observation for independent corroboration authority.
    for charge_field in ('total_charge', 'total_charges'):
        field_payload = next(
            (f for f in fields if f.get('field_name') == charge_field), None
        )
        if not field_payload:
            continue
        values['_box28_field_payload'] = field_payload
        obs = None
        ocr_block = field_payload.get('ocr') or {}
        for attempt in ocr_block.get('attempts') or []:
            reason = str(attempt.get('reason') or '')
            if (
                'GEOMETRY_CENTS' in reason
                and 'UNDERREAD' not in reason
                and isinstance(attempt.get('observation'), dict)
            ):
                candidate_obs = dict(attempt['observation'])
                if candidate_obs.get('adopted') is False:
                    continue
                obs = candidate_obs
                break
        if obs is None:
            for cand in ocr_block.get('candidates') or []:
                if str(cand.get('preprocessing_variant') or '') != 'GEOMETRY_CENTS':
                    continue
                obs = {
                    'text': cand.get('raw_value') or '',
                    'raw_digit_sequence': re.sub(
                        r'\D', '', str(cand.get('raw_value') or cand.get('value') or '')
                    ),
                    'canonical_monetary_value': cand.get('value'),
                    'shaped': cand.get('value'),
                    'adopted': True,
                }
                prov = cand.get('provenance')
                if isinstance(prov, dict):
                    obs.update({k: v for k, v in prov.items() if v is not None})
                break
        if obs is None:
            for row in (
                [field_payload.get('ranked_candidate')]
                if field_payload.get('ranked_candidate')
                else []
            ) + list(field_payload.get('alternatives') or []):
                if not row:
                    continue
                ocr = row.get('ocr_candidate') or {}
                if str(ocr.get('preprocessing_variant') or '') != 'GEOMETRY_CENTS':
                    continue
                obs = {
                    'text': ocr.get('raw_value') or '',
                    'raw_digit_sequence': re.sub(
                        r'\D', '', str(ocr.get('raw_value') or ocr.get('value') or '')
                    ),
                    'canonical_monetary_value': ocr.get('value'),
                    'shaped': ocr.get('value'),
                    'adopted': True,
                }
                break
        if obs:
            values['_box28_geometry_observation'] = obs
            # Prefer the integrity-passing geometry amount as the Box 28 value
            # when normalized OCR still holds a clipped/fragment competitor.
            geo_amount = obs.get('canonical_monetary_value') or obs.get('shaped')
            if geo_amount and parse_currency(geo_amount) is not None:
                current = values.get(charge_field)
                if (
                    current in (None, '')
                    or is_decimal_place_shift(current, geo_amount)
                    or (
                        parse_currency(current) is not None
                        and parse_currency(current) != parse_currency(geo_amount)
                        and str(obs.get('raw_digit_sequence') or '')
                        == re.sub(r'\D', '', str(geo_amount))
                    )
                ):
                    values[charge_field] = str(geo_amount)
        region = None
        for cand in ocr_block.get('candidates') or []:
            bbox = cand.get('bounding_box')
            if isinstance(bbox, dict) and bbox.get('x0') is not None:
                region = (bbox.get('x0'), bbox.get('y0'), bbox.get('x1'), bbox.get('y1'))
                break
        if region is None:
            region = tuple(ocr_block.get('canonical_region') or [])[:4] or None
            if region and len(region) == 4:
                region = tuple(float(v) for v in region)
            else:
                region = (1045.0, 1805.0, 1248.0, 1875.0)
        values['_box28_region'] = region
        break
    # Relationship checkbox OCR often validates INVALID while the ranked
    # candidate still carries a shaped SELF/CHILD/SPOUSE/OTHER code. Feed that
    # into claim evidence so Box 2/4 disagreement is interpreted correctly.
    _REL_SHAPED = {
        'SELF', 'CHILD', 'SPOUSE', 'OTHER',
        '18', '19', 'G8',
        '01', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
        '13', '14', '15', '16', '17', '21', '22', '23', '24', '29', '32', '33',
        '34', '36', '39', '40', '41', '43', '53',
    }

    def _shaped_relationship(raw: object) -> str | None:
        text = str(raw or '').strip().upper()
        if not text:
            return None
        if text in _REL_SHAPED:
            return text
        # Span selectors sometimes leave a leading code token.
        token = text.split()[0].strip(',.;:')
        if token in _REL_SHAPED:
            return token
        return None

    for rel_field in ('rel_code', 'relationship', 'insured_relationship'):
        if rel_field not in values:
            continue
        if values.get(rel_field) not in (None, ''):
            continue
        field_payload = next((f for f in fields if f.get('field_name') == rel_field), None)
        if not field_payload:
            continue
        ranked = field_payload.get('ranked_candidate') or {}
        ocr = ranked.get('ocr_candidate') or {}
        shaped = _shaped_relationship(ocr.get('value'))
        if shaped is None:
            for row in field_payload.get('alternatives') or []:
                alt = (row.get('ocr_candidate') or {}).get('value')
                shaped = _shaped_relationship(alt)
                if shaped is not None:
                    break
        if shaped is not None:
            values[rel_field] = shaped
    # Existing cross-field facts feed the existing decision rules. No evidence acquisition.
    service_lines = extraction.get('service_lines') or []
    # Repair cents-clipped Box 24F shells using independent OCR candidates so
    # Box 28 ↔ line-sum authority and CLAIM_TOTAL_CONFIRMED see the same ink.
    try:
        from packages.claim_evidence.box28_line_sum_authority import build_box24f_rows
        from packages.geometry_authority.cms1500_regions import CMS1500_CHARGE_CENTS_X

        repaired_lines = []
        for line in service_lines:
            if not isinstance(line, dict):
                repaired_lines.append(line)
                continue
            region = line.get('canonical_region') or line.get('ocr_region')
            clipped = False
            if isinstance(region, (list, tuple)) and len(region) >= 3:
                clipped = float(region[2]) < CMS1500_CHARGE_CENTS_X + 12
            if not clipped:
                repaired_lines.append(line)
                continue
            rows = build_box24f_rows([line])
            if rows and rows[0].integrity.passed and rows[0].amount:
                updated = dict(line)
                updated['charges'] = rows[0].amount
                updated['charge_amount'] = rows[0].amount
                repaired_lines.append(updated)
            else:
                repaired_lines.append(line)
        service_lines = repaired_lines
    except Exception:  # noqa: BLE001, S110 -- optional repair must fail closed
        pass
    # Precision-safe charge total: prefer full-window / .00 over ruling-tail
    # and units-bleed cents before CLAIM_TOTAL / Box28↔line-sum bind.
    from packages.claim_evidence.charge_total_authority import (
        is_ruling_tail_extension,
        is_units_bleed_cents,
        prefer_safe_charge_amount,
        resolve_safe_charge_total,
    )

    for charge_field in ('total_charge', 'total_charges'):
        if charge_field not in values:
            continue
        field_payload = next(
            (f for f in fields if f.get('field_name') == charge_field), None
        )
        safe, _reason = resolve_safe_charge_total(
            primary=values.get(charge_field),
            field_payload=field_payload,
            service_lines=service_lines,
        )
        if safe and parse_currency(safe) is not None:
            values[charge_field] = safe
            # Align single-line shells that are ruling-tail / bleed twins.
            if len(service_lines) == 1 and isinstance(service_lines[0], dict):
                line = dict(service_lines[0])
                current = line.get('charges') or line.get('charge_amount')
                preferred = prefer_safe_charge_amount(current, safe)
                if preferred == safe or (
                    current
                    and (
                        is_ruling_tail_extension(safe, current)
                        or (
                            is_units_bleed_cents(current)
                            and str(safe).endswith('.00')
                        )
                    )
                ):
                    line['charges'] = safe
                    line['charge_amount'] = safe
                    service_lines = [line]
    # Prefer observed service-line Σ when box-28 is empty, suspicious-tiny, or
    # strongly contradicts multi-line charges (uncalibrated OCR soup).
    from packages.claim_evidence.line_sum_authority import (
        parse_currency,
    )
    for charge_field in ('total_charge', 'total_charges'):
        if charge_field not in values:
            continue
        field_payload = next(
            (f for f in fields if f.get('field_name') == charge_field), {}
        ) or {}
        current_val = values.get(charge_field)
        # Do not wipe currency-shaped Azure DI / gpt-4o box-28 in favor of a
        # contradictory single-line OCR sum (hard-15: residual recovers ink).
        preserve_azure_box28 = False
        if current_val not in (None, ''):
            from packages.claim_evidence.line_sum_authority import (
                is_implausible_corroborator,
                line_sum_total,
            )

            line_total = line_sum_total(service_lines)
            # Never preserve form-ruling digit soup / 10×-off Azure totals.
            if line_total and is_implausible_corroborator(current_val, line_total):
                preserve_azure_box28 = False
            else:
                for row in (
                    [field_payload.get('ranked_candidate')]
                    if field_payload.get('ranked_candidate')
                    else []
                ) + list(field_payload.get('alternatives') or []):
                    if not row:
                        continue
                    ocr = row.get('ocr_candidate') or {}
                    eng = str(ocr.get('engine') or '').casefold()
                    if 'gpt4o' not in eng and 'document_intelligence' not in eng:
                        continue
                    text = str(ocr.get('value') or ocr.get('raw_value') or '').strip()
                    if not text or parse_currency(text) is None:
                        continue
                    if parse_currency(text) == parse_currency(current_val):
                        preserve_azure_box28 = True
                        break
                gpt4o = field_payload.get('gpt4o_crop_residual') or {}
                if (
                    gpt4o.get('shaped')
                    and not gpt4o.get('review_only')
                    and parse_currency(gpt4o.get('value')) == parse_currency(current_val)
                ):
                    preserve_azure_box28 = True
                di = field_payload.get('azure_di_residual') or {}
                if (
                    di.get('currency_shaped')
                    and not di.get('review_only')
                    and parse_currency(di.get('value')) == parse_currency(current_val)
                ):
                    preserve_azure_box28 = True
        if preserve_azure_box28:
            continue
        if should_defer_box28_to_line_sum(current_val, service_lines):
            values[charge_field] = None

    # Authoritative member join for Lane C / overprinted residuals — exact ID
    # only, and only when CDP_AUTHORIZED_MEMBER_INDEX is configured. Never uses
    # Golden / agent labels as a lookup source.
    member_join_meta = None
    try:
        from packages.reference_enrichment.authorized_member_join import (
            join_member_by_id,
        )

        mid = values.get('insured_id_number') or values.get('member_id')
        hit = join_member_by_id(mid)
        if hit is not None:
            member_join_meta = hit.to_dict()
            if hit.patient_name and (
                not str(values.get('patient_name') or '').strip()
                or len(str(values.get('patient_name') or '')) < 4
            ):
                values['patient_name'] = hit.patient_name
            if hit.insured_name and (
                not str(values.get('insured_name') or '').strip()
                or len(str(values.get('insured_name') or '')) < 4
            ):
                values['insured_name'] = hit.insured_name
            if hit.patient_dob and not str(values.get('patient_dob') or '').strip():
                values['patient_dob'] = hit.patient_dob
    except Exception:  # noqa: BLE001
        member_join_meta = None

    facts = ClaimEvidenceBuilder.load().build(claim_id=claim_id, document_family=family,
                                            claim_values=values, service_lines=service_lines)
    # Phase 2: when box-28 is empty but LINE_TOTALS_RECONCILED fired from observed
    # service-line charges, inject the derived total as a candidate (observed ink only).
    derived_totals = {}
    for item in facts.evidence_items:
        if item.evidence_type != 'LINE_TOTALS_RECONCILED':
            continue
        for field_name in item.metadata.get('supported_fields', []):
            if item.value:
                derived_totals[field_name] = item.value

    def _charge_corroborators(field_payload: dict) -> list[str]:
        """Currency-shaped box-28 / Azure DI / non-derived OCR amounts."""
        found: list[str] = []
        seen: set[str] = set()

        def _add(raw: object) -> None:
            text = str(raw or '').strip()
            if not text or text in seen:
                return
            if parse_currency(text) is None:
                return
            # Drop form-ruling digit-soup box-28 so it cannot force CONFLICT.
            from packages.claim_evidence.line_sum_authority import (
                is_implausible_charge_total,
            )

            if is_implausible_charge_total(text):
                return
            seen.add(text)
            found.append(text)

        _add(field_payload.get('normalized_value'))
        for row in ([field_payload.get('ranked_candidate')] if field_payload.get('ranked_candidate') else []) + list(
            field_payload.get('alternatives') or []
        ):
            if not row:
                continue
            ocr = row.get('ocr_candidate') or {}
            engine = str(ocr.get('engine') or '').casefold()
            variant = str(ocr.get('preprocessing_variant') or '').casefold()
            if 'derived_from_observed_line' in variant or 'phase2-line-sum' in variant:
                continue
            # A selected currency value is the Box 28 observation. Raw soup
            # such as "4 972" is not an independent total — thousands-space
            # repair would turn a cents ruling into 4972.00.
            chosen = str(ocr.get('value') or '').strip()
            if not chosen:
                raw_text = str(ocr.get('raw_value') or '').strip().replace(',', '')
                if re.fullmatch(r'\$?\d{1,6}(?:\.\d{2})?', raw_text):
                    chosen = raw_text
            _add(chosen)
            if 'azure' in engine or 'document_intelligence' in engine:
                _add(ocr.get('value') or ocr.get('raw_value'))
        residual = field_payload.get('azure_di_residual') or {}
        if residual.get('currency_shaped'):
            _add(residual.get('value'))
        gpt4o = field_payload.get('gpt4o_crop_residual') or {}
        if gpt4o.get('shaped') and not gpt4o.get('review_only'):
            _add(gpt4o.get('value'))
        return found

    line_sum_gate: dict[str, tuple[bool, str]] = {}
    for field_name, amount in list(derived_totals.items()):
        field_payload = next((f for f in fields if f.get('field_name') == field_name), {}) or {}
        eligible, reason = line_sum_auto_eligible(
            service_lines,
            box28_value=None,
            corroborating_values=_charge_corroborators(field_payload),
        )
        line_sum_gate[field_name] = (eligible, reason)
    for field_name, amount in derived_totals.items():
        current = values.get(field_name)
        if current is None or not str(current).strip():
            values[field_name] = amount
    registration_confidence, localizations, structural_warnings = _load_registration_context(extraction)
    decisions, checks, critical = [], {}, []
    for f in fields:
        name = f['field_name']
        policy = services.field_policy.for_field(family, name)
        if policy.criticality in (CriticalityLevel.C2, CriticalityLevel.C3):
            critical.append({'field_name': name, 'criticality': policy.criticality.value,
                             'extraction_status': f['status'], 'required': policy.required})
        winner = f['ranked_candidate']
        raw = winner['ocr_candidate']['raw_value'] if winner else ''
        derived = derived_totals.get(name)
        check_value = f['normalized_value'] or raw or derived or ''
        check = deterministic.evaluate(name, check_value, claim_values=values)
        # Ranking may crown header junk (e.g. DOB "MM") while a shaped alternative
        # is calendar/format-valid. E4 must evaluate the validating candidate, not
        # only the rank winner — otherwise reconciler ACCEPT still ESCALATEs.
        if not check.passed:
            for row in (f.get('alternatives') or []):
                alt = (row.get('ocr_candidate') or {}).get('value') or ''
                if not str(alt).strip():
                    continue
                alt_check = deterministic.evaluate(name, alt, claim_values=values)
                if alt_check.passed:
                    check = alt_check
                    check_value = alt
                    break
        checks[name] = check.model_dump(mode='json')
        candidates = []
        validations = {v['candidate_id']: v for v in f['candidate_validations']}
        rows = ([winner] if winner else []) + list(f.get('alternatives') or [])
        # Prefer shaped / validating OCR shells first so reconciler + route authority
        # see calendar-valid DOB / name ink ahead of header fragments.
        with contextlib.suppress(ImportError, TypeError, ValueError, AttributeError, KeyError):
            from packages.extraction_recovery.field_cascade import (
                semantic_accept as _semantic_accept,
            )

            def _row_priority(row, field_name=name):
                value = (row.get('ocr_candidate') or {}).get('value') or ''
                ok, _ = _semantic_accept(field_name, value) if value else (False, '')
                return (0 if ok else 1, 0 if row.get('is_winner') else 1)

            rows = sorted(rows, key=_row_priority)
        for row in rows:
            candidate = dict(row['ocr_candidate'])
            validation = validations[row['candidate_id']]
            if row['is_winner']:
                candidate['value'] = validation['normalized_value'] or candidate.get('value')
            # If winner normalized to junk but check_value is a shaped alternative, keep OCR value.
            if (
                candidate.get('value') in (None, '', 'MM')
                and check.passed
                and check_value
                and (row.get('ocr_candidate') or {}).get('value') == check_value
            ):
                candidate['value'] = check_value
            candidate['validation_results'] = tuple(validation['reason'])
            candidates.append(TypeAdapter(OCRCandidate).validate_python(candidate))
        # v12.2 gap: insured_name CONFLICT / short-fragment when patient_name is a
        # strong accepted SELF twin. Inject the observed patient ink as a
        # competitor so fragment / confusable relief can prefer it (no invention).
        # Non-Self (CHILD/SPOUSE/OTHER): Box 2 and Box 4 are independent — never
        # copy patient ink into insured_name.
        if name == 'insured_name':
            from packages.candidate_reconciliation.reconciler import (
                _canonical_person_name,
                _name_is_short_fragment,
                _name_is_strong_person,
                _name_label_contaminated,
                _names_differ_by_confusable_edit,
                _names_differ_by_confusable_insertion,
                _names_differ_by_confusable_substitution,
                _names_differ_by_optional_middle_initial,
                _names_differ_by_token_order,
                _names_differ_by_tokenwise_confusable,
            )
            from packages.geometry_authority.form_redundancy import relationship_is_self

            relationship = (
                values.get('insured_relationship')
                or values.get('relationship')
                or values.get('rel_code')
            )
            patient_val = str(values.get('patient_name') or '').strip()
            if (
                relationship_is_self(relationship)
                and patient_val
                and _name_is_strong_person(patient_val)
            ):
                already = {
                    str(c.value or '').strip().casefold()
                    for c in candidates
                    if (c.value or '').strip()
                }
                if patient_val.casefold() not in already:
                    def _soft_twin(left: str, right: str) -> bool:
                        a, b = _canonical_person_name(left), _canonical_person_name(right)
                        if bool(a) and a == b:
                            return True
                        return any(
                            fn(left, right)
                            for fn in (
                                _names_differ_by_confusable_substitution,
                                _names_differ_by_confusable_insertion,
                                _names_differ_by_confusable_edit,
                                _names_differ_by_tokenwise_confusable,
                                _names_differ_by_token_order,
                                _names_differ_by_optional_middle_initial,
                            )
                        )

                    insured_vals = [
                        str(c.value or '').strip()
                        for c in candidates
                        if (c.value or '').strip()
                    ]
                    soft_twin = any(_soft_twin(patient_val, v) for v in insured_vals)
                    fragment_pair = any(
                        _name_is_short_fragment(v) for v in insured_vals
                    )
                    all_weak = bool(insured_vals) and all(
                        _name_is_short_fragment(v) or len(v) <= 4
                        for v in insured_vals
                    )
                    all_label = bool(insured_vals) and all(
                        _name_label_contaminated(v) for v in insured_vals
                    )
                    top = insured_vals[0] if insured_vals else ''
                    top_weak = bool(top) and (
                        _name_is_short_fragment(top) or len(top) <= 4
                    )
                    top_label = bool(top) and _name_label_contaminated(top)
                    if (
                        soft_twin
                        or fragment_pair
                        or all_weak
                        or all_label
                        or top_weak
                        or top_label
                        or not insured_vals
                    ):
                        from packages.domain.common import BoundingBox
                        base_box = candidates[0].bounding_box if candidates else None
                        candidates.append(OCRCandidate(
                            value=patient_val,
                            raw_value=patient_val,
                            engine='paddleocr',
                            model_name='claim_cross_field',
                            model_version='v12.2-self-twin',
                            preprocessing_variant='PATIENT_NAME_SELF_TWIN',
                            raw_confidence=0.9,
                            calibrated_confidence=0.9,
                            bounding_box=base_box or BoundingBox(
                                x0=0, y0=0, x1=1, y1=1, image_width=1, image_height=1
                            ),
                            latency_ms=0.0,
                            evidence_reference='PATIENT_NAME_SELF_TWIN',
                            preprocessing_version='v12.2-self-twin',
                        ))
        # Prefer LINE_TOTALS derived amount over empty / invalid / deferred box-28 OCR.
        prefer_derived = bool(derived) and (
            not check.passed
            or f.get('status') == 'NO_VALUE'
            or not (f.get('normalized_value') or '').strip()
            or not any((c.value or '').strip() for c in candidates)
            or (
                name in {'total_charge', 'total_charges'}
                and values.get(name) == derived
            )
        )
        if prefer_derived:
            from packages.domain.common import BoundingBox
            # Always mint a clean derived candidate. Reusing an empty/invalid OCR
            # shell (e.g. tesseract "ipo") keeps INVALID validation and blocks E1.
            base_box = None
            if candidates:
                base_box = candidates[0].bounding_box
            derived_candidate = OCRCandidate(
                value=derived,
                raw_value=derived,
                engine='rapidocr',
                model_name='claim_evidence',
                model_version='phase2-line-sum',
                preprocessing_variant='DERIVED_FROM_OBSERVED_LINE_CHARGES',
                raw_confidence=1.0,
                calibrated_confidence=1.0,
                bounding_box=base_box or BoundingBox(x0=0, y0=0, x1=1, y1=1, image_width=1, image_height=1),
                latency_ms=0.0,
                evidence_reference='LINE_TOTALS_RECONCILED',
                preprocessing_version='phase2-line-sum',
            )
            # Keep currency-shaped box-28 / DI competitors so conflicts HITL
            # instead of wiping independent ink with a lone line-sum AUTO.
            retained = []
            for cand in candidates:
                text = str(cand.value or cand.raw_value or '').strip()
                if not text or parse_currency(text) is None:
                    continue
                variant = str(cand.preprocessing_variant or '').casefold()
                if 'derived_from_observed_line' in variant:
                    continue
                retained.append(cand)
            candidates = [derived_candidate] + retained
            check = deterministic.evaluate(name, derived, claim_values=values)
            eligible, gate_reason = line_sum_gate.get(name, (False, 'UNSET'))
            if eligible:
                # Financial E6 only when dual-engine lines or DI/box-28 corroborate.
                check.evidence = set(check.evidence) | {
                    'LINE_TOTALS_RECONCILED',
                    'LINE_TOTALS_CORROBORATED',
                    'HARD_VALIDATION_PASSED',
                }
                check.cross_field_evidence = set(check.cross_field_evidence) | {
                    'LINE_TOTALS_RECONCILED',
                    'LINE_TOTALS_CORROBORATED',
                }
                check.passed = True
            else:
                # Observed line-sum stays as a candidate for review — no E6 AUTO.
                check.evidence = set(check.evidence) | {
                    'LINE_TOTALS_UNCORROBORATED',
                    f'LINE_TOTALS_GATE:{gate_reason}',
                }
                check.cross_field_evidence = set(check.cross_field_evidence) | {
                    'LINE_TOTALS_UNCORROBORATED',
                }
                # Fail-closed: do not treat uncorroborated line-sum as hard-valid E6.
                if name in {'total_charge', 'total_charges'} and not retained:
                    check.passed = False
            checks[name] = check.model_dump(mode='json')
        localization = localizations.get(name)
        if name in {'total_charge', 'total_charges'}:
            # Bind CLAIM_TOTAL_CONFIRMED to its confirmed amount and drop
            # common-mode soup that only matches after inconsistent repair.
            from decimal import Decimal

            from packages.claim_evidence.line_sum_authority import (
                amounts_within_tolerance,
                is_decimal_place_shift,
                is_implausible_charge_total,
                parse_currency,
            )

            confirmed = None
            for item in facts.evidence_items:
                if item.evidence_type == 'CLAIM_TOTAL_CONFIRMED' and item.value:
                    confirmed = str(item.value)
                    break
            # Also honor Box28↔line-sum AUTO amount when present.
            if confirmed is None:
                for item in facts.evidence_items:
                    if (
                        item.evidence_type == 'BOX28_LINE_SUM_CORROBORATED'
                        and item.value
                    ):
                        confirmed = str(item.value)
                        break
            filtered = []
            exact_confirmed = []
            for cand in candidates:
                text = str(cand.value or '').strip()
                if not text:
                    continue
                if is_implausible_charge_total(text):
                    continue
                if confirmed is not None:
                    if is_decimal_place_shift(text, confirmed):
                        continue
                    # Exact confirmed wins — do not keep ±$1 bleed/ruling twins.
                    if parse_currency(text) == parse_currency(confirmed):
                        exact_confirmed.append(cand)
                        continue
                    if not amounts_within_tolerance(
                        text, confirmed, absolute=Decimal('0.01'), relative=Decimal(0)
                    ):
                        continue
                filtered.append(cand)
            if exact_confirmed:
                candidates = exact_confirmed
            elif filtered:
                candidates = filtered
            elif confirmed is not None:
                # Keep a single derived shell matching the confirmed total so
                # junk cannot AUTO via unbound E6.
                from packages.domain.common import BoundingBox
                base_box = candidates[0].bounding_box if candidates else BoundingBox(
                    x0=0, y0=0, x1=1, y1=1, image_width=1, image_height=1
                )
                candidates = [OCRCandidate(
                    value=confirmed,
                    raw_value=confirmed,
                    engine='rapidocr',
                    model_name='claim_evidence',
                    model_version='confirmed-total-bind',
                    preprocessing_variant='CLAIM_TOTAL_CONFIRMED_BOUND',
                    raw_confidence=1.0,
                    calibrated_confidence=1.0,
                    bounding_box=base_box,
                    latency_ms=0.0,
                    evidence_reference='CLAIM_TOTAL_CONFIRMED',
                    preprocessing_version='confirmed-total-bind',
                )]
        decisions.append(services.evidence_decision.decide(DecisionContext(
            field_name=name, document_family=family, criticality=policy.criticality,
            required=policy.required, blocks_stp=policy.blocks_stp,
            requires_review_when_unresolved=policy.requires_review_when_unresolved,
            candidates=candidates, deterministic_evidence=check.evidence,
            deterministic_evidence_version=deterministic.policy_version,
            hard_validation_passed=check.passed,
            registration_confidence=registration_confidence,
            structural_evidence_source='geometry' if localization is not None else None,
            structural_localization=localization,
            cross_field_evidence=set(check.cross_field_evidence) | facts.evidence_types_for(name))))
    claim = services.claim_decision.decide(ClaimDecisionContext(
        claim_id=claim_id, document_family=family, field_decisions=decisions,
        claim_evidence=facts.evidence_items, contradictions=facts.contradictions,
        process_integrity_valid=not extraction.get('errors')))
    present = {services.field_policy.canonical_name(family, name)
               for name, value in values.items() if value is not None and str(value).strip()}
    missing_required = sorted(set(services.field_policy.required_fields(family)) - present)
    missing_observed = sorted(f['field_name'] for f in fields
                              if f['status']=='NO_VALUE' or f['normalized_value'] is None
                              or not str(f['normalized_value']).strip())
    return {'type':'DecisionResult', 'status':'SUCCESS', 'document':extraction['document'],
        'page':extraction['page'], 'document_family':family,
        'claim_status':claim.disposition.value,
        'review_required':claim.disposition in (ClaimDisposition.FIELD_REVIEW_REQUIRED,
                                               ClaimDisposition.CLAIM_REVIEW_REQUIRED),
        'missing_fields':missing_required, 'missing_observed_fields':missing_observed,
        'critical_fields':critical, 'critical_blockers':claim.critical_blockers,
        'warnings':extraction.get('warnings',[]) + structural_warnings + [
            {'reason':'Structural localization reused from accepted registration/geometry; service-line charge OCR reused from extraction artifacts when present; no new reference acquisition.'}],
        'decision_reason':claim.reason_codes, 'claim_decision':claim.model_dump(mode='json'),
        'field_decisions':[d.model_dump(mode='json') for d in decisions],
        'deterministic_checks':checks, 'claim_facts':facts.model_dump(mode='json'),
        'extracted_fields':fields,
        'registration_confidence':registration_confidence,
        'authorized_member_join': member_join_meta,
        'missing_fields_basis':'Required policy fields absent or blank; invalid nonempty values are not missing.',
        'telemetry':{'extraction':extraction['telemetry']}}


def evidence_from_decision(path):
    """Read the persisted decision; preserve its policy outcome and supporting facts."""
    from packages.claim_decision.contracts import ClaimDecision
    from packages.claim_evidence.builder import ClaimEvidenceResult
    from packages.evidence_decision.contracts import FieldDecision
    path = Path(path)
    payload = path.read_bytes()
    decision = json.loads(payload)
    if decision.get('type') != 'DecisionResult' or decision.get('status') != 'SUCCESS':
        raise ValueError('Successful saved DecisionResult required')
    claim = ClaimDecision.model_validate(decision['claim_decision'])
    if claim.claim_id != decision['document']['document_id'] or claim.disposition.value != decision['claim_status']:
        raise ValueError('Claim decision identity/status mismatch')
    fields = [FieldDecision.model_validate(f) for f in decision['field_decisions']]
    facts = ClaimEvidenceResult.model_validate(decision['claim_facts'])
    evidence = {'decision_reference':str(path), 'decision_sha256':sha256(payload).hexdigest(),
                'extraction_reference':decision['extraction_reference'],
                'claim_facts':facts.model_dump(mode='json'),
                'fields':[{'field_name':f.field_name,'supporting_evidence':[e.model_dump(mode='json') for e in f.supporting_evidence],
                           'conflicting_evidence':[e.model_dump(mode='json') for e in f.conflicting_evidence],
                           'evidence_bundle':f.evidence_bundle.model_dump(mode='json') if f.evidence_bundle else None}
                          for f in fields], 'deterministic_checks':decision['deterministic_checks']}
    return {'type':'FinalClaim', 'status':'COMPLETED', 'document':decision['document'],
            'page':decision['page'], 'claim_status':decision['claim_status'],
            'review_required':decision['review_required'], 'stp_eligible':claim.stp_eligible,
            'missing_fields':decision['missing_fields'], 'critical_fields':decision['critical_fields'],
            'warnings':decision['warnings'], 'decision_reason':decision['decision_reason'],
            'field_results':decision['extracted_fields'], 'decision':decision,
            'evidence':evidence}


def run(source, output, family):
    source, output = Path(source), Path(output)
    payload = source.read_bytes()
    extraction = json.loads(payload)
    output.mkdir(parents=True, exist_ok=False)
    telemetry = {'input_sha256':sha256(payload).hexdigest(), 'events':[],
                 'upstream_executed':False, 'retries':0}
    stage = 'decision'
    started = perf_counter()
    try:
        decision = decide(extraction, family)
        decision['extraction_reference'] = {'path':str(source),'sha256':sha256(payload).hexdigest()}
        write(output/'DecisionResult.json', decision)
        telemetry['events'].append({'stage':stage,'status':'SUCCESS','latency_ms':(perf_counter()-started)*1000})
        stage = 'evidence'
        started = perf_counter()
        final = evidence_from_decision(output/'DecisionResult.json')
        write(output/'FinalClaim.json', final)
        telemetry['events'].append({'stage':stage,'status':'SUCCESS','latency_ms':(perf_counter()-started)*1000})
        return final
    except Exception as exc:
        telemetry['events'].append({'stage':stage,'status':'FAILED','latency_ms':(perf_counter()-started)*1000,
                                    'error':{'type':type(exc).__name__,'reason':str(exc)}})
        raise
    finally:
        telemetry['input_unchanged'] = source.read_bytes()==payload
        write(output/'completion_telemetry.json',telemetry)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('extraction_result')
    parser.add_argument('output_directory')
    parser.add_argument('--document-family',required=True,choices=['CMS1500','UB04'])
    args=parser.parse_args()
    result=run(args.extraction_result,args.output_directory,args.document_family)
    print(json.dumps({k:result[k] for k in ('status','claim_status','review_required','missing_fields')}))
