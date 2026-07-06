# Eval Workflow

This package runs backend-only extraction evaluation against a folder of claim examples.

It reuses the backend scan route logic by calling `/api/ai/scan-documents` through a local `TestClient`.

## Dataset Layout

Each claim uses one `<folder>.json`.

```text
eval_examples/
  claim_001/
    images/
      page1.jpg
      page2.jpg
    claim_001.json
  claim_002/
    images/
      scan.jpg
    claim_002.json
```

Flat image folders also work:

```text
claim_003/
  receipt.jpg
  note.png
  claim_003.json
```

## Ground Truth Format

```json
{
  "images": ["images/page1.jpg", "images/page2.jpg"],
  "patient_name_candidates": ["Budi Saputra", "Siti Aminah"],
  "expected": {
    "summary": {
      "detected_documents": ["Kwitansi", "Formulir Rawat Jalan"],
      "medical_facility_name": "RS Sehat Sentosa",
      "service_date": "2024-08-10",
      "transaction_date": null,
      "currency": "IDR",
      "total_amount": 125000
    },
    "patient_info": {
      "name": "Budi Saputra",
      "date_of_birth": "1985-05-15",
      "age": 39,
      "gender": "MALE",
      "address": "JL. Pejuang No.1"
    },
    "medical_context": {
      "anamnesis": "Demam dua hari",
      "diagnoses": [
        {"description": "Influenza", "doctor_name": "dr. Budi"}
      ],
      "therapy_note": "Istirahat dan minum obat"
    },
    "medicines": [
      {
        "name": "Paracetamol",
        "quantity": 10,
        "unit": "TABLET",
        "dosage_instruction": "3 x 1",
        "unit_price": 1000,
        "total_price": 10000
      }
    ],
    "other_charges": [
      {
        "category": "CONSULTATION",
        "description": "Konsultasi dokter",
        "amount": 115000
      }
    ],
    "financial_recap": {
      "total_medicines": 10000,
      "total_other_charges": 115000,
      "grand_total": 125000
    }
  }
}
```

Notes:
- `images` paths must be relative to the claim folder.
- `patient_name_candidates` is optional.
- `summary.detected_documents` should preserve distinct packet documents in packet order.
- Ground truth should prefer the extraction prompt's document labels such as `Kwitansi`, `Resep`, `Salinan Resep`, `Nota Penjualan`, `Tanda Penerimaan Pembayaran`, and `Formulir Rawat Jalan`.
- The evaluator still normalizes legacy/internal labels such as `receipt`, `medical_form`, `doctor_note`, `lab_result`, and `radiology_result` for compatibility with older cached predictions.

## Generate a Template

Use the backend scan flow to generate a blank template plus a `_predicted` reference block.

```bash
python -m eval.generate_template --folder "path/to/claim_001" --candidates "Budi Saputra" "Siti Aminah"
```

Shortcut from `ta_backend/`:

```bash
python run_eval.py template -- --folder "path/to/claim_001" --candidates "Budi Saputra" "Siti Aminah"
```

Windows batch shortcut:

```bash
run_eval.bat template -- --folder "path/to/claim_001" --candidates "Budi Saputra" "Siti Aminah"
```

This writes `<folder>.json` into the claim folder by default.

Then:
- fill in the `expected` values manually
- remove the `_predicted` field before using the file for evaluation

## Run Evaluation

```bash
python -m eval.run_evaluation --folder "path/to/eval_examples"
```

Default behavior (`--mode auto`):
- load per-claim cache from `<dataset>/results`
- reuse cached `latest.json` when fingerprint matches
- otherwise perform a fresh scan and save logs/results

Modes:
- `--mode auto`: cache-or-scan (default)
- `--mode eval-only`: evaluate from cache only (no scan)
- `--mode scan-only`: scan and save only (no scoring)

Cache controls:
- `--results-dir "path/to/results"` to override cache root
- `--refresh` to force fresh scan and update cache
- `--allow-stale-cache` to evaluate stale cache when fingerprint changed

Shortcut from `ta_backend/`:

```bash
python run_eval.py evaluate -- --folder "path/to/eval_examples"
```

Windows batch shortcut:

```bash
run_eval.bat evaluate -- --folder "path/to/eval_examples"
```

Clear cached results:

```bash
python run_eval.py clear-cache --folder "path/to/eval_examples"
```

Clear one claim cache only:

```bash
python run_eval.py clear-cache --folder "path/to/eval_examples" --claim "claim_001"
```

Preview without deleting:

```bash
python run_eval.py clear-cache --folder "path/to/eval_examples" --dry-run
```

Optional output path:

```bash
python -m eval.run_evaluation --folder "path/to/eval_examples" --output "path/to/eval_results.json"
```

## Fit Self-Consistency Calibration

After collecting labeled claims and fresh scan results under `results/`, fit the runtime confidence calibrator:

```bash
python -m eval.self_consistency_calibration --folder "path/to/eval_examples"
```

Optional outputs:

```bash
python -m eval.self_consistency_calibration \
  --folder "path/to/eval_examples" \
  --results-dir "path/to/eval_examples/results" \
  --artifact-out "path/to/ai_ensemble_self_consistency_logistic_v1.json" \
  --report-out "path/to/self_consistency_calibration_report.json"
```

This:
- reads `field_confidence` from cached scan run logs
- builds claim-level calibration rows
- runs claim-level cross-validation
- writes the runtime JSON artifact consumed by `ai_ensemble.py`

## Output

The CLI prints a console summary and writes a JSON report using schema `medical_claim_extraction_eval_v2`.

Console output during evaluation:

```text
CLAIM                      KIE_F1    PREC     REC    ANLS     CER        SRC  ELAPSED  STATUS
------------------------------------------------------------------------------------------------------------------------------------
458                        0.5095  0.5909  0.5909  0.8420  0.1200      cache      0.0s  evaluated
461                        0.6925  0.7600  0.7600  0.9100  0.0600      cache      0.0s  evaluated
------------------------------------------------------------------------------------------------------------------------------------

Strict KIE Macro F1=0.6010  (2 claims evaluated)
Scan Sources: cache=2, fresh_scan=0

EVALUATION SUMMARY
================================================================================
Schema: medical_claim_extraction_eval_v2
Primary: strict_kie_micro_f1=0.6182  P=0.6800  R=0.5667
Claim Macro F1=0.6010  Section Macro F1=0.6235
Critical Exact=0.8750 (14/16)  Financial Exact=0.7500 (6/8)
Relaxed Text: ANLS=0.8810  CER=0.0900  n=18
Token Usage: avg_input=1200.00  avg_output=180.00  avg_total=1380.00  claims_with_usage=2  api_calls=6
Usability Diagnostics: avg_edit_burden=1.5000  pass_verify_only_rate=0.5000
```

JSON top-level structure:

```json
{
  "evaluated_at": "2026-05-21 17:00:00",
  "total_claims": 2,
  "evaluation_schema": "medical_claim_extraction_eval_v2",
  "primary_metric": "strict_kie_micro_f1",
  "aggregate_summary": {
    "schema_version": "medical_claim_extraction_eval_v2",
    "primary_metric": "strict_kie_micro_f1",
    "strict_kie_micro": {
      "precision": 0.68,
      "recall": 0.5667,
      "f1": 0.6182,
      "tp": 34,
      "fp": 16,
      "fn": 26
    },
    "strict_kie_macro_f1": 0.601,
    "strict_kie_section_macro_f1": 0.6235,
    "field_exact_accuracy": {
      "accuracy": 0.875,
      "correct": 14,
      "total": 16,
      "incorrect_fields": ["financial_recap.grand_total"]
    },
    "financial_exact_accuracy": {
      "accuracy": 0.75,
      "correct": 6,
      "total": 8,
      "incorrect_fields": ["financial_recap.grand_total"]
    },
    "relaxed_text": {
      "average_anls": 0.881,
      "average_cer": 0.09,
      "count": 18
    },
    "token_usage": {
      "claims_with_usage": 2,
      "api_call_count": 6,
      "successful_api_call_count": 6,
      "failed_api_call_count": 0,
      "total_input_tokens": 2400,
      "total_output_tokens": 360,
      "total_tokens": 2760,
      "average_input_tokens": 1200.0,
      "average_output_tokens": 180.0,
      "average_total_tokens": 1380.0,
      "average_input_tokens_per_call": 400.0,
      "average_output_tokens_per_call": 60.0,
      "average_total_tokens_per_call": 460.0,
      "average_latency_ms_per_call": 1200.0
    },
    "diagnostics": {
      "average_edit_burden_score": 1.5,
      "pass_verify_only_rate": 0.5
    },
    "per_section_micro": {},
    "per_section_macro_f1": {},
    "per_field_micro": {},
    "per_field_macro_f1": {}
  },
  "results": []
}
```

## Evaluation Metrics

Primary academic metric:

- `strict_kie_micro_f1`: micro-averaged precision/recall/F1 over extracted key-information units. Each applicable scalar field or matched entity attribute contributes TP/FP/FN counts. This follows common key information extraction evaluation practice, where exact entity/field correctness is measured with precision, recall, and F1.

Secondary academic metrics:

- `strict_kie_macro_f1`: average of each claim's strict KIE F1. This prevents a claim with many fields from completely dominating the dataset score.
- `strict_kie_section_macro_f1`: average section F1 across `summary`, `patient_info`, `medical_context`, `medicines`, `other_charges`, and `financial_recap`.
- `field_exact_accuracy`: exact-match accuracy over critical claim fields only.
- `financial_exact_accuracy`: exact-match accuracy over `summary.total_amount`, `financial_recap.total_medicines`, `financial_recap.total_other_charges`, and `financial_recap.grand_total`.
- `relaxed_text.average_anls`: Average Normalized Levenshtein Similarity for text fields such as facility name, address, anamnesis, therapy note, diagnosis text, medicine names, and charge descriptions.
- `relaxed_text.average_cer`: Character Error Rate for the same text fields. Lower is better; `0.0` means no character edits are needed.

Project diagnostics:

- `diagnostics.usability.edit_burden_score`: a weighted estimate of how much manual correction remains. It uses field-level, entity-level, relaxed-text, and financial-tolerance evidence so list items and their attributes are not over-counted.
- `diagnostics.usability.pass_verify_only`: `true` when patient identity passes, financial fields are exact or within the configured tolerance, and edit burden stays below the configured limit.
- `diagnostics.usability.blocking_fail_reasons`: hard reasons why the claim cannot be verify-only, currently patient-name mismatch and financial fields outside tolerance.
- `diagnostics.usability.advisory_fail_reasons`: soft review warnings, such as date, facility/provider, entity, context, or relaxed-text issues that are allowed when edit burden stays bounded.
- `diagnostics.usability.critical_fail_reasons`: backward-compatible alias for `blocking_fail_reasons`.
- `token_usage.average_input_tokens`: average prompt/input tokens per claim with recorded token usage.
- `token_usage.average_output_tokens`: average completion/output tokens per claim with recorded token usage.
- `token_usage.average_total_tokens`: average total tokens per claim with recorded token usage.
- `token_usage.average_*_per_call`: per-AI-call averages, useful when a claim uses multiple model runs or judges.
- `token_usage.average_latency_ms_per_call`: average AI API call latency for calls with recorded latency.

The diagnostics are intentionally separated from the academic metrics. They help explain workflow impact and operational cost, but they are not the main evaluation score.

## Scoring Rules

- Empty prediction and empty ground truth are treated as not applicable, so blank optional fields do not inflate the score.
- Scalar fields use exact, case-insensitive, normalized facility, or numeric exact matching depending on the field.
- `summary.detected_documents` is normalized to the evaluator document vocabulary, then matched as an ordered sequence with duplicates preserved.
- `diagnoses` are entities keyed by normalized diagnosis description; doctor name is scored as an attribute when the diagnosis entity matches.
- `medicines` are entities keyed by normalized medicine name; quantity, unit, dosage instruction, unit price, and total price are scored as attributes when the medicine entity matches.
- `other_charges` are entities keyed by normalized charge category plus exact amount; description is scored as an attribute when the charge entity matches.
- `financial_recap` uses numeric exact matching.
- Relaxed text metrics are reported as secondary academic evidence. They can rescue only the project-specific verify-only diagnostic; they do not change strict KIE, exact accuracy, ANLS, or CER scores.

## Methodology Basis

The evaluator is grounded in freely accessible information extraction and document-understanding evaluation practice:

- SROIE/ICDAR receipt information extraction uses precision, recall, and F1 for key information extraction: https://arxiv.org/abs/2103.10213
- LayoutLMv2 reports entity-level precision, recall, and F1 for FUNSD form understanding and SROIE receipt extraction: https://aclanthology.org/2021.acl-long.201.pdf
- DocVQA uses Average Normalized Levenshtein Similarity (ANLS) for document question-answering text answers: https://openaccess.thecvf.com/content/WACV2021/papers/Mathew_DocVQA_A_Dataset_for_VQA_on_Document_Images_WACV_2021_paper.pdf
- OCR-D documents character error rate and related OCR text evaluation metrics: https://ocr-d.de/en/spec/ocrd_eval.html
- The i2b2 medication extraction challenge used precision, recall, and F-measure for medication information extraction: https://faculty.washington.edu/fxia/mpapers/2009/uzuner2009_i2b2_eval.pdf
- Calibration of confidence estimates can be evaluated separately from extraction correctness; a standard reference is Guo et al., "On Calibration of Modern Neural Networks": https://proceedings.mlr.press/v70/guo17a.html

## Current Limits

- Evaluation is extraction-only. Medicine relevance is not scored here.
- The runner processes claims sequentially.
- The strict entity keys require correct canonical medicine, diagnosis, or charge anchors before scoring attributes.
- `edit_burden_score` and `pass_verify_only` are project-specific diagnostics, not paper-standard metrics.

## Result Cache and Run Logs

Each claim stores reusable outputs and append-only run logs:

```text
results/
  claim_001/
    latest.json
    runs/
      20260423T103155Z__claim-001__scan-1234.json
      20260424T081101Z__claim-001__scan-1301.json
```

- `latest.json` is the snapshot used for cache reuse during evaluation.
- `runs/*.json` is append-only history for every fresh scan attempt.
- Each run record stores scan metadata and `ai_model_runs` payload so per-model/per-run logs are retained.

If you want all fresh inferences again, delete the `results/` folder.

## Detailed Breakdown and Diff

Show detailed metrics for a specific claim:

```bash
# Using run_eval.bat
run_eval.bat evaluate -- --folder "path/to/eval_examples" --detail 464

# Using python module
python -m eval.run_evaluation --folder "path/to/eval_examples" --detail 464
```

Show visual diff between predicted and ground truth:

```bash
# Using run_eval.bat
run_eval.bat evaluate -- --folder "path/to/eval_examples" --detail 464 --diff

# Using python module
python -m eval.run_evaluation --folder "path/to/eval_examples" --detail 464 --diff
```

Output diff to HTML file for better readability:

```bash
# Console + HTML file
run_eval.bat evaluate -- --folder "path/to/eval_examples" --detail 464 --diff --diff-out "claim_464_diff.html"
python -m eval.run_evaluation --folder "path/to/eval_examples" --detail 464 --diff --diff-out "claim_464_diff.html"
```

### Output: --detail

```text
CLAIM: 464

CLAIM SUMMARY:
  schema_version: medical_claim_extraction_eval_v2
  primary_metric: strict_kie_micro_f1
  strict_kie_micro: f1=0.7143 precision=0.7500 recall=0.6818
  strict_kie_section_macro_f1: 0.7024
  critical_exact_accuracy: 0.8750 (7/8)
  financial_exact_accuracy: 0.7500 (3/4)
  relaxed_text: ANLS=0.8420 CER=0.1200 n=9
  edit_burden_score: 2.0000
  pass_verify_only: False
  blocking_fail_reasons:
    - financial_recap.grand_total outside verify-only tolerance
  advisory_fail_reasons:
    - medical_context KIE F1 below relaxed verify-only threshold

SECTION BREAKDOWN:
SECTION                   F1     PREC      REC       TP     FP     FN
--------------------------------------------------------------------------------
summary               0.7143   0.7500   0.6818        5      2      3
patient_info          1.0000   1.0000   1.0000        4      0      0
medical_context       0.5455   0.6000   0.5000        3      2      3
medicines             0.6667   0.6667   0.6667        4      2      2
other_charges         0.5000   0.5000   0.5000        1      1      1
financial_recap       0.6667   0.6667   0.6667        2      1      1

PER-FIELD BREAKDOWN:
  summary.service_date                     F1=1.0000 P=1.0000 R=1.0000
  financial_recap.grand_total              F1=0.0000 P=0.0000 R=0.0000
  medical_context.anamnesis                F1=0.0000 P=0.0000 R=0.0000  ANLS=0.8800  CER=0.1200

MEDICINE MATCHES:
  ✓ GT: Paracetamol → PRED: Paracetamol (sim=1.00, strict=pass)
  ✓ GT: Amoxicillin → PRED: Amoxicillin (sim=1.00, strict=pass)
  [MISSING] GT: Vitamin C 20 TABLET
  [EXTRA] PRED: Ibuprofen 10 TABLET
```

### Output: --diff

```text
VISUAL DIFF: predicted vs ground_truth
================================================================================
SECTION: summary
  FIELD                    GROUND TRUTH       PREDICTED        STATUS
-----------------------------------------------------------------
  service_date            2024-08-10         2024-08-10         ✓
  total_amount            125000            130000            ✗
  medical_facility_name   RS Sehat          RS Sehat Sentos   F1=0.50

SECTION: medical_context
  FIELD                    GROUND TRUTH            PREDICTED          STATUS
---------------------------------------------------------------------------
  anamnesis              Pasien mengeluh...       Demam disertai...  CER=0.12
  therapy_note           Istirahat 3 hari        Istirahat dan...  CER=0.55
```

### Color Legend

- `✓` (green) — strict exact/entity match
- `F1=X.XX` (yellow) — partial ordered/entity score shown for an applicable field
- `CER=X.XX` (yellow) — text edit distance shown for a text field
- `✗` (red) — strict mismatch
- `[MISSING]` (red) — present in ground truth but not in prediction
- `[EXTRA]` (red) — present in prediction but not in ground truth
