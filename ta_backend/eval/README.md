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

The CLI prints a console summary and writes a JSON report.

Console output during evaluation:

```
CLAIM                         F1      PREC     REC     CER        SRC  ELAPSED  STATUS
------------------------------------------------------------------------------------------------------------------------
458                        0.5095  0.5909  0.5909  0.0000      cache      0.0s  evaluated
461                        0.6925  0.7600  0.7600  0.0000      cache      0.0s  evaluated
462                        0.5429  0.8333  0.7895  0.0000      cache      0.0s  evaluated
464                        0.9143  0.8571  0.9000  0.5000      cache      0.0s  evaluated
468                        0.5224  0.8333  0.8824  0.0000      cache      0.0s  evaluated
------------------------------------------------------------------------------------------------------------------------

Overall Macro: F1=0.6363  P=0.7732  R=0.7657  (5 claims evaluated)

EVALUATION SUMMARY
================================================================================
Overall Macro:  F1=0.6363  P=0.7732  R=0.7657

SECTION BREAKDOWN (Macro Avg):
SECTION              F1      PREC    REC
-------------------------------------------------
summary            0.71    0.75    0.68
patient_info       1.00    1.00    1.00
medical_context    0.54    0.58    0.51
medicines          0.60    0.65    0.56
other_charges      0.50    0.55    0.46
financial_recap    0.33    0.40    0.28

FIELDS NEEDING ATTENTION (lowest F1):
  1. financial_recap.grand_total       F1=0.33
  2. medical_context.diagnoses       F1=0.42
  3. medicines                    F1=0.60

Scan Sources: cache=5, fresh_scan=0
```

### Color Legend (Console)

- `✓` (green) — exact match / perfect similarity (sim ≥ 0.8)
- `~` (yellow) — partial/similar (sim 0.5-0.79, CER < 0.5)
- `✗` (red) — mismatch / low similarity (sim < 0.5, F1 < 0.5)
- `[MISSING]` (red) — present in GT but not in prediction
- `[EXTRA]` (red) — present in prediction but not in GT
- `(sim=X.XX)` — similarity score shown for list items

JSON top-level structure:

```json
{
  "evaluated_at": "2026-04-22 17:00:00",
  "total_claims": 2,
  "aggregate_summary": {
    "overall_macro_f1": 0.91,
    "overall_macro_precision": 0.93,
    "overall_macro_recall": 0.89,
    "successful_claims": 2,
    "failed_claims": 0,
    "per_section_macro_f1": {
      "summary": 0.95,
      "patient_info": 1.0,
      "medical_context": 0.86,
      "medicines": 0.82,
      "other_charges": 0.94,
      "financial_recap": 0.95
    },
    "per_section_macro_precision": {
      "summary": 0.96,
      "patient_info": 1.0,
      "medical_context": 0.88,
      "medicines": 0.84,
      "other_charges": 0.95,
      "financial_recap": 0.96
    },
    "per_section_macro_recall": {
      "summary": 0.94,
      "patient_info": 1.0,
      "medical_context": 0.84,
      "medicines": 0.80,
      "other_charges": 0.93,
      "financial_recap": 0.94
    },
    "per_field_macro_f1": {
      "summary.service_date": 1.0,
      "patient_info.name": 1.0,
      "medical_context.diagnoses": 0.75,
      "medicines": 0.82,
      "other_charges": 0.94,
      "financial_recap.total_medicines": 1.0,
      "financial_recap.total_other_charges": 0.92,
      "financial_recap.grand_total": 0.95
    }
  },
  "results": []
}
```

## Scoring Rules

- Scalar fields: exact match or numeric exact match → precision, recall, F1
- `summary.detected_documents`: ordered packet-inventory matching after evaluator normalization; duplicates and order both affect score
- `anamnesis` and `therapy_note`: token-based F1 + character error rate (CER)
- `diagnoses`: token similarity plus doctor-name support
- `medicines`: Hungarian matching using medicine-name similarity and quantity proximity
- `other_charges`: prompt-contract category normalization plus description similarity
- `financial_recap`: numeric exact match on `total_medicines`, `total_other_charges`, and `grand_total`

**Note:** CER (Character Error Rate) measures character-level edit distance. Lower is better (0 = perfect, 1 = completely wrong).

## Current Limits

- Evaluation is extraction-only. Medicine relevance is not scored.
- The runner processes claims sequentially.
- Exact score semantics are intentionally simple and deterministic for backend regression tracking.

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

```
CLAIM: 464
================================================================================
SECTION BREAKDOWN:
SECTION              F1      PREC    REC     TP   FP   FN
-------------------------------------------------
summary            0.71    0.75    0.68     5    2    3
patient_info       1.00    1.00    1.00     4    0    0
medical_context    0.54    0.58    0.51     3    4    3
medicines           0.60    0.65    0.56     3    2    3
other_charges      0.50    0.55    0.46     2    2    3
financial_recap     0.33    0.40    0.28     1    2    3

PER-FIELD BREAKDOWN:
  summary.service_date              F1=1.0000
  summary.total_amount              F1=0.0000
  medical_context.anamnesis          F1=0.4000  CER=0.1200
  medical_context.therapy_note       F1=0.2200  CER=0.5500

MEDICINE MATCHES:
  ✓ GT: Paracetamol 10 TABLET → PRED: Paracetamol 10 TABLET (sim=1.00)
  ✓ GT: Amoxicillin 15 CAPSUL → PRED: Amoxicillin 15 CAPSUL (sim=1.00)
  [MISSING] GT: Vitamin C 20 TABLET
  [EXTRA] PRED: Ibuprofen 10 TABLET
```

### Output: --diff

```
VISUAL DIFF: predicted vs ground_truth
================================================================================
SECTION: summary
  FIELD                    GROUND TRUTH       PREDICTED        STATUS
-----------------------------------------------------------------
  service_date            2024-08-10         2024-08-10         ✓
  total_amount            125000            130000            ~ (+5000)
  medical_facility         RS Sehat          RS Sehat Sentos  ~ (partial)

SECTION: medical_context
  FIELD                    GROUND TRUTH            PREDICTED          STATUS
---------------------------------------------------------------------------
  anamnesis              Pasien mengeluh...       Demam disertai...  CER=0.12
  therapy_note           Istirahat 3 hari        Istirahat dan...  CER=0.55
```

### Color Legend

- `✓` (green) — exact match / perfect similarity
- `~` (yellow) — partial match / similar (numeric diff < 10%, text similarity 0.5-0.99, CER < 0.5)
- `✗` (red) — mismatch / low similarity (F1 < 0.5)
- `[MISSING]` (red) — present in ground truth but not in prediction
- `[EXTRA]` (red) — present in prediction but not in ground truth
