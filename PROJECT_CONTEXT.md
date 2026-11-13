# Project Context For AI Agents

Last reviewed from the working tree on 2026-07-07.

This file is a project-level orientation document. Its purpose is to let a new AI agent understand the system without first reading the whole repository. It is not a substitute for checking the current code before making edits, but it captures the verified architecture, core flows, data contracts, and implementation details found in this repo.

## 1. Project Purpose

This project is a prototype medical insurance claim automation system.

The originating proposal is:

`D:\Documents\S1\Semester7\TA\ProposalTugasAkhir_222117024_v1-2_commentless.pdf`

The proposal title is:

`Sistem Otomatisasi Klaim Asuransi Medis Berbasis Pengenalan Dokumen Cerdas Dengan Memanfaatkan Vision Language Model (VLM)`

The proposed product is a mobile claim-submission app with a Smart Auto-fill feature. A user uploads photos of medical claim documents, such as receipts, prescriptions, and outpatient forms. The backend preprocesses images, validates the documents, sends the images to a VLM, extracts structured claim data, and returns JSON that can autofill the claim form. The user then verifies or edits the extracted data before submitting the final claim.

The current implementation follows that idea and adds several more advanced features:

- FastAPI backend with SQLAlchemy persistence.
- Expo React Native frontend.
- JWT authentication and user/admin roles.
- Family member management for patient-name matching.
- Image preprocessing with Pillow/OpenCV.
- VLM document validation.
- VLM medical claim extraction.
- Multi-model/multi-run ensemble extraction.
- Field-level adjudication and packet-level financial adjudication.
- Financial reconciliation logic.
- Medicine ordering and ambiguity postprocessing.
- Medicine relevance review on final claim submission.
- Scan-job progress over Server-Sent Events.
- Admin scan-log and claim inspection.
- AI model activation/priority management.
- Backend evaluation tooling for extraction accuracy.

## 2. Repository Layout

Top-level structure:

```text
project_ta/
  .vscode/
    mcp.json
  spec/
  ta_backend/
  ta_frontend/
  PROJECT_CONTEXT.md
  PROFILE_AND_INHEALTH_EXPORT_CHANGES.md
```

Important backend areas:

```text
ta_backend/
  app/
    api/routes/              FastAPI route handlers
    core/                    config and AI pipeline logging
    db/                      SQLAlchemy engine/session/init
    models/                  SQLAlchemy ORM models
    prompts/ai/              YAML prompt templates and output contracts
    repositories/            database access layer
    schemas/                 Pydantic API schemas
    services/                AI pipeline, preprocessing, fusion, relevance
    artifacts/               runtime calibration artifact
  eval/                      extraction evaluation tooling
  tests/                     pytest test suite
  app.db                     local SQLite database
  main.py                    imports app.main.app
  run.py                     backend runner
  run_backend.bat            Windows backend shortcut
  run_eval.py                evaluation helper
  requirements.txt
  app/templates/claim_forms/ static claim-form PDF templates
```

Important frontend areas:

```text
ta_frontend/
  app/                       Expo Router routes
  components/                shared UI components
  constants/                 app theme
  contexts/                  auth and theme contexts
  features/
    admin/                   admin screens/components
    auth/                    login/register UI
    claims/                  user claim screens/components
    profile/                 profile screen
  services/api/              API clients and shared TS types
  assets/images/
  package.json
```

## 3. Runtime Architecture

The system is client-server:

```text
React Native / Expo mobile app
  -> FastAPI HTTP API under /api
    -> SQLAlchemy database
    -> image preprocessing
    -> LangChain chat model adapters
      -> external VLM provider APIs
```

Backend application entrypoint:

- `ta_backend/main.py` imports `app.main.app`.
- `ta_backend/app/main.py` creates the FastAPI app.
- All main routers are mounted under `/api`.
- On lifespan startup, `initialize_database()` runs.

The app title is currently `"Python Backend"` and version `"0.1.0"`.

## 4. Configuration And Environment

Backend settings live in `ta_backend/app/core/config.py`.

The backend loads `.env` from `ta_backend/.env` by resolving `ROOT_DIR` as `ta_backend`.

Settings fields:

- `APP_NAME`, default `python-backend`
- `APP_ENV`, default `dev`
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `OPENROUTER_API_KEY`
- `OLLAMA_API_KEY`
- `GROQ_API_KEY`
- `MISTRAL_API_KEY`
- `DEEPSEEK_API_KEY`
- `COHERE_API_KEY`
- `CEREBRAS_API_KEY`
- `JWT_SECRET`, default `fallback_dev_secret_change_this`
- `ACCESS_TOKEN_EXPIRE_MINUTES`, default `10080`
- `DATABASE_URL`, default `sqlite:///./app.db`

Database URL normalization:

- `postgresql://...` is converted to `postgresql+psycopg://...`.
- SQLite uses `check_same_thread=False`.

The proposal says PostgreSQL, but this working implementation defaults to SQLite via `app.db`. PostgreSQL-compatible configuration exists through `DATABASE_URL`.

Frontend API base URL:

- Defined in `ta_frontend/services/api/config.ts`.
- Reads `Constants.expoConfig?.extra?.apiBaseUrl`.
- Android fallback: `http://10.0.2.2:8000/api`.
- Other platforms fallback: `http://127.0.0.1:8000/api`.

## 5. Database Initialization

`ta_backend/app/db/init.py` performs startup setup:

- `Base.metadata.create_all(bind=get_engine())`
- schema compatibility/migration helpers
- default admin seeding
- default AI model catalog seeding

Seeded default admin:

```text
email: admin@mail.com
password: admin123
role: admin
```

Seeded AI model catalog includes:

- `google_genai` / `gemini-3.1-flash-lite`, active by default, priority 10
- `openrouter` / `meta-llama/llama-3.2-11b-vision-instruct`, inactive
- `ollama` / `llama3.2-vision`, inactive
- `groq` / `llama-3.3-70b-versatile`, inactive
- `mistralai` / `mistral-small-latest`, inactive and not scan-compatible
- `deepseek` / `deepseek-chat`, inactive and not scan-compatible
- `cohere` / `command-r-plus`, inactive and not scan-compatible
- `cerebras` / `llama-3.3-70b`, inactive and not scan-compatible

Startup schema helpers add or migrate columns/tables for:

- `ai_models.priority`
- `scan_logs` legacy fields to current schema
- `scan_log_ai_runs`
- `ai_api_call_logs`
- `claims.medicines`
- `claims.medicines_payload`
- `claims.requires_additional_review`
- `claims.review_note`
- claim patient/profile snapshot columns used by the InHealth PDF export
- `claim_medicine_knowledge`

## 6. Data Model

Models are in `ta_backend/app/models/`.

### users

Defined in `models/user.py`.

Fields:

- `id`
- `full_name`
- `email`, unique and indexed
- `password_hash`
- `role`, constrained to `user` or `admin`
- claim-profile fields: `nip`, `company_name`, `insurance_card_number`, `phone_number`, `mobile_number`
- bank fields: `bank_account_number`, `bank_account_holder_name`, `bank_name`
- insurance-card image fields: filename, MIME type, byte size, SHA-256 hash, binary content
- `created_at`

Passwords use PBKDF2-HMAC-SHA256 with per-password random salt and 120,000 iterations.

The registered email is reused as the user's profile email. Profile completeness is returned by the API. The frontend is the primary New Claim gate, and backend selected-patient claim creation/export also validates required profile data when it needs profile snapshots.

### family_members

Defined in `models/family_member.py`.

Fields:

- `id`
- `user_id`
- `name`
- `relation`
- `insurance_card_number`
- contact fields: `phone_number`, `mobile_number`, `email`
- insurance-card image fields: filename, MIME type, byte size, SHA-256 hash, binary content

Used to create patient-name candidates for document validation and extraction.
Family-member profile completeness is also returned by the API so the frontend can block New Claim access until all saved family profiles are complete.

### ai_models

Defined in `models/ai_model.py`.

Fields:

- `id`
- `display_name`
- `model_name`
- `provider`
- `priority`
- `is_active`

There is a uniqueness constraint on `(provider, model_name)`.

Active scan-compatible models are used by the AI scan pipeline in priority order.

### scan_logs

Defined in `models/scan_log.py`.

Fields:

- `id`
- `user_id`
- `status`
- `ai_execution_mode`
- `primary_model_provider`
- `primary_model_name`
- `validation_payload`
- `extraction_payload`
- `extraction_meta_payload`
- `progress_payload`
- `error_payload`
- `created_at`
- `updated_at`

Relationships:

- `user`
- `files`
- `ai_runs`
- `ai_api_call_logs`
- `linked_claims`

Common statuses include:

- `pending`
- `running`
- `succeeded`
- `cancelled`
- `preprocess_failed`
- `validation_failed`
- `patient_name_mismatch`
- `extraction_failed`
- `scan_failed`

### scan_log_files

Defined in `models/scan_log.py`.

Stores both original and processed images:

- original filename, MIME type, byte size, SHA-256 hash, binary content
- processed filename, MIME type, byte size, SHA-256 hash, binary content

### scan_log_ai_runs

Defined in `models/scan_log.py`.

Stores per-producer extraction-run traces:

- `producer_id`
- `provider`
- `model_name`
- `producer_rank`
- `run_index`
- `temperature`
- `success`
- `error_message`
- `payload`
- `created_at`

### ai_api_call_logs

Defined in `models/scan_log.py`.

Stores detailed API-call audit data for validation, extraction, judging, medicine relevance, and test invocations:

- scan/user context
- provider/model/call kind
- producer/run metadata
- success/error
- token usage
- latency
- sanitized prompt payload
- raw response text
- response content payload
- parsed response payload
- validation error
- result context payload

Images in prompt logs are redacted as data URLs rather than storing full base64 prompt payloads.

### claims

Defined in `models/claim.py`.

Fields:

- `id`
- `user_id`
- `user_email`
- `nama_pasien`
- selected patient snapshot fields: `selected_patient_type`, `selected_family_member_id`, `patient_relation`
- patient export snapshot fields: `patient_card_number`, `patient_phone_number`, `patient_mobile_number`, `patient_email`
- user export snapshot fields: `user_full_name_snapshot`, `user_nip_snapshot`, `user_company_name_snapshot`, `user_bank_account_number_snapshot`, `user_bank_account_holder_name_snapshot`, `user_bank_name_snapshot`
- `tanggal_pelayanan`
- `fasilitas_kesehatan`
- `jenis_manfaat`
- `nominal_klaim`
- `keterangan_penyakit`
- `kronologi`
- `medicines`
- `medicines_payload`
- `requires_additional_review`
- `review_note`
- `status`
- `source_scan_log_id`
- `created_at`

`source_scan_log_id` links a final claim to the scan result used to autofill it.

The snapshot fields are internal storage for export workflows and do not expand the user-facing `ClaimResponse` or `ClaimDetailResponse` schema.

### claim_files

Defined in `models/claim.py`.

Stores submitted claim file binary content:

- `filename`
- `mime_type`
- `byte_size`
- `sha256_hash`
- `content`

### claim_medicine_knowledge

Defined in `models/claim_medicine_knowledge.py`.

Stores learned medicine relevance observations by diagnosis/anamnesis context and normalized medicine name:

- `context_signature`
- `medicine_name_normalized`
- normalized diagnosis/anamnesis text
- diagnosis/anamnesis term JSON
- support and relevance counts
- average confidence
- last reason/source
- first/last claim IDs
- timestamps

Used by medicine relevance checking and medicine scan prioritization.

## 7. API Surface

All backend routes are mounted under `/api`.

### Auth

File: `ta_backend/app/api/routes/auth.py`

Routes:

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `PUT /api/auth/me/profile`
- `POST /api/auth/me/insurance-card`

Register creates a normal `user`. Login returns a JWT bearer token. `/me` reads the bearer token and returns the persisted user, including profile-completeness metadata.

Registration also returns `access_token` and `token_type`, so the frontend can auto-login immediately after successful registration.

JWT payload contains:

- `sub`
- `email`
- `role`
- `exp`

Global exception handlers in `app/main.py` normalize validation and HTTP errors into:

```json
{
  "message": "Input validation failed.",
  "errors": [
    {
      "field": "field_name",
      "message": "Human-readable message."
    }
  ]
}
```

### Family Members

File: `ta_backend/app/api/routes/family_members.py`

Routes:

- `GET /api/family-members`
- `POST /api/family-members`
- `PUT /api/family-members/{family_member_id}`
- `POST /api/family-members/{family_member_id}/insurance-card`
- `DELETE /api/family-members/{family_member_id}`

All require user authentication. Access is scoped by current user ID.

### AI Scan

File: `ta_backend/app/api/routes/ai.py`

Routes:

- `POST /api/ai/scan-documents/jobs`
- `GET /api/ai/scan-documents/jobs/{scan_log_id}`
- `GET /api/ai/scan-documents/jobs/{scan_log_id}/events`
- `POST /api/ai/scan-documents/jobs/{scan_log_id}/cancel`
- `POST /api/ai/scan-documents`

The `/jobs` path is the frontend's main path. It returns immediately with a `scan_log_id` and starts a FastAPI background task. The frontend then follows progress via SSE and/or polling.

The direct `POST /api/ai/scan-documents` route performs the full scan synchronously and returns the result.

### Claims

File: `ta_backend/app/api/routes/claims.py`

Routes:

- `POST /api/claims`
- `GET /api/claims`
- `GET /api/claims/{claim_id}/inhealth-form`
- `GET /api/claims/{claim_id}`

Create claim accepts multipart form data. It may link to a previous scan via `source_scan_log_id`.

Create claim can also receive:

- `selected_patient_type`, either `self` or `family`
- `family_member_id`, required when `selected_patient_type` is `family`

When selected patient identity is provided, backend resolves the user or family-member profile, stores internal snapshot data needed for PDF export, and appends the selected patient's stored insurance-card image as an extra `claim_files` row. This patient-card file is stored unprocessed and is not part of scan-log hash matching or medical document validation.

When linked to a scan:

- The source scan must belong to the user.
- Uploaded file hashes must match the scan-log files.
- Successful scan validation may be reused.
- If scan validation lacks usable patient-name evidence, validation is rerun.
- Selected patient name is checked against stored or rerun validation.

When not linked to a scan:

- The uploaded claim files are preprocessed and validated with `DocumentValidationProcessor`.
- Expected patient name is the selected `nama_pasien`.

Claim creation also evaluates medicine relevance and may set:

- `requires_additional_review`
- `review_note`

`GET /api/claims/{claim_id}/inhealth-form` checks ownership, fills the static InHealth PDF template, and returns `application/pdf` with a download filename. The PDF template lives at `ta_backend/app/templates/claim_forms/formulir_klaim_inhealth-1.pdf`; the implementation uses `reportlab` to create an overlay and `pypdf` to merge it onto the static template.

### Admin AI Models

File: `ta_backend/app/api/routes/admin_ai_models.py`

Routes:

- `GET /api/admin/ai-models`
- `PUT /api/admin/ai-models/{model_id}`

Admin only. Allows viewing and updating model activation and priority.

Providers that are not scan-compatible cannot be activated for the scan pipeline.

### Admin Claims

File: `ta_backend/app/api/routes/admin_claims.py`

Routes:

- `GET /api/admin/claims`
- `GET /api/admin/claims/summary`
- `GET /api/admin/claims/{claim_id}`

Admin only. Supports search, linked/manual filtering, sorting, limits, summaries, and detail with file data URLs.

### Admin Scan Logs

File: `ta_backend/app/api/routes/admin_scan_logs.py`

Routes:

- `GET /api/admin/scan-logs`
- `GET /api/admin/scan-logs/summary`
- `GET /api/admin/scan-logs/{scan_log_id}`

Admin only. Detail returns:

- validation payload
- extraction payload
- extraction meta payload
- error payload
- original and processed image data URLs
- AI extraction run payloads
- AI API call logs

### Test Routes

File: `ta_backend/app/api/routes/test.py`

Routes:

- `GET /api/test/health`
- `POST /api/test/echo`
- `POST /api/test/greet`
- `POST /api/test/invoke-model`

The model invocation route is used for testing provider/model calls and has extensive tests around text-only, image support, missing credentials, provider runtime errors, and response parsing.

## 8. AI Provider Model System

Provider metadata lives in `ta_backend/app/services/ai_provider_catalog.py`.

Registered providers:

- `google_genai`, class `langchain_google_genai.ChatGoogleGenerativeAI`, image-compatible.
- `openrouter`, class `langchain_openrouter.ChatOpenRouter`, image-compatible.
- `ollama`, class `langchain_ollama.ChatOllama`, image-compatible.
- `groq`, class `langchain_groq.ChatGroq`, image-compatible.
- `mistralai`, class `langchain_mistralai.ChatMistralAI`, not scan-compatible.
- `deepseek`, class `langchain_deepseek.ChatDeepSeek`, not scan-compatible.
- `cohere`, class `langchain_cohere.ChatCohere`, not scan-compatible.
- `cerebras`, class `langchain_cerebras.ChatCerebras`, not scan-compatible.

`BaseDocumentAIProcessor` creates model instances dynamically from the active database rows. It requires the matching credential environment variable for the provider.

Provider-specific notes:

- Google uses `google_api_key`.
- OpenRouter uses `api_key` and explicitly disables default attribution fields due to SDK compatibility.
- Ollama is configured against `https://ollama.com` with bearer auth headers.
- The model request timeout is 180 seconds, with an additional asyncio timeout wrapper.
- There is a Google async compatibility fallback for an installed-provider mismatch involving `generate_content()` and `max_retries`.

## 9. Prompt Files

AI prompt YAML files live in `ta_backend/app/prompts/ai/`.

Prompt file format:

- `system_template`: rendered into the `SystemMessage`; defines the model role, JSON-only behavior, and schema compliance requirements.
- `user_template`: rendered into the first text block of the `HumanMessage`; contains task instructions, packet context, rendered runtime variables, field authority rules, and anti-inference rules.
- `output_contract`: a YAML mapping rendered into `{output_contract_json}` and used as the expected response shape.
- Uploaded images are appended after the rendered user prompt as `image_url` blocks in the same `HumanMessage`.
- Runtime variables are rendered with LangChain `PromptTemplate`, for example patient-name candidates, validation packet hints, filenames, ambiguous medicines, and candidate ledgers.

### medical_document_validation.yaml

Used by `DocumentValidationProcessor`.

Purpose:

- Detect how many documents are visible per image.
- Classify each document.
- Validate readability.
- Check facility identity.
- Detect stamp/signature.
- Assess packet-level patient-name match.

Output schema:

- `packet_patient_name`
- `patient_name_match_status`
- `patient_name_match_reason`
- `images[]`
  - `image_index`
  - `filename`
  - `document_count_detected`
  - `documents[]`
    - `document_index`
    - `document_type`
    - `is_medical_document`
    - `readability_ok`
    - `facility_identity_visible`
    - `stamp_detected`
    - `signature_detected`
    - `decision`
    - `reasons`

Document types:

- `receipt`
- `medical_form`
- `doctor_note`
- `prescription`
- `lab_result`
- `radiology_result`
- `non_medical`
- `unknown`

Validation rules in code:

- Receipts can be valid without stamp/signature if readable and facility identity is visible.
- `medical_form`, `doctor_note`, and `prescription` require visible stamp or doctor signature when standalone.
- `lab_result` and `radiology_result` do not strictly require stamp/signature.
- Rotated-but-readable documents are accepted.
- Collage ambiguity can become soft review instead of hard rejection.
- Patient mismatch can block the scan or claim submission.

### medical_document_scan.yaml

Used by `MedicalDocumentProcessor`.

Purpose:

- Analyze all uploaded images as one claim packet.
- Extract structured claim JSON.
- Treat uploaded images as Indonesian outpatient medical insurance claim evidence such as claim forms, doctor notes, prescriptions, pharmacy receipts, lab/radiology results, billing summaries, and payment receipts.
- Read handwritten doctor text as possible medicine, dosage, therapy, diagnosis, anamnesis, or follow-up evidence when it appears near prescription or clinical-note cues.
- Follow field authority rules.
- Use validation hints for document inventory/order, not for inventing text.
- Use patient-name candidates only to disambiguate visible names.
- Treat bills, invoices, payment receipts, sales notes, duplicate views, and billing summaries for the same expense as evidence for one payment, not separate charges.
- Keep pharmacy/prescription spending in the medicine ledger instead of duplicating it in `other_charges`.
- Preserve original language.
- Use ISO dates and numeric monetary fields.

Output schema:

```json
{
  "summary": {
    "detected_documents": [],
    "medical_facility_name": null,
    "service_date": null,
    "transaction_date": null,
    "currency": null,
    "total_amount": null
  },
  "patient_info": {
    "name": null,
    "date_of_birth": null,
    "age": null,
    "gender": null,
    "address": null
  },
  "medical_context": {
    "anamnesis": null,
    "diagnoses": [],
    "therapy_note": null
  },
  "medicines": [],
  "other_charges": [],
  "financial_recap": {
    "total_medicines": null,
    "total_other_charges": null,
    "grand_total": null
  }
}
```

Standard document labels requested by prompt:

- `Kwitansi`
- `Resep`
- `Salinan Resep`
- `Nota Penjualan`
- `Tanda Penerimaan Pembayaran`
- `Formulir Rawat Jalan`

Standard other-charge categories requested by prompt:

- `CONSULTATION`
- `ADMINISTRATION`
- `SERVICE`
- `LABORATORY`
- `ROOM`
- `OTHER`

### medical_document_field_judge.yaml

Used when ensemble fusion has unresolved field-level candidate conflicts.

The judge receives:

- field path
- candidate values
- support count
- producer count
- original images

It returns:

- `selected_candidate_index`
- `reason`

It must choose only among candidates or return null.

### financial_packet_judge.yaml

Used when financial reconciliation finds close ledger candidates.

The judge receives serialized candidate ledgers and original images. It is packet-level, not field-level; normal financial scalar conflicts should be resolved by reconciliation first.

It returns:

- `selected_candidate_index`
- `reason`

It is instructed to choose one provided candidate unless the packet is genuinely unreadable, prefer visible numeric receipt amounts, and use terbilang/amount-in-words as consistency evidence.

### medical_claim_medicine_image_interpretation.yaml

Used by scan medicine prioritization for ambiguous medicine names/order.

Input includes:

- images
- diagnosis text
- anamnesis text
- ambiguous medicine candidates
- ranked group suggestions
- recent database medicine hints

Output contains batch judgements with interpreted name, confidence, relevance, and match flags.

### medical_claim_medicine_relevance_judge.yaml

Used on final claim submission to judge whether a medicine is relevant to diagnosis/anamnesis when historical evidence is not strong enough.

## 10. Image Preprocessing

Implemented in `ta_backend/app/services/image_preprocessing.py`.

Steps:

1. Open image bytes with Pillow.
2. Apply EXIF transpose/autorotate.
3. Resize if longest side is above 1024 pixels.
4. Convert to RGB/RGBA.
5. Convert RGB to LAB using OpenCV.
6. Apply CLAHE on L channel:
   - clip limit 2.0
   - tile grid 8x8
7. Convert back to RGB.
8. Apply light unsharp masking:
   - sigma 1.0
   - amount 0.15
9. Preserve alpha and encode PNG if alpha exists.
10. Otherwise encode JPEG at quality 85.

If image decoding fails, backend raises 422 with detail:

`Uploaded file is not a readable image.`

## 11. AI Scan Pipeline In Detail

The main orchestration function is `execute_scan_documents_pipeline()` in `ta_backend/app/api/routes/ai.py`.

### Step 1: Input buffering

Uploaded files are read into `BufferedUpload` objects:

- filename
- MIME type
- raw content
- byte size
- SHA-256 hash

### Step 2: Scan log creation

For async jobs, `POST /api/ai/scan-documents/jobs`:

- stores original uploads in `scan_log_files`
- creates a `scan_logs` row with queued progress
- schedules `run_scan_documents_job(scan_log_id, user_id)`
- returns `scan_log_id` and initial progress

For sync scans, `POST /api/ai/scan-documents` creates the scan log and runs the pipeline before returning.

### Step 3: Preprocessing

Each original upload is preprocessed.

Processed bytes and metadata are stored back on `scan_log_files`.

The AI input images are `DocumentImageInput` objects:

- processed filename
- processed MIME type
- base64 data

### Step 4: Patient candidate construction

The backend reads:

- current user full name
- current user's family member names

It deduplicates and preserves order. These names are used for validation and extraction.

### Step 5: Document validation

`DocumentValidationProcessor.process()`:

- uses only the first active scan-compatible model
- sends validation prompt and images
- parses JSON
- validates against Pydantic `DocumentValidationResponse`
- applies deterministic validation rules

Hard validation failures stop the scan before extraction.

Patient mismatch stops the scan with status `patient_name_mismatch`.

Soft review conditions are stored in `validation_payload.review_required` and `review_reasons`.

### Step 6: Validation hints

`build_validation_packet_hints()` converts validation result into extraction hints:

- ordered medical documents
- document type counts
- image document counts
- total detected documents

These hints help extraction preserve document inventory and order.

### Step 7: Extraction model selection

`MedicalDocumentProcessor.process()`:

- loads active scan-compatible AI models from the database
- sorts by priority
- takes up to `AI_EXTRACTION_MAX_PRODUCERS`
- runs `AI_EXTRACTION_RUNS_PER_MODEL` extraction passes per model

Current ensemble config in `ai_ensemble_config.py`:

- `FUSION_VERSION = "ensemble-fusion-v3"`
- `CALIBRATION_VERSION = "self_consistency_logistic_v1"`
- `AI_EXTRACTION_MAX_PRODUCERS = 3`
- `AI_EXTRACTION_RUNS_PER_MODEL = 2`
- `AI_EXTRACTION_MIN_SUCCESSFUL_RUNS = 2`
- `AI_EXTRACTION_TEMPERATURE = 0.2`

Execution mode:

- `single_model` if one producer and one run.
- `ensemble` if multiple producers or multiple runs.

### Step 8: Individual extraction runs

For each scheduled run:

- build system/user messages from `medical_document_scan.yaml`
- attach every processed image as an `image_url` data URL
- invoke the LangChain chat model
- parse text as JSON
- validate against `AIMedicalScanResponse`
- persist an `ai_api_call_logs` row
- return a `ProducerInferenceTrace`

Failed runs are captured as traces with error messages instead of crashing the whole ensemble.

If fewer than the required number of successful runs exist, the scan fails with:

`AI ensemble failed to produce enough valid outputs.`

### Step 9: Ensemble fusion

Implemented in `ta_backend/app/services/ai_ensemble.py`.

Fusion takes successful traces and creates one `MedicalExtractionOutcome`.

Major responsibilities:

- select scalar fields by normalized candidate support
- preserve and score list items
- normalize comparable values
- handle numeric/date/gender values
- handle handwritten/semantic uncertainty
- optionally invoke field judge for conflicts
- cache equivalent field-judge decisions, including null/no-selection results
- compute `field_confidence`
- compute `overall_confidence`
- track judge invocations, attempts, applied selections, and cache hits
- track list group metadata
- identify review reasons
- nullify ambiguous fields when evidence is weak
- preserve extraction metadata for admin/debug/eval

Critical fields include:

- `summary.medical_facility_name`
- `summary.service_date`
- `summary.total_amount`
- `patient_info.name`
- `financial_recap.total_medicines`
- `financial_recap.total_other_charges`
- `financial_recap.grand_total`

Semantic fields include:

- `medical_context.anamnesis`
- `medical_context.therapy_note`
- dosage-instruction suffixes

### Step 10: Financial reconciliation

Financial reconciliation is inside `MedicalExtractionFusionEngine`.

It tries to keep the ledger coherent by comparing:

- medicine item totals
- other charge totals
- financial recap totals
- summary total amount
- terbilang/Indonesian amount-in-words cues
- visible receipt-backed evidence represented by model outputs

It can:

- backfill missing totals
- correct mismatched totals
- suppress financial outliers
- classify pharmacy/prescription payment rows as medicine spending when descriptors contain cues such as apotek, apotik, farmasi, resep, prescription, nota penjualan, struk obat, or pembelian obat
- deduplicate same-payment evidence across bills, invoices, receipts, payment slips, and duplicate document views
- merge same-amount, same-family non-medicine charges such as duplicate laboratory or consultation rows
- cluster near-equal OCR totals before comparing candidate ledgers
- rank plausible coherent ledger candidates using support, receipt backing, total matches, medicine-source quality, and dropped-outlier penalties
- skip packet-level financial judging when the top heuristic ledger is dominant
- invoke the packet-level financial judge only for close or materially ambiguous ledger ties
- preserve extracted financial totals when the financial judge abstains and no dominant ledger remains
- mark financial conflicts as review-worthy
- clear contradictory financial recap fields when no coherent ledger can be selected

Reconciliation metadata is stored under `extraction_meta_payload`, including:

- derived financials
- corrected fields
- unresolved conflicts
- terbilang consistency
- `financial_resolution_mode`: usually `heuristic`, or `adjudicated` when the packet-level financial judge selects a ledger
- `financial_evidence_summary`
- `financial_judge_skipped_reason`, especially `dominant_candidate`
- financial judge attempt/applied counts
- nullified inconsistent financial fields

### Step 11: Medicine prioritization

Implemented in `ta_backend/app/services/scan_medicine_prioritization.py`.

This runs after extraction/fusion.

It uses list group metadata from fusion, document images, diagnosis/anamnesis text, and recent medicine knowledge to:

- rank medicines by image support
- identify ambiguous medicine order/name candidates
- ask a batch medicine image interpretation judge when needed
- optionally auto-correct medicine names only under guarded conditions

Auto-correct guarded mode is used when `APP_ENV` is `dev` or `test`; otherwise `suggest_only` is used.

Guarded auto-correct requires:

- interpreted name present
- interpretation confidence at least 0.92
- name similarity at least 0.62
- relevance confidence at least 0.55
- image support score at least 0.5

The postprocess log is stored in:

`extraction_meta_payload["medicine_postprocess"]`

### Step 12: Patient-name fallback

If validation did not conclusively check the patient name (`not_found`, `uncertain`, or `not_checked`), the backend checks the extracted `patient_info.name` against user/family candidates.

Mismatch causes scan failure with `patient_name_mismatch`.

### Step 13: Persist success or failure

On success:

- scan status becomes `succeeded`
- validation payload is stored
- extraction payload is stored
- extraction meta payload is stored
- primary provider/model are stored
- AI run traces are stored
- progress becomes completed/100
- SSE `completed` event is published

On failure:

- scan status is set based on failure type
- validation/extraction partial payloads may be stored
- AI run traces may be stored
- error payload is stored
- progress becomes failed
- SSE `failed` event is published

## 12. Scan Progress And SSE

Progress types are defined in `ta_backend/app/schemas/ai.py`.

Stages:

- `queued`
- `preprocessing`
- `validating`
- `extracting`
- `fusing`
- `prioritizing_medicines`
- `completed`
- `failed`

Progress payload fields:

- `stage`
- `title`
- `detail`
- `percent`
- `status`
- `image_count`
- `total_runs`
- `completed_runs`
- `successful_runs`
- `medicine_count`

SSE implementation:

- `ta_backend/app/services/scan_progress.py`
- in-memory `ScanProgressEventHub`
- subscribers receive initial snapshot, progress, terminal events, and heartbeat
- frontend uses `react-native-sse`

Observed cancellation behavior:

- `POST /api/ai/scan-documents/jobs/{scan_log_id}/cancel` marks a pending/running scan as `cancelled` and publishes a cancelled event.
- The long-running background pipeline does not appear to include a cooperative cancellation check inside each stage. Future agents should verify current code before relying on hard cancellation semantics.

## 13. Claim Submission In Detail

The frontend submits the final reviewed form to `POST /api/claims`.

Required form fields:

- `nama_pasien`
- `tanggal_pelayanan`, expected `dd-mm-yyyy`
- `fasilitas_kesehatan`
- `jenis_manfaat`
- `nominal_klaim`
- `keterangan_penyakit`
- `kronologi`
- files

Supported `jenis_manfaat` values:

- `rawat_jalan`
- `rawat_inap`
- `emergency`
- `maternity`

Optional:

- `medicines_payload`, JSON array
- `source_scan_log_id`

Claim creation flow:

1. Validate benefit type.
2. Parse service date.
3. Parse medicines payload.
4. Evaluate medicine relevance with `evaluate_claim_medicines()`.
5. Read uploaded files and hashes.
6. If linked to a scan:
   - verify scan ownership
   - verify file hashes match the scan log
   - reuse validation if possible
   - otherwise rerun validation
   - ensure selected patient matches validation evidence
7. If not linked to a scan:
   - preprocess uploaded files
   - validate documents with expected selected patient name
8. Create claim and claim files.
9. Persist positive/uncertain medicine relevance observations.
10. Return `ClaimDetailResponse`.

## 14. Medicine Relevance System

Implemented in `ta_backend/app/services/claim_medicine_relevance.py`.

Purpose:

- Determine whether submitted medicines look relevant to diagnosis/anamnesis.
- Mark claims for manual review when medicines appear not relevant.
- Learn relevance observations over time.

Decision sources:

- `history_strong`
- `llm_judge`
- `heuristic_fallback`

Flow:

1. Normalize diagnosis and anamnesis text.
2. Tokenize and expand synonyms.
3. Build a context signature from normalized diagnosis/anamnesis.
4. For each medicine:
   - look for exact historical context+medicine row
   - otherwise retrieve rows by medicine name
   - compute similarity to current context
   - use strong history if support/dominance thresholds pass
   - otherwise ask LLM judge with historical evidence
   - if LLM fails, use heuristic fallback
5. Mark claim for additional review if any medicine is `not_relevant`.
6. Persist observations for medicines that are not `not_relevant`.

Some hardcoded context hints exist for common medicines such as paracetamol, ibuprofen, amoxicillin, cetirizine, omeprazole, salbutamol, ambroxol, and vitamins.

## 15. Frontend Architecture

Frontend uses Expo Router.

Root:

- `ta_frontend/app/_layout.tsx`
- wraps app in theme provider and auth provider
- stack routes: `index`, `(auth)`, `user`, `admin`

Auth:

- `ta_frontend/contexts/auth-context.tsx`
- token stored in AsyncStorage under `auth_token`
- app checks `/auth/me` when token exists
- login stores token
- register stores the token returned by backend registration

Important note:

- Backend `POST /auth/register` returns `message`, `access_token`, `token_type`, and `user`.
- Frontend `registerUser()` stores the registration token with the same auth-token path as login.
- This keeps frontend auth context and persisted bearer token aligned immediately after registration.

Route redirect:

- `ta_frontend/app/index.tsx`
- unauthenticated users go to `/login`
- admins go to `/admin/settings`
- normal users go to `/user/new-claim`

## 16. Frontend New Claim Flow

Main file:

`ta_frontend/features/claims/screens/new-claim-screen.tsx`

State includes:

- uploaded image URIs
- family members
- latest user/family profile completeness
- selected patient
- service date
- facility
- benefit type
- claim amount
- disease description
- medicine drafts
- chronology/anamnesis
- source scan log ID
- scan state and progress
- submit status/errors

User actions:

- Complete profile data from the Profile tab before claim entry is available.
- Pick multiple images from gallery.
- Take a photo with camera.
- Remove uploaded image.
- Start AI scan.
- Cancel active scan.
- Select patient from self/family member list.
- Select date.
- Select benefit type.
- Edit all autofilled fields.
- Add/remove/edit medicine names.
- Submit claim.

AI scan UI flow:

0. On focus, frontend refreshes `/auth/me` and `/family-members`. If the user profile or any saved family member is incomplete, the New Claim screen is replaced with a profile-completion message and a button to go to Profile.
1. User uploads images.
2. User taps `Scan with AI`.
3. Frontend calls `startScanDocumentsJob()`.
4. Upload progress is displayed as 0-15 percent.
5. Frontend opens SSE stream for scan job.
6. Progress events update `ClaimScanProgressCard`.
7. Completed event applies scan data to the form.
8. Failed/cancelled event shows error feedback.
9. If SSE errors, frontend polls `getScanDocumentsJob()` as fallback.

Autofill mapping:

- `patient_info.name` is matched to existing patient options by normalized exact name.
- `summary.medical_facility_name` fills facility.
- `summary.service_date` or `summary.transaction_date` fills service date if ISO parse succeeds.
- `summary.total_amount` fills claim amount.
- diagnosis descriptions join into disease description.
- `medical_context.anamnesis` fills chronology.
- medicines map to medicine draft names only.

Submit mapping:

- selected patient name -> `nama_pasien`
- date -> `dd-mm-yyyy`
- facility -> `fasilitas_kesehatan`
- benefit type -> `jenis_manfaat`
- amount -> `nominal_klaim`
- disease description -> `keterangan_penyakit`
- chronology -> `kronologi`
- medicine drafts -> `medicines_payload`
- image URIs -> `files`
- `sourceScanLogId` -> `source_scan_log_id`
- selected patient type -> `selected_patient_type`
- selected family member ID -> `family_member_id`

After successful submission, the frontend redirects to the new claim detail screen. Claim detail can display submitted document images, including an appended patient insurance-card image when profile data was available, and includes a `Download InHealth Form` action that downloads the filled PDF through `expo-file-system/legacy` and opens the native share/save sheet through `expo-sharing`.

## 17. Frontend Admin Flow

Admin route files live under:

```text
ta_frontend/app/admin/
ta_frontend/features/admin/
```

Implemented admin screens/components include:

- settings / AI model management
- scan log queue
- scan log detail
- claim queue
- claim detail
- profile
- queue summary strips and filter chips
- claim/scan queue cards

Admin API clients:

- `services/api/admin-ai-models.ts`
- `services/api/admin-scan-logs.ts`
- `services/api/admin-claims.ts`

Admin scan-log detail includes rich debug views for:

- validation payload
- extraction payload
- AI runs
- fusion metadata
- error payload
- original and processed image viewing

## 18. API Client Layer

File: `ta_frontend/services/api/client.ts`

Core helpers:

- `getJson`
- `postJson`
- `putJson`
- `deleteJson`
- `postFormData`
- `postFormDataWithProgress`

`postFormDataWithProgress` uses `XMLHttpRequest` to expose upload progress. JSON request helpers use `fetch`.

Errors are normalized to `ApiError`:

- `message`
- optional `status`

The first backend validation error message is preferred over a general message.

## 19. Evaluation System

Evaluation docs:

`ta_backend/eval/README.md`

Evaluation runs backend-only extraction tests against folders of claim examples. It reuses the backend scan route through FastAPI `TestClient`.

Dataset layout:

```text
eval_examples/
  claim_001/
    images/
      page1.jpg
      page2.jpg
    claim_001.json
```

Flat image folders also work.

Ground truth JSON contains:

- `images`
- `patient_name_candidates`
- `expected`

The `expected` object has the same extraction shape:

- `summary`
- `patient_info`
- `medical_context`
- `medicines`
- `other_charges`
- `financial_recap`

Evaluation modes:

- `auto`: cache-or-scan
- `eval-only`: evaluate from cache only
- `scan-only`: scan and save only

Cache:

```text
results/
  claim_001/
    latest.json
    runs/
      timestamp__claim-id__scan-id.json
```

Metrics:

- report schema: `medical_claim_extraction_eval_v2`
- primary metric: Strict KIE Micro F1 (`strict_kie_micro_f1`)
- strict KIE micro precision/recall/F1 over extracted key-information units
- Claim Macro F1: average of per-claim Strict KIE Micro F1
- Section Macro F1: average of section-level strict KIE F1
- Critical Field Exact Accuracy
- Financial Exact Accuracy
- relaxed text ANLS and CER
- entity-level scoring details for diagnoses, medicines, and other charges
- usability diagnostics: edit burden score, verify-only pass/rate, and critical fail reasons
- token usage aggregation from scan run records and AI API call logs

Metric basis:

- Key information extraction precision/recall/F1 follows common SROIE/ICDAR and LayoutLM-style document extraction evaluation practice: https://arxiv.org/abs/2103.10213 and https://aclanthology.org/2021.acl-long.201.pdf
- Text similarity uses DocVQA-style ANLS and OCR-style CER as secondary diagnostics: https://openaccess.thecvf.com/content/WACV2021/papers/Mathew_DocVQA_A_Dataset_for_VQA_on_Document_Images_WACV_2021_paper.pdf and https://ocr-d.de/en/spec/ocrd_eval.html
- The project-specific `pass_verify_only` gate is not the academic score; it is a workflow diagnostic built on top of strict extraction results.

Current documented limits:

- Evaluation is extraction-only.
- Medicine relevance is not scored.
- Runner processes claims sequentially.
- Strict entity keys require the correct canonical anchor before attributes are scored.
- `edit_burden_score` and `pass_verify_only` are project-specific diagnostics, not paper-standard metrics.

`run_eval.py` creates/uses a local `.venv`, installs requirements when hash changes, and dispatches to:

- `python -m eval.run_evaluation`
- `python -m eval.generate_template`
- cache-clearing utilities

Self-consistency calibration:

- `eval/self_consistency_calibration.py`
- reads cached scan run logs and `field_confidence`
- fits logistic calibrator
- writes artifact consumed by fusion

Runtime artifact:

`ta_backend/app/artifacts/ai_ensemble_self_consistency_logistic_v1.json`

## 20. Tests

Backend tests are extensive and live in `ta_backend/tests/`.

Coverage includes:

- auth registration/login/error envelopes
- family member CRUD/access control
- claim creation/list/detail/linking
- admin claim routes
- admin scan log routes
- admin AI model routes
- image preprocessing
- scan API behavior
- scan job progress/SSE/cancellation events
- AI pipeline logging
- provider invocation tests
- prompt message building
- JSON parsing/schema validation
- ensemble fusion behavior
- field judge behavior, including judge cache reuse
- financial reconciliation, dominance checks, adjudication, abstention, and packet-level financial judge selection context
- pharmacy receipt deduplication and multi-receipt packet reconciliation
- terbilang parsing
- gender ambiguity
- document validation rules
- medicine prioritization, including the `extracted_name` batch-judge contract
- medicine relevance/audit logging
- evaluation tooling, including schema v2 and token usage aggregation
- proposal-aligned black-box API partitions
- optional PostgreSQL persistence integration marked with `pytest.mark.postgres`

Test discovery command:

```powershell
cd ta_backend
pytest
```

Important: Real AI provider calls are generally mocked/stubbed in tests. Running real scans requires configured provider credentials and network access.

Optional PostgreSQL integration tests require `TEST_POSTGRES_DATABASE_URL` and are skipped unless the configured database name looks like a test database.

## 21. Proposal Alignment

The implementation matches the proposal in these areas:

- React Native mobile client.
- FastAPI backend.
- LangChain-based VLM orchestration.
- Image preprocessing: resize, CLAHE, sharpening, encoding.
- Multi-image document upload.
- Medical document validation.
- Patient identity verification.
- Stamp/signature validation.
- Structured JSON extraction.
- User verification/editing before final claim submission.
- Admin model management and scan inspection.
- Evaluation against ground truth with extraction metrics.
- Proposal-aligned black-box API tests for scan and claim flows.

Implementation goes beyond the proposal in these areas:

- Ensemble extraction with multiple producers/runs.
- Confidence calibration.
- Field-level conflict adjudication.
- Packet-level financial adjudication.
- Financial ledger reconciliation.
- Detailed AI API call logs.
- Scan progress streaming.
- Medicine relevance learning and review flags.
- Medicine postprocessing using image support and historical hints.
- Admin scan log debug views.

Differences or caveats:

- Proposal says PostgreSQL, but default implementation is SQLite. Optional PostgreSQL persistence coverage exists through `pytest.mark.postgres` and `TEST_POSTGRES_DATABASE_URL`.
- Proposal mentions VLM playground; current repo has admin model settings and detailed logs, but a distinct free-form playground UI was not identified.
- Proposal says output field `transaction_date`; current backend schema also has `service_date`.
- Current frontend only submits medicine names from scan output, not quantity/unit/dosage/price fields, although backend schemas can store those optional fields.

## 22. Common Data Contracts

### AI scan response

Pydantic class:

`AIMedicalScanResponse`

Returned to user as:

`AIScanDocumentsResponse`, which adds `scan_log_id`.

Shape:

```json
{
  "scan_log_id": 1,
  "summary": {
    "detected_documents": ["Kwitansi", "Resep"],
    "medical_facility_name": "string or null",
    "service_date": "YYYY-MM-DD or null",
    "transaction_date": "YYYY-MM-DD or null",
    "currency": "IDR or null",
    "total_amount": 12345
  },
  "patient_info": {
    "name": "string or null",
    "date_of_birth": "YYYY-MM-DD or null",
    "age": 30,
    "gender": "MALE/FEMALE/null",
    "address": "string or null"
  },
  "medical_context": {
    "anamnesis": "string or null",
    "diagnoses": [
      {
        "description": "string or null",
        "doctor_name": "string or null"
      }
    ],
    "therapy_note": "string or null"
  },
  "medicines": [
    {
      "name": "string or null",
      "quantity": 1,
      "unit": "string or null",
      "dosage_instruction": "string or null",
      "unit_price": 1000,
      "total_price": 1000
    }
  ],
  "other_charges": [
    {
      "category": "CONSULTATION",
      "description": "string or null",
      "amount": 100000
    }
  ],
  "financial_recap": {
    "total_medicines": 1000,
    "total_other_charges": 100000,
    "grand_total": 101000
  }
}
```

Actual fields may be null when unsupported or unreadable.

### Scan job status response

Pydantic class:

`AIScanJobStatusResponse`

Shape:

```json
{
  "scan_log_id": 1,
  "progress": {
    "stage": "extracting",
    "title": "Extracting medical data",
    "detail": "Running 1 of 2 AI extraction passes.",
    "percent": 60,
    "status": "running",
    "image_count": 2,
    "total_runs": 2,
    "completed_runs": 1,
    "successful_runs": 1,
    "medicine_count": null
  },
  "result": null,
  "error_message": null
}
```

When succeeded, `result` contains `AIScanDocumentsResponse`.

When failed, `error_message` is populated.

### Claim create medicine payload

Backend schema:

`ClaimMedicineInput`

Fields:

- `name`, required
- `quantity`, optional
- `unit`, optional
- `dosage_instruction`, optional
- `source`, either `ai_extracted` or `user_input`

Frontend currently sends only name and source.

### Claim create patient/export payload

Claim creation accepts optional form fields:

- `selected_patient_type`: `self` or `family`
- `family_member_id`: required when `selected_patient_type` is `family`

These fields are used to resolve the patient profile, store internal export snapshots, and append the selected patient's insurance card image to claim files. They are not returned as public claim response fields.

### Claim response

Backend schema:

`ClaimResponse`

Fields:

- claim identity and user fields
- patient/service/facility/benefit/amount
- disease and chronology
- evaluated medicines
- review flag and review note
- file count
- status
- source scan log ID
- created timestamp

`ClaimDetailResponse` additionally includes:

- `source_scan_extraction_payload`
- `files`, which may include the appended patient insurance-card image

## 23. Important Implementation Notes For Future Agents

1. Treat this repo as a prototype, not a production insurance core system.

2. The AI pipeline is stateful through the database. Active model rows affect runtime behavior.

3. The default active model is Google Gemini. Without `GOOGLE_API_KEY`, real scans fail unless another compatible provider is activated and credentialed.

4. When editing scan behavior, check tests in `ta_backend/tests/test_ai_api.py`; many edge cases are covered there.

5. When editing final claim behavior, check `ta_backend/tests/test_claims_api.py`.

6. When editing image preprocessing, check `ta_backend/tests/test_image_preprocessing.py`.

7. When editing medicine prioritization, check `ta_backend/tests/test_scan_medicine_prioritization.py`.

8. When editing evaluation metrics or ground-truth behavior, check `ta_backend/tests/test_eval_tools.py` and `ta_backend/eval/README.md`; the current evaluator contract is schema `medical_claim_extraction_eval_v2` with primary metric `strict_kie_micro_f1`.

9. The frontend TypeScript types may not perfectly model backend nullability. Backend schemas allow many AI fields to be null, while some frontend types use plain `string`/`number`.

10. The scan logs store binary original and processed images in the database, and admin detail returns those images as data URLs. Be careful about payload size.

11. AI API call logging sanitizes prompt images but may store raw model text responses and parsed payloads. Be careful with PII when adding logs.

12. The current extraction output contract expects no extra keys. Any prompt/schema change must keep Pydantic validation and tests aligned.

13. The ensemble fusion layer is complex and should be changed with targeted tests. It handles field confidence, field-judge caching, list coherence, financial reconciliation, packet-level financial adjudication, and review reasons.

14. When touching financial reconciliation or field judging, update `PROJECT_CONTEXT.md` and tests together. Current tests cover dominant heuristic ledgers, financial judge adjudication/abstention, pharmacy receipt deduplication, and judge cache behavior.

15. The eval ground-truth guidance intentionally says ground truth should match the current pipeline's realistic output style, not an idealized schema. This matters for scoring consistency.

16. User-file matching between scan and claim uses SHA-256 hashes of original uploaded files. If frontend image URIs are transformed between scan and submit, linking can fail.

17. Claims can be submitted manually without a source scan log, but backend still validates uploaded documents during claim creation.

18. InHealth PDF export depends on complete profile data, selected patient identity, the static PDF template, and installed `pypdf`/`reportlab` dependencies. Existing/old claims without snapshot data use a best-effort current-profile fallback.

19. Document validation can produce soft review reasons instead of hard failure for some collage/orientation cases.

20. In `APP_ENV=dev` or `test`, medicine prioritization may apply guarded auto-correction. In other environments it should only suggest.

21. Admin AI model activation blocks providers that are not image scan-compatible according to `ai_provider_catalog.py`.

22. Proposal-aligned black-box tests live in `ta_backend/tests/test_proposal_black_box_api.py`; the PostgreSQL case is optional and guarded by the `postgres` marker.

23. The `.vscode/mcp.json` file configures a sequential thinking MCP server via `npx`, but that does not mean the running agent session has a callable sequential-thinking tool.

## 24. Quick End-To-End Flow Summary

User scan/autofill:

```text
Mobile New Claim screen
  -> pick/take document images
  -> POST /api/ai/scan-documents/jobs
  -> backend creates scan_log and stores originals
  -> background scan preprocesses images
  -> validation VLM checks documents and patient name
  -> extraction VLM runs multiple passes
  -> ensemble fusion selects fields and reconciles financials
  -> medicine prioritization postprocesses medicines
  -> scan_log stores payloads, runs, API calls, processed images
  -> SSE completed event returns result
  -> frontend autofills claim form
  -> user reviews/edits fields
```

Final claim:

```text
Mobile Submit Claim
  -> POST /api/claims multipart form
  -> backend resolves selected patient profile and stores export snapshot
  -> backend appends patient insurance-card image when available
  -> backend validates selected patient and uploaded medical documents
  -> if linked scan exists, verify same files and reuse/rerun validation
  -> evaluate medicine relevance
  -> persist claim and claim files
  -> optionally persist medicine relevance knowledge
  -> return claim detail
  -> claim detail can request /api/claims/{claim_id}/inhealth-form
```

Admin review:

```text
Admin app
  -> list scan logs / claims
  -> inspect original and processed images
  -> inspect validation, extraction, fusion, errors, AI runs, API call logs
  -> manage active AI models and priorities
```

Evaluation:

```text
Ground-truth claim folders
  -> eval runner calls backend scan route
  -> predictions cached under results/
  -> evaluator compares prediction vs expected JSON
  -> reports strict KIE micro-F1, exact checks, relaxed text metrics, diagnostics, diffs
```
