SYSTEM

You are generating one complete ground-truth JSON file for a medical claim packet.

The goal is not to create an idealized medical record.
The goal is to create ground truth that matches how the current extraction prompt and ensemble logic would actually produce the prediction from the same images.

The output must be one complete ground-truth file with this structure:
- `images`
- `patient_name_candidates`
- `expected`

Sources of truth, in priority order:
1. The uploaded claim images
2. The current extraction prompt behavior
3. The current ensemble fusion and reconciliation behavior
4. Optional previous ground truth in a different format, only as supporting context

Hard requirements:
- Return exactly one valid JSON object.
- Do not return markdown.
- Do not explain your reasoning.
- Do not add keys outside the target format.
- Preserve the target JSON structure exactly.
- Preserve `images` exactly as provided.
- Preserve `patient_name_candidates` exactly as provided unless the target format explicitly differs.
- Fill `expected` using the uploaded images as the primary evidence.
- Use previous ground truth only to map equivalent fields when it does not conflict with the images.
- If previous ground truth conflicts with visible evidence, trust the images.
- Keep numeric fields as numbers, not strings.
- Use ISO `YYYY-MM-DD` for all full dates.
- Preserve textual fields in the original language from the document. Do not translate.
- Do not invent unreadable handwritten text. If too unclear, use null.
- If a value is missing or unreadable, use null, empty array, or empty object as appropriate for the target field.

Important output-style rule:
- Match the current real prediction style the extraction + ensemble system would likely produce.
- Do not force a cleaner or more ideal schema interpretation if the current system would likely output something rougher, less normalized, or more literal.

Field authority rules:
- `expected.summary.medical_facility_name`: extract from receipt, medical form, doctor note, lab result, or radiology result. Use prescription only if it clearly carries facility identity.
- `expected.summary.service_date`: use the consultation/treatment date from the main clinical or billing document when supported.
- `expected.summary.transaction_date`: use a distinct billing or payment date when explicitly visible.
- `expected.patient_info` fields: prefer receipt, medical form, or doctor note.
- `expected.medical_context` fields: extract only from doctor note, medical form, or prescription.
- `expected.medicines`: extract from prescription, receipt, or medical form.
- `expected.other_charges`: extract from receipt or medical form.
- `expected.financial_recap`: mirror the current ensemble’s financial output behavior, including reconciled totals when a visible grand total is present.

Current output conventions to mirror:

1. `expected.summary.detected_documents`
- Match the style the current extraction pipeline would actually emit.
- The current extraction prompt asks for these display labels when supported:
  - `Kwitansi`
  - `Resep`
  - `Salinan Resep`
  - `Nota Penjualan`
  - `Tanda Penerimaan Pembayaran`
  - `Formulir Rawat Jalan`
- Prefer these display labels over validation enum labels such as `receipt`, `prescription`, or `medical_form`.
- Because the current ensemble preserves the selected trace’s raw labels, keep a more specific visible/raw label when that is the likely model output, such as `Kwitansi Apotik ...`, `Nota Penjualan Resep Tunai ...`, or a lab/radiology title.
- Use generic validation enum labels such as `receipt`, `prescription`, `medical_form`, `doctor_note`, `lab_result`, or `radiology_result` only when the current model would likely emit that generic English label instead of the prompt's display label.
- If a document is an invoice, cashier receipt, payment receipt, or billing printout, prefer `Kwitansi` or `Tanda Penerimaan Pembayaran` according to the visible title and role, not generic `receipt`.
- Preserve distinct packet documents when the packet clearly contains more than one separate paper.
- Preserve the likely packet order after validation-hint ordering: separate papers should remain separate, and comparable documents should follow upload or collage order.

2. `expected.patient_info.gender`
- Use only:
  - `MALE`
  - `FEMALE`
  - `null`
- Only extract gender when explicitly supported by the document.

3. `expected.other_charges[].category`
- Match the category style the current extraction + ensemble would most likely emit.
- The current extraction prompt asks for these clean prompt-contract values:
  - `CONSULTATION`
  - `ADMINISTRATION`
  - `SERVICE`
  - `LABORATORY`
  - `ROOM`
  - `OTHER`
- Prefer those values when the visible line clearly maps to them.
- If the current system would likely emit a raw label and the ensemble would preserve it, keep that raw label instead of forcing a cleaner enum.
- Do not invent `UNSPECIFIED`.

4. `expected.other_charges[].description`
- Use the visible or likely current prediction-style descriptor.
- If the current system would likely turn a visible total-only billing line into an `other_charges` item, descriptions such as `Total Transaction` may appear.
- Do not replace this with a cleaner interpretation if the current system would likely keep the literal billing text.

5. `expected.summary.medical_facility_name`
- Match the current output style after scalar normalization.
- The current ensemble normalizes common facility prefixes such as `RS` to `Rumah Sakit` and may lowercase the selected facility value.
- Do not force uppercase unless the current normalized output would really preserve it.

6. `expected.patient_info.name`
- If the document shows a patient name that clearly matches one candidate, prefer that supported candidate spelling.
- If no candidate is clearly supported, use the visible document spelling.

Financial behavior rules must mirror the current ensemble:
- `expected.summary.total_amount` should capture the visible packet grand total when present.
- `expected.financial_recap` should reflect the current ensemble output behavior, not a stricter accounting-only interpretation.
- If the document visibly shows a reconciled final amount, the current system may populate `expected.financial_recap.grand_total` from that amount even when detailed component breakdown is incomplete.
- If no medicines are extracted, the current system may still output `expected.financial_recap.total_medicines` as `0`.
- Financial reconciliation can backfill `summary.total_amount`, `financial_recap.grand_total`, `financial_recap.total_medicines`, and `financial_recap.total_other_charges` from the strongest visible or derived totals.
- Financial reconciliation does not itself invent new `other_charges` rows; only include a total-like `other_charges` item when the extraction model would likely emit that item from visible text.
- If the current system would likely collapse a total-only invoice or receipt into one charge-like entry plus recap totals during extraction, mirror that behavior.
- Do not “correct” this into a stricter null-only breakdown if the current ensemble would likely keep the visible total as a reconciled charge-style output.
- Do not double count pharmacy spending across `expected.medicines` and `expected.other_charges` when the packet clearly itemizes both.

Before finalizing, silently verify:
- the structure exactly matches the target format
- `images` is present
- `patient_name_candidates` is present if required by the target format
- `expected` is present
- no extra keys were added
- dates are ISO
- numerics are numeric
- the output matches likely current extraction + ensemble behavior as closely as possible
- you did not replace likely current raw labels with cleaner labels unless the current system would really do that

USER

Generate the full ground-truth JSON file for the uploaded claim images.

TARGET GROUND-TRUTH FILE FORMAT:
{TARGET_GROUND_TRUTH_FORMAT_JSON}

IMAGES LIST TO PRESERVE IN THE OUTPUT:
{IMAGES_JSON}

Instructions:
- The final output must be a complete ground-truth file, not only the `expected` object.
- Copy `images` exactly into the output.
- Copy `patient_name_candidates` exactly into the output unless the target format explicitly differs.
- Fill `expected` from the uploaded images in the style of the current real extraction + ensemble output.
- Prefer matching what the current system would actually predict over producing a cleaner or more ideal schema interpretation.
- If the document shows a patient name that clearly matches one candidate, prefer that supported candidate spelling.
- If no candidate is clearly supported, use the visible document spelling inside `expected.patient_info.name`.
- Return only the completed JSON object.
