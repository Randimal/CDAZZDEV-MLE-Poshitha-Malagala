# Generated data

Task 2A writes the following artifacts here after real teacher execution:

- `raw_teacher_examples.jsonl`: one batch envelope per line, including assignments,
  exact teacher response, model, seed, system-prompt hash and safe failure category.
- `cleaned_examples.jsonl`: locally validated, globally deduplicated PolicyExample records.
- `analysis_summary.json`: configuration, validation/rejection and deduplication counts.
- `train.jsonl`, `validation.jsonl`, `test.jsonl`: chat messages and provenance IDs.
- `split_manifest.json`: exact counts, stratification, seed, IDs and SHA-256 hashes.

No example datasets are prefilled. Unit-test fixtures are never exported here.
The notebook saves raw responses after each batch so an interruption preserves
completed responses. Resume revalidates that journal and appends only missing-ID
requests. Existing successes are never regenerated, old failures remain auditable,
and raw bytes are preserved. A corrupt/incompatible journal fails before any API
call or write. Keep both raw and clean files; do not delete a failed run to resume.
Existing splits are protected from overwrite. Once split, keep
`test.jsonl` untouched for Task 2C evaluation, including its manifest hash.
