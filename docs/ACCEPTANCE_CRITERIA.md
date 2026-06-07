# Acceptance Criteria

This file defines how we decide whether something is really done.

## Global Rule

Do not mark a feature done unless:

1. The files exist.
2. The feature is linked from docs where relevant.
3. The behavior was tested locally, or the blocker is written in `IMPLEMENTATION_STATUS.md`.
4. The checklist in `TASKS.md` is updated.
5. Unit and user-story tests are added or updated.

If a bug or failed check is discovered, add or update a test that reproduces it before marking
the fix done.

## MVP App

Done when:

- `python app.py` launches without errors.
- The Gradio UI opens locally.
- Chat tab returns a response.
- Vision tab accepts an image and returns a response.
- Field Notes tab saves a row to `data/field_notes.csv`.
- README explains that current inference is placeholder/local-safe.
- Unit and user-story tests pass.

## Testing And Quality

Done when:

- Unit tests cover new logic.
- User-story tests cover new workflows.
- `scripts/run_tests.ps1` passes.
- `ruff`, `mypy`, `pylint`, `bandit`, and `pip-audit` are installed or blockers are documented.
- `scripts/run_quality.ps1` passes before release or documents exact blockers.

## Real Local Inference

Done when:

- A user can choose a backend.
- Model loading happens only after an explicit click.
- The app does not download weights on startup.
- At least one small model returns a real response.
- Setup steps are documented.

## Hugging Face Space

Done when:

- Space exists.
- Repo is pushed.
- Build succeeds.
- Public/private Space URL is added to README.
- Any hardware or model limitations are documented.

## GitHub

Done when:

- GitHub repo exists.
- Initial commit is pushed.
- README includes GitHub URL.
- Large files and secrets are excluded.

## Training

Done when:

- Training input format is documented.
- A dry-run or actual LoRA run completes.
- Output checkpoint path is shown.
- Metrics or logs are captured.
- Hardware assumptions are documented.

## Field Notes

Done when:

- Corrections save reliably.
- Corrected records can be exported.
- Export schema is documented.
- Export can feed the training pipeline.

## Hackathon Submission

Done when:

- Space URL is live.
- Demo video is recorded.
- Social post is published or drafted.
- Submission form fields are complete.
- Model-size compliance is documented.
