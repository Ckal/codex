# Roadmap V2 - Critical Hackathon Improvement Plan

This document is intentionally hard on the project. It answers: "If a strong automated judge or
human judge reviews this repo, what will cost points, and what should be fixed first?"

Sources checked on 2026-06-08:

- Official Build Small Hackathon page: <https://huggingface.co/build-small-hackathon>
- Current repo status: `docs/IMPLEMENTATION_STATUS.md`
- PRD coverage: `docs/PRD_IMPLEMENTATION_MATRIX.md`
- Task checklist: `docs/TASKS.md`

## Executive Verdict

This is not yet a SOTA hackathon entry. It is a serious engineering scaffold with good local
quality discipline, but it is not yet a sharp, shipped product.

Current estimated rating:

| Dimension | Score | Hard reason |
| --- | ---: | --- |
| Hackathon fit | 5/10 | It uses Gradio and small-model framing, but the user story is diluted by too many tabs. |
| Shipped product | 3/10 | No verified Hugging Face Space URL, no generated screenshots, no submitted demo media. |
| Real usefulness | 5/10 | Field Notes plus correction loop can be useful, but it needs one real user, one real workflow, and evidence. |
| Technical architecture | 6/10 | Modules are clean enough, but UI callbacks still orchestrate too much and global state is too central. |
| Security posture | 5/10 | Basic secret/model-weight policy exists, but URL/path inputs and future command execution need hardening. |
| Test discipline | 7/10 | Good Python quality gates and unit tests; missing executed browser E2E, visual checks, and real backend integration. |
| Design and UX | 4/10 | Functional Gradio layout, but not memorable, not guided, and not custom enough for an "Off-Brand" award. |
| SOTA small-model angle | 4/10 | Many backend planners exist, but there is no benchmarked small-model capability story yet. |

Overall: **5/10 as a repo, 3.5/10 as a winning submission today.**

The strongest path is to stop trying to show everything and make one narrow workflow excellent:

> A local-first Field Notes Correction Workbench for collecting messy human observations, correcting
> small-model/OCR outputs, exporting training data, and showing how small models improve with human
> feedback.

That can plausibly compete in Backyard AI, Field Notes, Sharing is Caring, Llama Champion, Tiny Titan,
OpenBMB, and Best Demo if executed cleanly.

## What A GPT-Style Judge Will Likely Measure

The official page does not publish a numeric rubric, but it does state the visible judging signals.
Assume an automated GPT judge will inspect the Space, README, demo, screenshots, repository, and
submission text for the following:

1. **Track alignment**
   - Backyard AI: specific real problem, real person used it, honest small-model fit, Gradio polish.
   - Thousand Token Wood: delight, AI as the core experience, originality, Gradio polish.

2. **Hard constraints**
   - Gradio app.
   - Hosted as a Hugging Face Space.
   - Models at or below 32B parameters.
   - Short demo video and social/social-style post.

3. **Bonus quest evidence**
   - Off the Grid: no cloud APIs.
   - Well-Tuned: published fine-tuned model.
   - Off-Brand: custom UI beyond default Gradio.
   - Llama Champion: llama.cpp runtime.
   - Sharing is Caring: public trace on the Hub.
   - Field Notes: report or blog post about build and lessons.

4. **Repo truthfulness**
   - README claims match implementation.
   - No hidden placeholders presented as working model features.
   - Tests and status docs do not contradict the demo.

5. **User outcome**
   - A judge can answer: who is this for, what did they do before, what do they do now, and what got
     measurably better?

6. **Demo reliability**
   - App loads quickly.
   - One golden path works without local-only secrets.
   - Screenshots and demo video show actual outputs, not empty controls.

7. **Novelty and "small model" honesty**
   - The app should show why a small model is enough or better.
   - A tiny/small model should do something concrete, not merely be listed in config.

## Current Critical Weaknesses

### Product Weaknesses

- The app reads like a toolbox, not a product. A judge will not spend long discovering value across
  Chat, Vision, Dataset, Train, vLLM, Export, Field Notes, Traces, Agent, and Status.
- The project story is not yet visceral. "OpenBMB Local AI Workbench" sounds useful to builders,
  but the hackathon asks for either a real person or a delightful experience.
- The strongest workflow is Field Notes/OCR/correction, but it is not yet the first screen or the
  main README story.
- No demo screenshots have been generated yet because Node/npm is missing.
- No public Space URL means the current entry cannot satisfy the central shipping requirement.

### Architecture Weaknesses

- UI callbacks currently act as controllers. They validate inputs, call services, emit events, and
  format output. This is acceptable for an MVP, but it will become brittle.
- `APP_STATE` is a global singleton. That is simple locally, but it is risky for multi-user Spaces,
  test isolation, and future per-session traces.
- Service creation is centralized in `models/service_factory.py`, but there is no application-layer
  facade for workflows such as "correct note -> export dataset -> evaluate -> log trace".
- There are many non-executing planners. That is honest, but it creates a product risk: the app can
  look more complete than it is.
- There is no domain model for a complete demo run. The app records events, but not a first-class
  `DemoSession`, `CorrectionSession`, or `EvaluationRun`.

### Security Weaknesses

- User-controlled URLs are accepted for local backends. This can become SSRF if deployed on a Space
  and allowed to reach internal services.
- User-controlled local paths are accepted for datasets, GGUF files, OCR files, exports, and media.
  This is useful locally, but dangerous on a public Space without path allowlists.
- Future command execution is planned. Command builders are currently non-executing, which is good,
  but the moment execution is added, strict argument arrays, allowlists, and confirmation gates are
  mandatory.
- MCP exposure is attractive for judges, but it can widen the attack surface. Tool inputs need
  explicit schemas, rate limits, and audit logs before public use.
- Agent mode is non-autonomous today. If autonomy is added, shell/git/deploy/model-download actions
  need hard policy gates.

### Testing Weaknesses

- Playwright E2E exists but is not run locally because Node/npm is missing.
- No browser screenshots have been generated yet for docs.
- No visual regression baseline.
- No real backend integration test for LM Studio, llama.cpp, Ollama, SGLang, vLLM, or MiniCPM Vision.
- Coverage is enough for the current threshold, but important UI code remains lightly covered.
- Security tests are thin: path allowlist, URL allowlist, SSRF prevention, file-size limits, and
  malformed JSONL/CSV cases should be tested.

### Design Weaknesses

- The UI is still mostly default Gradio. That likely misses the Off-Brand award.
- The first screen does not guide a judge into a single demo path.
- There is no "Judge Mode" that preloads sample data, explains what to click, and proves the core
  value in under two minutes.
- The README does not yet lead with screenshots, output examples, and the exact demo path.
- Status/setup surfaces are useful, but they should not dominate the judged experience.

## Architecture Recommendation

Do not add heavy enterprise architecture. Add a thin application layer.

### Current Shape

```text
ui tab callback -> model/dataset/training module -> event logging
```

This keeps files small, but the UI owns too much orchestration.

### Recommended Shape

```text
ui tab callback
  -> controller
    -> workflow facade
      -> domain service / backend adapter
      -> event/tracking/security policy
```

Add these modules:

- `core/context.py`
  - `AppContext`
  - owns catalog, event bus, tracking client, local config, and security policy.
  - replaces direct use of the global `APP_STATE` in new code.

- `controllers/chat_controller.py`
  - validates chat input.
  - selects backend through an inference facade.
  - emits trace events.
  - returns UI-safe results.

- `controllers/field_notes_controller.py`
  - owns save/import/export correction workflows.
  - produces documentation-ready demo summaries.

- `controllers/dataset_controller.py`
  - owns local/HF dataset preview with path policy checks.

- `controllers/serving_controller.py`
  - owns llama.cpp, SGLang, vLLM, Ollama command/status planning.

- `workflows/correction_loop.py`
  - `CorrectionLoopFacade`
  - one golden path: import sample -> correct -> export JSONL -> evaluate -> trace.

- `security/policies.py`
  - URL allowlist for local backends.
  - path allowlist for public Space mode.
  - file size/type limits.
  - execution policy for future command-running features.

Keep the existing model services as adapters. Do not replace everything at once.

### Singleton Decision

The current `APP_STATE` singleton is tolerable for a local prototype. It is not ideal for a public
Space. Move toward:

- `AppContext` created in `build_app()`.
- Per-session UI state through Gradio `gr.State` where user-specific data matters.
- Global event log only for public-safe aggregate events.
- Tests that create isolated contexts.

## Roadmap V2

### P0 - Make It Shippable And Judgeable

- [ ] Choose one final judged track: Backyard AI is recommended.
- [ ] Rename the public product around the user outcome, not the infrastructure.
- [ ] Make Field Notes Correction the first and primary demo path.
- [ ] Add a "Judge Mode" tab or first-screen panel with sample data and exact steps.
- [ ] Generate Playwright screenshots into `assets/e2e/`.
- [ ] Add screenshots to README and `docs/HACKATHON_SUBMISSION.md`.
- [ ] Deploy a working Hugging Face Space.
- [ ] Add the Space URL to README and submission docs.
- [ ] Record a 60-90 second demo video.
- [ ] Publish or draft the social post with Space, GitHub, and demo video links.

Definition of done: a judge can open the Space and understand the product in under 30 seconds.

### P1 - Prove One Real Small-Model Backend

- [ ] Pick the final backend for the demo: LM Studio today, llama.cpp if badge is achievable.
- [ ] Run one verified local generation path end-to-end.
- [ ] Capture model ID, parameter count, runtime, hardware, and latency.
- [ ] Add a "Model Evidence" section to README.
- [ ] Add a tiny benchmark: response latency, tokens/sec if available, and sample output quality.
- [ ] If using OpenBMB, show exactly where the OpenBMB model is used.
- [ ] If claiming Llama Champion, install llama.cpp, select GGUF, and verify real text generation.

Definition of done: the app has one undeniable real-model path, not just planners.

### P2 - Make The Field Notes Workflow Real

- [ ] Create a small public-safe example dataset.
- [ ] Add sample OCR predictions or sample model mistakes.
- [ ] Build "import -> correct -> export -> evaluate" as one guided workflow.
- [ ] Add before/after examples to README.
- [ ] Export corrected JSONL and trace data.
- [ ] Publish public-safe traces or dataset to Hugging Face.
- [ ] Write the Field Notes report/blog.

Definition of done: the app demonstrates measurable improvement from human corrections.

### P3 - Simplify The UI For Judges

- [ ] Add a first-screen guided demo panel.
- [ ] Move advanced backend tabs behind an "Advanced setup" accordion or tab group.
- [ ] Add concise status badges: local-first, model size, backend, Space-ready, trace-ready.
- [ ] Add visual hierarchy and screenshots.
- [ ] Use a small custom CSS theme that fits the product story.
- [ ] Keep default Gradio controls where useful, but avoid an unstyled toolbox feel.

Definition of done: the app feels intentional, not like a pile of tabs.

### P4 - Add Controller And Facade Layer

- [ ] Add `core/context.py` with injectable `AppContext`.
- [ ] Add `security/policies.py` with path and URL checks.
- [ ] Add `controllers/field_notes_controller.py`.
- [ ] Add `controllers/dataset_controller.py`.
- [ ] Add `controllers/chat_controller.py`.
- [ ] Add `controllers/serving_controller.py`.
- [ ] Add `workflows/correction_loop.py`.
- [ ] Refactor one tab at a time to call controllers instead of raw service modules.
- [ ] Keep old behavior covered by tests during each refactor.

Definition of done: UI callbacks become thin and domain workflows are testable without Gradio.

### P5 - Security Hardening

- [ ] Add public/local mode setting.
- [ ] In public Space mode, restrict backend URLs to an allowlist or disable arbitrary URL checks.
- [ ] Restrict file paths to repo-local safe directories.
- [ ] Add max file size and supported extension checks.
- [ ] Add tests for path traversal attempts.
- [ ] Add tests for SSRF-style URLs.
- [ ] Add tests for malformed CSV/JSONL/OCR inputs.
- [ ] Add clear privacy labels for traces and field notes.
- [ ] Ensure demo data contains no secrets, private prompts, or hidden reasoning.

Definition of done: the public Space cannot be used to probe internal URLs or arbitrary files.

### P6 - Real Integration And E2E Tests

- [ ] Install Node.js/npm.
- [ ] Run `npm install`, `npm run e2e:install`, and `npm run e2e`.
- [ ] Commit generated screenshot assets that are safe for docs.
- [ ] Add Playwright to CI if Node is available.
- [ ] Add one live-backend integration profile for LM Studio/OpenAI-compatible.
- [ ] Add optional integration profiles for llama.cpp, Ollama, SGLang, and vLLM.
- [ ] Add visual smoke checks for mobile and desktop.

Definition of done: the demo path is proven in a browser and documented with images.

### P7 - SOTA Stretch

- [ ] Fine-tune or publish a tiny adapter if time allows.
- [ ] Publish a public-safe correction dataset or trace dataset.
- [ ] Add a simple reward/evaluation leaderboard for base vs corrected/fine-tuned outputs.
- [ ] Add a model card or dataset card with limitations.
- [ ] Show why the small model is good enough: privacy, latency, cost, offline use, or local control.
- [ ] If using llama.cpp, publish exact GGUF/runtime instructions.

Definition of done: the project has a defensible small-model contribution, not just a UI.

## What To Cut

Cut or hide these from the judged first impression unless they support the demo:

- vLLM tab.
- SGLang controls.
- Training execution promises.
- VINDEX boundary.
- Desk-Pet backlog.
- MiniCPM-o audio backlog.
- Generic agent mode, unless it directly produces the public trace badge.

Keep them in docs as extension points, but do not make the judge click through them first.

## Better Tool Vision

A better tool is not "all PRD features implemented." A better tool is:

1. **Specific**
   - Helps one person correct and turn messy field observations into useful training/eval data.

2. **Local-first**
   - Runs with a small local model and explains why that matters.

3. **Evidence-driven**
   - Shows before/after outputs, corrected examples, traces, and screenshots.

4. **Safe**
   - Does not leak private notes, probe arbitrary URLs, or execute commands unexpectedly.

5. **Small but complete**
   - One polished path beats ten partial tabs.

## Brutal Final Advice

If the deadline is close, do not chase full PRD completion. The PRD is too broad for a winning
hackathon submission. Win by making one real, visual, end-to-end story undeniable.

Priority order:

1. Space URL.
2. One real model path.
3. Field Notes golden workflow.
4. Screenshots and demo video.
5. README story with evidence.
6. Security hardening for public mode.
7. Controller/facade cleanup.

Everything else is secondary.
