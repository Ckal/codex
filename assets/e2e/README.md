# E2E Documentation Assets

Run the Playwright user-story test to generate screenshots for the README and hackathon docs:

```powershell
npm install
npm run e2e:install
npm run e2e
```

The test starts or reuses the local Gradio app at `http://127.0.0.1:7860`, walks all major tabs,
executes safe local actions, and writes screenshots plus `user-story.md` into this folder.

To record or edit the flow manually:

```powershell
npm run e2e:record
```

Generated images are intentionally stored under `assets/e2e/` so they can be linked from
`README.md`, `docs/HACKATHON_SUBMISSION.md`, and demo notes.
