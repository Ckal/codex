import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.E2E_BASE_URL ?? "http://127.0.0.1:7860";
const defaultCommand = "python app.py";
const appCommand = process.env.E2E_APP_COMMAND ?? defaultCommand;
const webServer = process.env.E2E_NO_WEBSERVER === "1"
  ? undefined
  : {
      command: appCommand,
      url: baseURL,
      reuseExistingServer: true,
      timeout: 120_000,
      env: {
        ...process.env,
        PYTHONUTF8: "1",
        WORKBENCH_DEPLOYMENT: process.env.WORKBENCH_DEPLOYMENT ?? "space"
      }
    };

export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: /workbench_.*\.spec\.ts/,
  timeout: 60_000,
  expect: {
    timeout: 10_000
  },
  fullyParallel: false,
  reporter: [["list"], ["html", { outputFolder: "assets/e2e/playwright-report" }]],
  use: {
    baseURL,
    trace: "retain-on-failure",
    video: "retain-on-failure",
    screenshot: "only-on-failure"
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] }
    }
  ],
  webServer
});
