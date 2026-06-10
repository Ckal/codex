import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.E2E_BASE_URL ?? "http://127.0.0.1:7861";
const defaultCommand = "python plant_space_app.py";
const appCommand = process.env.E2E_APP_COMMAND ?? defaultCommand;
const webServer = process.env.E2E_NO_WEBSERVER === "1"
  ? undefined
  : {
      command: appCommand,
      url: baseURL,
      reuseExistingServer: true,
      timeout: 180_000,
      env: {
        ...process.env,
        PORT: "7861",
        PYTHONUTF8: "1",
        WORKBENCH_DEPLOYMENT: "space"
      }
    };

export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: /plant_.*\.spec\.ts/,
  timeout: 700_000,
  expect: {
    timeout: 30_000
  },
  fullyParallel: false,
  reporter: [["list"], ["html", { outputFolder: "assets/e2e/plant-playwright-report" }]],
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
