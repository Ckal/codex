import { expect, type Page, test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

type StoryStep = {
  title: string;
  description: string;
  fileName: string;
};

const assetsDir = path.join(process.cwd(), "assets", "e2e", "plant");
const sampleImage = path.join(process.cwd(), "assets", "plant_sample.jpg");
const runRealInference = process.env.RUN_REAL_MODEL_E2E === "1";

test.describe("Plant Identification Tool real-model journey", () => {
  test("captures the Plant tool in OpenBMB mode", async ({ page }) => {
    test.slow();
    mkdirSync(assetsDir, { recursive: true });
    const breadcrumbs: string[] = [];
    const mark = (step: string): void => {
      breadcrumbs.push(`${new Date().toISOString()} ${step}`);
      writeFileSync(path.join(assetsDir, "real-inference-breadcrumbs.txt"), `${breadcrumbs.join("\n")}\n`);
    };

    mark("goto");
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { name: /Plant Discovery/i })).toBeVisible();
    await expect(page.getByText(/Local-first reference app/i)).toBeVisible();

    const steps: StoryStep[] = [];
    await capture(page, "01-plant-home.png", "Plant Tool Home", "The Plant Space opens in OpenBMB real-model mode.", steps);
    mark("home-captured");

    await selectTab(page, "Identify");
    await page.getByLabel("Analysis mode").selectOption("standard").catch(async () => {
      await page.getByLabel("Analysis mode").click();
      await page.getByRole("option", { name: /^standard$/i }).click();
    });

    if (runRealInference) {
      mark("upload-start");
      await page.locator("input[type='file']").first().setInputFiles(sampleImage);
      mark("upload-done");
      await page.getByRole("button", { name: /^Identify plant$/i }).click();
      mark("identify-clicked");
      await expect(
        page.getByRole("heading", { name: /Lady's Mantle|Model unavailable/i }).first()
      ).toBeVisible({ timeout: 180_000 });
      mark("result-visible");
    }
    await capture(page, "02-identify-real-model.png", "Real Model Identify", "A public-safe image is ready for MiniCPM-V identification.", steps);

    await selectTab(page, "Corrections");
    await expect(page.getByRole("button", { name: /^Export training JSONL$/i })).toBeVisible();
    await capture(page, "03-corrections-export.png", "Corrections Export", "Corrections export remains available for real model outputs.", steps);

    await selectTab(page, "Stats");
    await expect(page.getByRole("button", { name: /^Refresh stats$/i })).toBeVisible();
    await page.getByRole("button", { name: /^Refresh stats$/i }).click();
    await capture(page, "04-plant-stats.png", "Plant Stats", "The Plant app reports species and correction counts.", steps);

    writeFileSync(path.join(assetsDir, "user-story.md"), renderStoryMarkdown(steps));
  });
});

async function selectTab(page: Page, name: string): Promise<void> {
  const tab = page.getByRole("tab", { name: new RegExp(`^${name}$`, "i") });
  await expect(tab).toBeVisible();
  await tab.click();
  await expect(tab).toHaveAttribute("aria-selected", "true");
}

async function capture(
  page: Page,
  fileName: string,
  title: string,
  description: string,
  steps: StoryStep[],
): Promise<void> {
  await page.screenshot({ path: path.join(assetsDir, fileName), fullPage: true });
  steps.push({ title, description, fileName });
}

function renderStoryMarkdown(steps: StoryStep[]): string {
  const lines = [
    "# Plant Identification Tool Screenshots",
    "",
    "These screenshots use the OpenBMB-mode Plant Space entrypoint. Real model inference is executed only when `RUN_REAL_MODEL_E2E=1` is set.",
    ""
  ];

  for (const [index, step] of steps.entries()) {
    lines.push(
      `## ${index + 1}. ${step.title}`,
      "",
      step.description,
      "",
      `![${step.title}](./${step.fileName})`,
      ""
    );
  }

  return `${lines.join("\n")}\n`;
}
