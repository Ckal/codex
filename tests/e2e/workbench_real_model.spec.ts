import { expect, type Locator, type Page, test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

type StoryStep = {
  title: string;
  description: string;
  fileName: string;
};

const assetsDir = path.join(process.cwd(), "assets", "e2e", "workbench");
const sampleDataset = path.join(process.cwd(), "data", "public_sample_field_notes.csv");

test.describe("OpenBMB Workbench real-backend journey", () => {
  test("captures the Workbench with real backend choices and public sample data", async ({ page }) => {
    test.slow();
    mkdirSync(assetsDir, { recursive: true });

    await page.goto("/");
    await expect(page.getByRole("heading", { name: /OpenBMB Local AI Workbench/i })).toBeVisible();

    const steps: StoryStep[] = [];
    await capture(page, "01-workbench-home.png", "Workbench Home", "The Workbench opens with real-backend setup visible.", steps);

    await selectTab(page, "Chat");
    await expect(page.getByLabel("Backend").filter({ visible: true }).first()).not.toContainText("placeholder");
    await selectDropdown(page, "Backend", "llama-cpp-python");
    await fillVisibleTextbox(page, 0, "You are a concise local model assistant.");
    await fillVisibleTextbox(page, 1, "Reply in one short sentence that local GGUF inference is working.");
    await clickButton(page, "Run");
    await expect(page.getByText(/Chat response generated/i)).toBeVisible({ timeout: 180_000 });
    const chatResponse = page.getByRole("textbox").filter({ visible: true }).nth(2);
    await expect(chatResponse).toHaveValue(/.+/, { timeout: 10_000 });
    await expect(chatResponse).not.toHaveValue(/\[(llama-cpp-python unavailable|llama\.cpp unavailable|Transformers unavailable)\]/i);
    await capture(page, "02-chat-real-backend-response.png", "Real Text Response", "Chat generated a visible answer through the local GGUF llama-cpp-python backend.", steps);

    await selectTab(page, "Vision");
    await expect(page.getByLabel("Backend").filter({ visible: true }).first()).not.toContainText("placeholder");
    await selectDropdown(page, "Backend", "transformers");
    await fillByLabel(page, "Prompt", "Describe the plant image for a field note.");
    await capture(page, "03-vision-real-backend.png", "Real Vision Backend", "Vision is configured for a real MiniCPM-V Transformers backend.", steps);

    await selectTab(page, "Dataset");
    await fillByLabel(page, "Dataset ID or local path", sampleDataset);
    await fillByLabel(page, "Split", "train");
    await clickButton(page, "Preview dataset");
    await expect(page.getByText(/Status: Local dataset preview loaded\./i)).toBeVisible();
    await capture(page, "04-public-sample-data.png", "Public Sample Data", "The Workbench previews committed public-safe sample corrections.", steps);

    await selectTab(page, "Status");
    await expect(page.getByText(/Model and backend status/i)).toBeVisible();
    await capture(page, "05-backend-status.png", "Backend Status", "Status lists OpenBMB models and real local backend readiness.", steps);

    writeFileSync(path.join(assetsDir, "user-story.md"), renderStoryMarkdown(steps));
  });
});

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

async function selectTab(page: Page, name: string): Promise<void> {
  const tab = page.getByRole("tab", { name: exactText(name) });
  await expect(tab).toBeVisible();
  await tab.click();
  await expect(tab).toHaveAttribute("aria-selected", "true");
}

async function selectDropdown(page: Page, label: string, value: string): Promise<void> {
  const field = page.getByLabel(exactText(label)).filter({ visible: true }).first();
  await field.selectOption(value).catch(async () => {
    await field.click();
    await page.getByRole("option", { name: exactText(value) }).click();
  });
}

async function fillByLabel(page: Page, label: string, value: string): Promise<void> {
  const field = page.getByLabel(label).filter({ visible: true }).first();
  await expect(field, `Expected editable field labelled "${label}"`).toBeVisible();
  await field.fill(value);
}

async function fillVisibleTextbox(page: Page, index: number, value: string): Promise<void> {
  const field = page.getByRole("textbox").filter({ visible: true }).nth(index);
  await expect(field, `Expected visible textbox at index ${index}`).toBeVisible();
  await field.fill(value);
}

async function clickButton(page: Page, name: string): Promise<void> {
  const button = visibleButton(page, name);
  await expect(button, `Expected button "${name}"`).toBeVisible();
  await button.click();
}

function visibleButton(page: Page, name: string): Locator {
  return page.getByRole("button", { name: exactText(name) }).filter({ visible: true }).first();
}

function exactText(text: string): RegExp {
  return new RegExp(`^${escapeRegex(text)}$`, "i");
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function renderStoryMarkdown(steps: StoryStep[]): string {
  const lines = [
    "# Workbench Real-Backend Screenshots",
    "",
    "These screenshots are generated without mock responses. The chat screenshot runs the local GGUF llama-cpp-python backend and captures the model response.",
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
