import { expect, type Locator, type Page, test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

type StoryStep = {
  tab: string;
  title: string;
  description: string;
  fileName: string;
};

type TabStory = {
  name: string;
  description: string;
  act?: (page: Page) => Promise<void>;
};

const assetsDir = path.join(process.cwd(), "assets", "e2e");
const dataDir = path.join(process.cwd(), "data");

const tabStories: TabStory[] = [
  {
    name: "Chat",
    description: "A new user sends a safe placeholder prompt and sees a local response area update.",
    act: async (page) => {
      await fillByLabel(page, "System prompt", "You are a concise local workbench assistant.");
      await fillByLabel(page, "Prompt", "Explain why this project is useful for a hackathon judge.");
      await clickButton(page, "Run");
      await expect(page.getByText(/Chat response generated|local workbench|placeholder/i)).toBeVisible();
    }
  },
  {
    name: "Vision",
    description: "The user reviews the vision workflow and captures the local-first image prompt surface."
  },
  {
    name: "Dataset",
    description: "The user previews a small local CSV dataset and sees schema/statistics-ready output.",
    act: async (page) => {
      const demoCsv = path.join(dataDir, "e2e_demo_dataset.csv");
      mkdirSync(dataDir, { recursive: true });
      writeFileSync(demoCsv, "prompt,response\nWhat is local-first?,Run small models locally\n");
      await fillByLabel(page, "Dataset ID or local path", demoCsv);
      await fillByLabel(page, "Split", "train");
      await clickButton(page, "Preview dataset");
      await expect(page.getByText(/Local dataset preview loaded|prompt|response/i)).toBeVisible();
    }
  },
  {
    name: "Train",
    description: "The user prepares a non-executing LoRA plan without downloading weights or starting training.",
    act: async (page) => {
      await fillByLabel(page, "Training dataset", "data/e2e_demo_dataset.csv");
      await clickButton(page, "Prepare training plan");
      await expect(page.getByText(/dry run|LoRA|checkpoint|training/i)).toBeVisible();
    }
  },
  {
    name: "vLLM",
    description: "The user prepares an explicit vLLM serve command and can document the serving plan.",
    act: async (page) => {
      await clickButton(page, "Prepare vLLM command");
      await expect(page.getByText(/vllm|serve|chat\/completions/i)).toBeVisible();
    }
  },
  {
    name: "Export",
    description: "The user plans GGUF export and quantization commands before running any conversion.",
    act: async (page) => {
      await clickButton(page, "Prepare export plan");
      await expect(page.getByText(/quantize|GGUF|export/i)).toBeVisible();
    }
  },
  {
    name: "Field Notes",
    description: "The user saves a correction that can later become training data.",
    act: async (page) => {
      await fillByLabel(page, "Prompt", "Identify this field note.");
      await fillByLabel(page, "Model response", "The answer needs review.");
      await fillByLabel(page, "Human correction", "Corrected answer for the demo dataset.");
      await fillByLabel(page, "Tags", "demo,e2e,correction");
      await clickButton(page, "Save field note");
      await expect(page.getByText(/Saved to/i)).toBeVisible();
    }
  },
  {
    name: "Traces",
    description: "The user opens local event traces to inspect what the app recorded."
  },
  {
    name: "Agent",
    description: "The user drafts a local research-plan-verify trace with safety boundaries.",
    act: async (page) => {
      await fillByLabel(page, "Agent task", "Plan a README screenshot section for the hackathon demo.");
      await clickButton(page, "Draft agent trace");
      await expect(page.getByText(/research|plan|verify|trace/i)).toBeVisible();
    }
  },
  {
    name: "Status",
    description: "The user checks model configuration, backend readiness, and local setup commands."
  }
];

test.describe("OpenBMB Local AI Workbench documentation journey", () => {
  test("walks all major tabs and captures user-story screenshots", async ({ page }) => {
    test.slow();
    mkdirSync(assetsDir, { recursive: true });

    await page.goto("/");
    await expect(page.getByRole("heading", { name: /OpenBMB Local AI Workbench/i })).toBeVisible();
    await page.screenshot({
      path: path.join(assetsDir, "00-home-app-shell.png"),
      fullPage: true
    });

    const discoveredTabs = await page.getByRole("tab").allTextContents();
    const normalizedTabs = discoveredTabs.map((tab) => tab.trim()).filter(Boolean);
    const missingTabs = tabStories
      .map((story) => story.name)
      .filter((name) => !normalizedTabs.includes(name));
    expect(missingTabs, `Missing major tabs: ${missingTabs.join(", ")}`).toEqual([]);

    const steps: StoryStep[] = [
      {
        tab: "App",
        title: "App Shell",
        description: "The Gradio workbench opens with the project title and all major tabs visible.",
        fileName: "00-home-app-shell.png"
      }
    ];

    for (const [index, story] of tabStories.entries()) {
      await test.step(`User story tab: ${story.name}`, async () => {
        await selectTab(page, story.name);
        if (story.act) {
          await story.act(page);
        }
        const fileName = `${String(index + 1).padStart(2, "0")}-${slug(story.name)}.png`;
        await page.screenshot({ path: path.join(assetsDir, fileName), fullPage: true });
        steps.push({
          tab: story.name,
          title: `${story.name} Tab`,
          description: story.description,
          fileName
        });
      });
    }

    writeFileSync(path.join(assetsDir, "user-story.md"), renderStoryMarkdown(steps));
  });
});

async function selectTab(page: Page, name: string): Promise<void> {
  const tab = page.getByRole("tab", { name: exactText(name) });
  await expect(tab).toBeVisible();
  await tab.click();
  await expect(tab).toHaveAttribute("aria-selected", "true");
  await page.waitForTimeout(350);
}

async function fillByLabel(page: Page, label: string, value: string): Promise<void> {
  const field = page.getByLabel(label).filter({ visible: true }).first();
  await expect(field, `Expected editable field labelled "${label}"`).toBeVisible();
  await field.fill(value);
}

async function clickButton(page: Page, name: string): Promise<void> {
  const button = visibleButton(page, name);
  await expect(button, `Expected button "${name}"`).toBeVisible();
  await button.click();
  await page.waitForTimeout(700);
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

function slug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function renderStoryMarkdown(steps: StoryStep[]): string {
  const lines = [
    "# E2E User Story Screenshots",
    "",
    "These screenshots are generated by `npm run e2e` while the Playwright user-story test walks the Gradio app.",
    "Use them in the README, hackathon submission, or demo script as documentation-ready proof of the UI flow.",
    ""
  ];

  for (const [index, step] of steps.entries()) {
    lines.push(
      `## ${index + 1}. ${step.title}`,
      "",
      `**Tab:** ${step.tab}`,
      "",
      step.description,
      "",
      `![${step.title}](./${step.fileName})`,
      ""
    );
  }

  return `${lines.join("\n")}\n`;
}
