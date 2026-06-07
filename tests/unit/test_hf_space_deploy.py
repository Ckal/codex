from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from deployment.hf_space import (
    create_space_commands,
    has_space_remote,
    missing_required_files,
    readme_has_space_metadata,
    space_build_status,
)


class HuggingFaceSpaceDeployTest(unittest.TestCase):
    def test_creates_manual_space_commands(self) -> None:
        commands = create_space_commands("demo-user", "demo-space")

        self.assertEqual(commands[0], "huggingface-cli login")
        self.assertIn("repo create demo-space --type space --space-sdk gradio", commands[1])
        self.assertIn("https://huggingface.co/spaces/demo-user/demo-space", commands[2])
        self.assertEqual(commands[3], "git push space main")

    def test_requires_hf_user(self) -> None:
        with self.assertRaises(ValueError):
            create_space_commands("")

    def test_detects_space_remote(self) -> None:
        remotes = (
            "origin\thttps://github.com/Ckal/codex.git (fetch)\n"
            "space\thttps://huggingface.co/spaces/u/s (push)"
        )

        self.assertTrue(has_space_remote(remotes))

    def test_validates_space_build_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            for path in [
                "app.py",
                "requirements.txt",
                "config/models.yaml",
                "config/training.yaml",
            ]:
                (root / path).write_text("", encoding="utf-8")
            (root / "README.md").write_text(
                "---\ntitle: Demo\nsdk: gradio\napp_file: app.py\n---\n",
                encoding="utf-8",
            )

            status = space_build_status(root, "space\thttps://huggingface.co/spaces/u/s (push)")

            self.assertEqual(missing_required_files(root), [])
            self.assertTrue(readme_has_space_metadata(root))
            self.assertTrue(status.required_files_present)
            self.assertTrue(status.has_space_remote)

    def test_plan_script_runs_from_script_path(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/plan_hf_space.py", "--user", "demo-user"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Commands to run manually", result.stdout)
        self.assertIn("huggingface-cli login", result.stdout)


if __name__ == "__main__":
    unittest.main()
