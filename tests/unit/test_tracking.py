from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.app_state import AppState
from core.events import Event, EventType
from tracking.trackio_client import (
    TrackingClient,
    TrackingConfig,
    export_traces,
    load_tracking_config,
    read_trace_rows,
    tracking_status_dict,
)
from training.evaluation import default_prompt_cases, evaluate_responses, log_eval_metrics


class TrackingTest(unittest.TestCase):
    def test_loads_tracking_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training.yaml"
            path.write_text(
                "trackio:\n"
                "  enabled: true\n"
                "  project: demo\n"
                "  local_path: data/demo.jsonl\n",
                encoding="utf-8",
            )

            config = load_tracking_config(path)

            self.assertTrue(config.enabled)
            self.assertEqual(config.project, "demo")
            self.assertEqual(config.local_path, "data/demo.jsonl")

    def test_logs_events_to_local_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            client = TrackingClient(TrackingConfig(local_path=str(path)))

            saved = client.log("dataset_loaded", {"rows": 2})

            self.assertEqual(saved, path)
            self.assertEqual(read_trace_rows(path)[0]["payload"]["rows"], 2)

    def test_app_state_logs_events_to_tracking_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            client = TrackingClient(TrackingConfig(local_path=str(path)))
            state = AppState(tracking_client=client)

            state.emit(Event(EventType.INFERENCE_REQUEST, {"model_id": "demo"}))

            self.assertEqual(read_trace_rows(path)[0]["event"], "inference_request")

    def test_exports_traces_even_when_source_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "exports" / "traces.jsonl"

            exported = export_traces(Path(tmp) / "missing.jsonl", output)

            self.assertEqual(exported, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "")

    def test_logs_eval_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            report = evaluate_responses(default_prompt_cases(), ["field note"])
            client = TrackingClient(TrackingConfig(local_path=str(path)))

            saved = log_eval_metrics(report, client)

            self.assertEqual(saved, path)
            self.assertEqual(read_trace_rows(path)[0]["event"], "training_metrics")

    def test_tracking_status_dict(self) -> None:
        status = tracking_status_dict(TrackingClient(TrackingConfig(project="demo")))

        self.assertEqual(status["project"], "demo")
        self.assertIn(status["mode"], {"local_jsonl", "trackio"})


if __name__ == "__main__":
    unittest.main()
