from __future__ import annotations

import unittest

from models.model_catalog import load_model_catalog, model_choices, model_summary, validate_catalog


class ModelCatalogTest(unittest.TestCase):
    def test_loads_prd_models_from_config(self) -> None:
        catalog = load_model_catalog("config/models.yaml")

        self.assertIn("minicpm5_1b", catalog)
        self.assertIn("minicpm41_8b", catalog)
        self.assertIn("minicpm_v46", catalog)
        self.assertIn("minicpm_v46_thinking", catalog)

    def test_model_choices_filter_by_type(self) -> None:
        catalog = load_model_catalog("config/models.yaml")

        text_models = model_choices(catalog, "text")
        vision_models = model_choices(catalog, "vision")

        self.assertIn("minicpm5_1b", text_models)
        self.assertIn("minicpm_v46", vision_models)
        self.assertNotIn("minicpm_v46", text_models)

    def test_summary_contains_hackathon_compliance_fields(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        summary = model_summary(catalog["minicpm5_1b"])

        self.assertEqual(summary["hf_id"], "openbmb/MiniCPM5-1B")
        self.assertLessEqual(summary["parameters_b"], 32)
        self.assertIn("gguf", summary)
        self.assertIn("backend_capabilities", summary)
        self.assertIn("llama.cpp", summary["backend_capabilities"])

    def test_all_models_have_backend_capability_metadata(self) -> None:
        catalog = load_model_catalog("config/models.yaml")

        for model in catalog.values():
            self.assertIn(model.backend, model.backend_capabilities)
            self.assertTrue(model.backend_capabilities[model.backend])

    def test_catalog_warns_that_placeholder_backends_are_not_real_inference(self) -> None:
        catalog = load_model_catalog("config/models.yaml")
        warnings = validate_catalog(catalog)

        self.assertTrue(any("placeholder backend" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
