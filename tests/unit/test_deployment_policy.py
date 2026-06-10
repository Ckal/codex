from __future__ import annotations

import unittest

from core.deployment import (
    DeploymentPolicy,
    default_backend_for_policy,
    deployment_mode,
    ensure_backend_allowed,
    ensure_demo_mode_allowed,
    filter_backends_for_policy,
)


class DeploymentPolicyTest(unittest.TestCase):
    def test_default_deployment_mode_is_local(self) -> None:
        self.assertEqual(deployment_mode(""), "local")

    def test_rejects_unknown_deployment_mode(self) -> None:
        with self.assertRaises(ValueError):
            deployment_mode("prod")

    def test_space_mode_filters_placeholder_backend(self) -> None:
        policy = DeploymentPolicy("space")

        backends = filter_backends_for_policy(["placeholder", "transformers"], policy)

        self.assertEqual(backends, ["transformers"])

    def test_space_mode_refuses_placeholder_backend(self) -> None:
        with self.assertRaises(ValueError):
            ensure_backend_allowed("placeholder", DeploymentPolicy("space"))

    def test_space_mode_refuses_demo_mode(self) -> None:
        with self.assertRaises(ValueError):
            ensure_demo_mode_allowed(DeploymentPolicy("space"))

    def test_space_mode_prefers_real_backend(self) -> None:
        backend = default_backend_for_policy(
            ["placeholder", "transformers"],
            "placeholder",
            DeploymentPolicy("space"),
        )

        self.assertEqual(backend, "transformers")


if __name__ == "__main__":
    unittest.main()
