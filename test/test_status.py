"""Step 4: test service status."""

import unittest

import main


def health_status() -> dict[str, str]:
    """Read the service health response."""
    return main.health()


class StatusTests(unittest.TestCase):
    """Check that the service reports a healthy status."""

    def test_health_status(self):
        self.assertEqual(health_status(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
