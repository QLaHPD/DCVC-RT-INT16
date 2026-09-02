from __future__ import annotations

import unittest

from src.cli.dashboard import format_bytes, render_dashboard_lines


class DashboardRenderingTests(unittest.TestCase):
    def setUp(self):
        self.channels = [
            {
                "channel_id": "CHANNEL_A",
                "status": "encoding",
                "processed": 2,
                "pending": 5,
                "already_done": 10,
                "failures": 0,
                "active": 2,
                "total": 15,
                "reclaimable_bytes": 0,
                "last_error": "",
            },
            {
                "channel_id": "CHANNEL_B",
                "status": "awaiting-approval",
                "processed": 3,
                "pending": 3,
                "already_done": 0,
                "failures": 0,
                "active": 0,
                "total": 3,
                "reclaimable_bytes": 4_700_000_000,
                "last_error": "",
            },
        ]

    def render(self, view: str):
        return render_dashboard_lines(
            view=view,
            total=18,
            done=15,
            elapsed=123,
            workers={0: "video 100f 30.0fps Aud:on", 1: "[1] idle"},
            channels=self.channels,
            events=["CHANNEL_B: awaiting deletion approval"],
            width=120,
            height=20,
            free_bytes=25_000_000_000,
        )

    def test_every_view_renders_with_switching_help(self):
        for view in ("workers", "channels", "approvals", "events"):
            output = "\n".join(self.render(view))
            self.assertIn(f"View: {view}", output)
            self.assertIn("[w]orkers", output)

    def test_approval_view_reports_exact_candidate(self):
        output = "\n".join(self.render("approvals"))
        self.assertIn("CHANNEL_B", output)
        self.assertIn("awaiting-approval", output)
        self.assertIn("4.4GiB", output)

    def test_byte_format_is_binary_and_human_readable(self):
        self.assertEqual(format_bytes(0), "0B")
        self.assertEqual(format_bytes(1024), "1.0KiB")


if __name__ == "__main__":
    unittest.main()
