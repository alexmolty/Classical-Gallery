import copy
import json
import pathlib
import tempfile
import unittest
import urllib.error

from backfill_colors import backfill
from color_features import analyze_rgba


RED = analyze_rgba(1, 1, bytes([255, 0, 0, 255]))
BLUE = analyze_rgba(1, 1, bytes([0, 0, 255, 255]))


class BackfillTests(unittest.TestCase):
    def test_offline_resume_invalidates_changed_pixels_preserves_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "met-1.png").write_bytes(b"red")
            payload = {"records": [{"id": "met-1"}, {"id": "met-2"}], "embeddings": {"data": "unchanged", "sha256": "same"}}
            before = copy.deepcopy(payload)
            decoder = lambda data, legacy: RED if data == b"red" else BLUE
            fail_fetch = lambda url: self.fail("Offline mode touched network")
            options = {"decoder": decoder, "fetcher": fail_fetch}
            sidecar, report = backfill(payload, [root], root / "state.json", **options)
            self.assertEqual((sidecar["indexedCount"], report["pendingCount"], report["networkRequests"]), (1, 1, 0))
            self.assertEqual(sidecar["records"]["met-1"], RED)
            _, report = backfill(payload, [root], root / "state.json", **options)
            self.assertEqual(report["reused"], 1)
            (root / "met-1.png").write_bytes(b"blue")
            sidecar, report = backfill(payload, [root], root / "state.json", **options)
            self.assertEqual(report["reused"], 0)
            self.assertEqual(sidecar["records"]["met-1"], BLUE)
            self.assertEqual(payload, before)

    def test_fetch_is_opt_in_serial_bounded_and_stops_on_throttle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            payload = {"records": [{"id": "met-1", "indexPreviewUrl": "https://images.metmuseum.org/a.jpg"},
                                    {"id": "met-2", "indexPreviewUrl": "https://images.metmuseum.org/b.jpg"}]}
            calls = []
            def throttle(url):
                calls.append(url)
                raise urllib.error.HTTPError(url, 429, "Slow down", {}, None)
            _, report = backfill(payload, [root], root / "state.json", fetch_missing=True, fetcher=throttle)
            self.assertEqual(len(calls), 1)
            self.assertEqual(report["pendingCount"], 2)
            self.assertEqual(report["inferenceRuns"], 0)
            _, report = backfill(payload, [root], root / "state.json", fetch_missing=True, fetch_limit=0, fetcher=throttle)
            self.assertEqual(len(calls), 1)
            self.assertEqual(report["networkRequests"], 0)


if __name__ == "__main__":
    unittest.main()
