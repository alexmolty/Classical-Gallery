import base64
import copy
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

from color_features import analyze_rgba
from publish_colors import publish_colors

HERE = pathlib.Path(__file__).resolve()
UPSTREAM = HERE.parents[1]
if not (UPSTREAM / "publish_index.py").exists():
    UPSTREAM = HERE.parents[2] / "tmp/color-index-upstream/tools/art-index"


@unittest.skipUnless((UPSTREAM / "publish_index.py").exists(), "Upstream publisher checkout is required")
class PublicationTests(unittest.TestCase):
    def test_real_v2_publication_preserves_vector_bytes_and_v1(self):
        sys.path.insert(0, str(UPSTREAM))
        from publish_index import load_previous, publish_index
        vectors = bytes([1]) * 1792
        payload = {"version": 2, "indexedCount": 2, "generatedAt": "2026-11-23T00:00:00Z",
                   "minimumScore": .65, "clip": {"model": "fixture"}, "dino": {"model": "fixture"},
                   "records": [{"id": "met-1", "sourceMuseum": "met"}, {"id": "met-2", "sourceMuseum": "met"}],
                   "coverage": {"bySource": {"met": {"visual": 2}}},
                   "embeddings": {"encoding": "int8-rowwise-clip512-dino384-v1", "ids": ["met-1", "met-2"],
                                  "data": base64.b64encode(vectors).decode(), "sha256": hashlib.sha256(vectors).hexdigest()}}
        original = copy.deepcopy(payload["embeddings"])
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            work = root / "work"
            (work / "previews").mkdir(parents=True)
            (work / "preprocessor_config.json").write_text(json.dumps({"image_mean": [.5, .5, .5]}))
            (work / "previews/met-1.png").write_bytes(b"fixture-pixels")
            for target in ("desktop/v2", "v2"):
                publish_index(root / target, copy.deepcopy(payload))
            (root / "v1").mkdir()
            (root / "v1/manifest.json").write_text("untouched legacy publication")
            descriptor = analyze_rgba(1, 1, bytes([255, 0, 0, 255]))
            report = publish_colors(root, work, decoder=lambda data, padding: descriptor)
            self.assertEqual((report["indexedCount"], report["networkRequests"], report["inferenceRuns"]), (1, 0, 0))
            for target in ("desktop/v2", "v2"):
                manifest, result = load_previous(root / target)
                self.assertEqual(result["embeddings"], original)
                self.assertEqual(result["records"][0]["colorFeatures"], descriptor)
                self.assertNotIn("colorFeatures", result["records"][1])
                self.assertEqual(manifest["coverage"]["colorIndexed"], 1)
            self.assertEqual((root / "v1/manifest.json").read_text(), "untouched legacy publication")
            sidecar = json.loads((root / "colors/v1/color-index.json").read_text())
            self.assertEqual(set(sidecar["records"]), {"met-1"})


if __name__ == "__main__":
    unittest.main()
