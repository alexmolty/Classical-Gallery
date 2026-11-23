"""Enrich existing v2 publications from cached pixels without model inference."""
import argparse
import json
import pathlib
import sys

from backfill_colors import atomic, backfill
from color_features import validate


def publish_colors(destination, work, fetch_missing=False, fetch_limit=256, delay=1,
                   max_seconds=16200, decoder=None, fetcher=None):
    # Only the publisher is imported, never run.py or the inference builder.
    publisher_directory = pathlib.Path(__file__).resolve().parents[1]
    if (publisher_directory / "publish_index.py").is_file() and str(publisher_directory) not in sys.path:
        sys.path.insert(0, str(publisher_directory))
    from publish_index import load_previous, publish_index
    destination, work = pathlib.Path(destination), pathlib.Path(work)
    manifest, payload = load_previous(destination / "desktop/v2")
    if not payload or payload.get("version") != 2:
        raise ValueError("A verified desktop/v2 publication is required")
    config = work / "preprocessor_config.json"
    if not config.exists():
        raise ValueError("Cached DINO preprocessor config is required to remove model padding")
    padding = tuple(round(value * 255) for value in json.loads(config.read_text())["image_mean"])
    options = {"padding": padding, "fetch_missing": fetch_missing, "fetch_limit": fetch_limit,
               "delay": delay, "max_seconds": max_seconds, "decoder": decoder}
    if fetcher:
        options["fetcher"] = fetcher
    sidecar, report = backfill(payload, [work / "previews"], work / "color-state-v1.json", **options)
    atomic(destination / "colors/v1/color-index.json", sidecar)
    atomic(destination / "colors/v1/coverage.json", report)
    publications = []
    for relative in ("desktop/v2", "v2"):
        _, current = load_previous(destination / relative)
        if not current or current.get("version") != 2:
            continue
        original_vectors = json.dumps(current["embeddings"], sort_keys=True)
        by_source, count = {}, 0
        for art in current["records"]:
            features = sidecar["records"].get(str(art["id"]))
            if features:
                art["colorFeatures"] = features
            if validate(art.get("colorFeatures")):
                count += 1
                source = art.get("sourceMuseum", "unknown")
                by_source[source] = by_source.get(source, 0) + 1
        current["generatedAt"] = sidecar["generatedAt"]
        coverage = current.setdefault("coverage", {})
        coverage["colorIndexed"] = count
        coverage.setdefault("byCriterion", {})["color"] = count
        for source, group in coverage.get("bySource", {}).items():
            group["color"] = by_source.get(source, 0)
        if json.dumps(current["embeddings"], sort_keys=True) != original_vectors:
            raise RuntimeError("Color analysis modified image vectors")
        publications.append(publish_index(destination / relative, current))
    return {**report, "publications": publications}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-dir", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--fetch-limit", type=int, default=256)
    parser.add_argument("--delay", type=float, default=1)
    parser.add_argument("--max-seconds", type=float, default=16200)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    work = args.work_dir or root / "cache/work/ai-index"
    print(json.dumps(publish_colors(args.publish_dir, work, args.fetch_missing,
                                   args.fetch_limit, args.delay, args.max_seconds)))


if __name__ == "__main__":
    main()
