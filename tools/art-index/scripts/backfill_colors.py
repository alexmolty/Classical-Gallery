"""Color-only, resumable backfill. Offline by default; never imports AI models."""
import argparse
import datetime
import gzip
import hashlib
import io
import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from color_features import VERSION, analyze_image, analyze_legacy_preview, validate

SIGNATURE = "color-hsv12-s3-v3-neutral8-rgba64-v1"
MAX_BYTES = 4 * 1024 * 1024
HOSTS = {"images.metmuseum.org", "openaccess-cdn.clevelandart.org", "upload.wikimedia.org",
         "thumb.wikimedia.org", "api.nga.gov", "iiif.micr.io", "media.getty.edu", "iip.smk.dk",
         "kokoelma.kansallisgalleria.fi", "ids.si.edu"}


def atomic(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=True, separators=(",", ":")), encoding="utf8")
    os.replace(temporary, path)


def read_index(path):
    data = pathlib.Path(path).read_bytes()
    if str(path).endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as packed:
            data = packed.read(64 * 1024 * 1024 + 1)
    if len(data) > 64 * 1024 * 1024:
        raise ValueError("Index exceeds 64 MiB")
    text = data.decode("utf8")
    marker = "window.ARTWORK_SIMILARITY=Object.freeze("
    if marker in text:
        text = text.split(marker, 1)[1].rsplit(");", 1)[0]
    payload = json.loads(text)
    records = payload.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= 50000:
        raise ValueError("Invalid index records")
    ids = [str(art.get("id", "")) for art in records]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-zA-Z0-9-]{1,160}", id) for id in ids):
        raise ValueError("Invalid or duplicate artwork IDs")
    return payload


def source_key(art):
    source = (art.get("indexPreviewUrl") or "") + "|" + art.get("imageUrl", "") + "|" + str(art.get("metadataUpdatedAt", ""))
    return hashlib.sha256(source.encode()).hexdigest()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def fetch_preview(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("Preview URL is not an allowed HTTPS museum source")
    request = urllib.request.Request(url, headers={"User-Agent": "ClassicalGallery-ColorIndex/1.0"})
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=20) as response:
        if response.headers.get_content_type() not in ("image/jpeg", "image/png", "image/webp"):
            raise ValueError("Unexpected preview MIME type")
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Preview exceeds 4 MiB")
    return data


def backfill(payload, cache_directories, state_path, padding=None, fetch_missing=False,
             fetch_limit=256, delay=1, max_seconds=3600, decoder=None, fetcher=fetch_preview):
    """Return a sidecar and preserve all original vectors; one request at a time."""
    if not 0 <= fetch_limit <= 50000 or delay < 1 or max_seconds <= 0:
        raise ValueError("Invalid fetch budget")
    directories = [pathlib.Path(directory) for directory in cache_directories]
    signature = SIGNATURE + ":" + str(padding)
    state = json.loads(pathlib.Path(state_path).read_text(encoding="utf8")) if pathlib.Path(state_path).exists() else {}
    saved = state.get("records", {}) if state.get("signature") == signature else {}
    results, next_state, pending, failures = {}, dict(saved), [], []
    started, last_fetch = time.monotonic(), -float("inf")
    requests, reused, blocked_hosts = 0, 0, set()

    def decode(data, legacy):
        if decoder:
            return decoder(data, legacy)
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 48_000_000
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return analyze_legacy_preview(image, padding) if legacy and padding else analyze_image(image)

    for position, art in enumerate(payload["records"]):
        id, key = str(art["id"]), source_key(art)
        entry, features = saved.get(id, {}), art.get("colorFeatures")
        candidates = [directory / (id + extension) for directory in directories for extension in (".png", ".jpg", ".jpeg", ".webp")]
        cached = next((candidate for candidate in candidates if candidate.is_file()), None)
        if validate(features):
            results[id] = features
            continue
        same_pixels = not cached or cached.stat().st_size <= 32 * 1024 * 1024 and hashlib.sha256(cached.read_bytes()).hexdigest() == entry.get("sha256")
        if entry.get("sourceKey") == key and validate(entry.get("features")) and same_pixels:
            results[id] = entry["features"]
            next_state[id] = entry
            reused += 1
            continue
        if time.monotonic() - started >= max_seconds:
            pending.extend(str(row["id"]) for row in payload["records"][position:])
            break
        fetch_host = None
        try:
            data, legacy = None, False
            if cached:
                data = cached.read_bytes()
                if len(data) > 32 * 1024 * 1024:
                    raise ValueError("Cached pixels exceed 32 MiB")
                meta_path = cached.with_suffix(".json")
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf8"))
                    if meta.get("sourceKey") and meta["sourceKey"] != key:
                        data = None
                    elif meta.get("previewSha256") and hashlib.sha256(data).hexdigest() != meta["previewSha256"]:
                        raise ValueError("Cached preview checksum mismatch")
                legacy = cached.suffix.lower() == ".png" and padding is not None
            if data is None:
                if not fetch_missing or requests >= fetch_limit or entry.get("retryAt", 0) > time.time():
                    pending.append(id)
                    if entry.get("sourceKey") == key:
                        next_state[id] = entry
                    continue
                url = art.get("indexPreviewUrl")
                if not url:
                    raise ValueError("No preview URL; original image is not fetched implicitly")
                fetch_host = urllib.parse.urlsplit(url).hostname
                if fetch_host in blocked_hosts:
                    pending.append(id)
                    if entry.get("sourceKey") == key:
                        next_state[id] = entry
                    continue
                wait = delay - (time.monotonic() - last_fetch)
                if wait > 0:
                    time.sleep(wait)
                last_fetch, requests = time.monotonic(), requests + 1
                data = fetcher(url)
                legacy = False
            features = decode(data, legacy)
            if not validate(features):
                raise ValueError("No valid visible pixels")
            results[id] = features
            next_state[id] = {"sourceKey": key, "sha256": hashlib.sha256(data).hexdigest(), "features": features}
        except (OSError, ValueError, urllib.error.URLError) as error:
            pending.append(id)
            failures.append({"id": id, "error": str(error)[:240]})
            next_state[id] = {"sourceKey": key, "retryAt": int(time.time()) + 86400}
            if fetch_host and isinstance(error, urllib.error.HTTPError) and error.code in (403, 429):
                # Respect this provider's refusal without blocking unrelated museums.
                blocked_hosts.add(fetch_host)
        if (position + 1) % 100 == 0:
            atomic(state_path, {"signature": signature, "records": next_state})
    atomic(state_path, {"signature": signature, "records": next_state})
    generated = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    sidecar = {"version": VERSION, "generatedAt": generated, "indexedCount": len(results),
               "totalCount": len(payload["records"]), "records": dict(sorted(results.items()))}
    report = {"indexedCount": len(results), "totalCount": len(payload["records"]), "pendingCount": len(pending),
              "reused": reused, "networkRequests": requests, "inferenceRuns": 0,
              "vectorSha256": payload.get("embeddings", {}).get("sha256"), "failures": failures[:64]}
    return sidecar, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=pathlib.Path, required=True)
    parser.add_argument("--cache-dir", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--state", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("--write-index", type=pathlib.Path, help="Optional enriched JSON copy; vectors stay byte-identical")
    parser.add_argument("--legacy-padding", help="RGB of old 224px inference padding, e.g.124,116,104")
    parser.add_argument("--fetch-missing", action="store_true", help="Opt in to serial small previews; default is offline")
    parser.add_argument("--fetch-limit", type=int, default=256)
    parser.add_argument("--delay", type=float, default=1)
    parser.add_argument("--max-seconds", type=float, default=3600)
    args = parser.parse_args()
    padding = tuple(map(int, args.legacy_padding.split(","))) if args.legacy_padding else None
    if padding and (len(padding) != 3 or any(not 0 <= value <= 255 for value in padding)):
        parser.error("Invalid legacy padding RGB")
    payload = read_index(args.index)
    sidecar, report = backfill(payload, args.cache_dir, args.state, padding, args.fetch_missing,
                               args.fetch_limit, args.delay, args.max_seconds)
    atomic(args.output, sidecar)
    if args.report:
        atomic(args.report, report)
    if args.write_index:
        for art in payload["records"]:
            if str(art["id"]) in sidecar["records"]:
                art["colorFeatures"] = sidecar["records"][str(art["id"]) ]
        payload["generatedAt"] = sidecar["generatedAt"]
        payload.setdefault("coverage", {})["colorIndexed"] = len(sidecar["records"])
        atomic(args.write_index, payload)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
