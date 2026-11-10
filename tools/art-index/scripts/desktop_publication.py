"""Produce separate desktop coverage and preserve the existing vector contract."""
import base64
import collections
import hashlib
import re

LEGACY_ID = re.compile(r'(?:(?:(?:met|cma)-)?\d+|masterpiece-[a-z0-9-]{1,80})')


def legacy_record(record):
    return LEGACY_ID.fullmatch(str(record['id'])) is not None


def select_legacy_vectors(payload, coverage):
    source = payload['embeddings']
    raw = base64.b64decode(source['data'], validate=True)
    records = [art for art in payload['records'] if legacy_record(art)]
    accepted = {str(art['id']) for art in records}
    positions = [(index, id) for index, id in enumerate(source['ids']) if id in accepted]
    if len(positions) != len(records):
        raise ValueError('Legacy vector membership does not match the catalogue')
    selected = b''.join(raw[index * 896:(index + 1) * 896] for index, _ in positions)
    return {**payload, 'records': records, 'indexedCount': len(records), 'coverage': coverage,
            'embeddings': {**source, 'ids': [id for _, id in positions],
                           'data': base64.b64encode(selected).decode('ascii'),
                           'sha256': hashlib.sha256(selected).hexdigest()}}


def coverage_for(inventory, records, failures, collection):
    indexed = {str(art['id']): art for art in records}
    known = {art['id'] for art in inventory}
    counts = collections.Counter()
    by_source = collections.defaultdict(collections.Counter)
    missing = [row for row in failures if row['id'] in known]
    criteria = ('title', 'artist', 'origin', 'style', 'genre', 'period', 'subject', 'year',
                'source', 'orientation', 'brightness', 'visual')
    for art in inventory:
        source = art['metadata'].get('sourceMuseum') or 'unknown'
        group = by_source[source]
        group['catalogue'] += 1
        flags = dict(art.get('tagCoverage', {}))
        indexed_art = indexed.get(art['id'])
        if indexed_art:
            flags['visual'] = True
            flags['brightness'] = 'toneBrightness' in indexed_art
            flags['style'] = flags.get('style') or bool(indexed_art.get('workMovements') or indexed_art.get('artistMovements'))
        for field in criteria:
            if flags.get(field):
                group[field] += 1
                counts[field] += 1
    return {
        'collectionResult': collection.get('result', 'INCOMPLETE'),
        'enumeratedMetCandidates': collection.get('metEnumeratedIDs', 0),
        'metPendingMetadata': len(collection.get('metFailures', [])),
        'clevelandError': str(collection.get('clevelandError') or '')[:200],
        'imageCandidates': len(inventory), 'indexedImages': len(indexed),
        'pendingImageCount': len(missing),
        'pendingImages': [{'id': row['id'], 'error': row['error'][:200]} for row in missing[:16]],
        'brightnessIndexed': counts['brightness'], 'styleTaggedWorks': counts['style'],
        'styleUntaggedWorks': len(inventory) - counts['style'],
        'byCriterion': {field: counts[field] for field in criteria},
        'bySource': {source: dict(group) for source, group in sorted(by_source.items())}
    }
