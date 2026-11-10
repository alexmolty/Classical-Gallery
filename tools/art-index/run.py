"""Standalone GitHub Actions entry point. Builds data only; never executes in Wallpaper Engine."""
import argparse,hashlib,json,os,pathlib,shutil,subprocess,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
CONTEXT=ROOT/'cache';WORK=CONTEXT/'work/ai-index';SCHEMA=CONTEXT/'outputs/chicago-gallery'
parser=argparse.ArgumentParser();parser.add_argument('--offline',action='store_true');parser.add_argument('--skip-collection',action='store_true');parser.add_argument('--verify-portability',action='store_true');parser.add_argument('--publish-dir',type=pathlib.Path,required=True);args=parser.parse_args()
WORK.mkdir(parents=True,exist_ok=True);SCHEMA.mkdir(parents=True,exist_ok=True)
for name in ['core.js','museum-catalog.js']:shutil.copyfile(ROOT/'schema'/name,SCHEMA/name)
for name in ['collect-full-catalog.py','normalize-full-catalog.cjs','build-full-index.py','resolve-missing-previews.py','build-vector-index.py','tone_brightness.py','desktop_publication.py','desktop-catalog.json.gz']:shutil.copyfile(ROOT/'scripts'/name,WORK/name)
for name in ['gallery-records.json','catalog.json','preview-overrides.json']:
 if not (WORK/name).exists():shutil.copyfile(ROOT/'bootstrap'/name,WORK/name)
shutil.copyfile(ROOT/'bootstrap/masterpieces.json',WORK/'masterpieces.json')
previous_manifest=args.publish_dir/'manifest.json'
previous=None
if previous_manifest.exists():
 previous=json.loads(previous_manifest.read_text(encoding='utf8'))
 if not __import__('re').fullmatch(r'art-index-[a-f0-9]{16}\.json',previous['file']):raise RuntimeError('Invalid previous index filename')
 previous_bytes=(args.publish_dir/previous['file']).read_bytes()
 if hashlib.sha256(previous_bytes).hexdigest()!=previous['sha256']:raise RuntimeError('Previous index integrity mismatch')
 previous_data=json.loads(previous_bytes)
 (WORK/'gallery-records.json').write_text(json.dumps(previous_data['records']),encoding='utf8')
 (WORK/'catalog.json').write_text(json.dumps([{'id':str(a['id']),'previewUrl':a.get('indexPreviewUrl') or a['imageUrl'],'imageUrl':a['imageUrl']} for a in previous_data['records']]),encoding='utf8')
if not (WORK/'build-report.json').exists():(WORK/'build-report.json').write_text('{"provenance":[]}',encoding='utf8')
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
# Leave time for publishing and the Actions cache save before the job's six-hour
# timeout. Cache hits remain usable; new work beyond the deadline is deferred.
os.environ.setdefault('ART_INDEX_RUN_DEADLINE',str(int(time.time()+4.5*3600)))
os.environ.setdefault('ART_INDEX_METADATA_BUDGET_SECONDS','3600')
if args.offline:os.environ['ART_INDEX_OFFLINE']='1'
MODELS=[('model.onnx','https://huggingface.co/Xenova/dinov2-small/resolve/c2bb04a/onnx/model.onnx','83141175ec78b4ff9a2bb58a4c7c264ba0054d1c2e122e5a8114b79a8d4179ea'),
 ('clip-vision_model.onnx','https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/d15189d/onnx/vision_model.onnx','fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40')]
for name,url,expected in MODELS:
 target=WORK/name
 if not target.exists():
  if args.offline:raise RuntimeError('Missing cached model for offline verification')
  with urllib.request.urlopen(url,timeout=90) as response:body=response.read(400_000_001)
  if len(body)>400_000_000 or hashlib.sha256(body).hexdigest()!=expected:raise RuntimeError('Model download/hash mismatch')
  target.write_bytes(body)
 if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:raise RuntimeError('Cached model hash mismatch')
for name in ['preprocessor_config.json','clip-preprocessor_config.json']:shutil.copyfile(ROOT/'bootstrap'/name,WORK/name)
import numpy as np
FEATURES=WORK/'full-features';FEATURES.mkdir(exist_ok=True)
bootstrap=json.loads((ROOT/'bootstrap/features.json').read_text(encoding='utf8'))
with np.load(ROOT/'bootstrap/features.npz') as vectors:
 clip_bootstrap,dino_bootstrap=vectors['clip'],vectors['dino']
 for i,row in enumerate(bootstrap['entries']):
  target=FEATURES/(row['id']+'.npz')
  use_bootstrap=not target.exists()
  if target.exists():
   with np.load(target) as previous_feature:use_bootstrap=str(previous_feature['signature'])!=bootstrap['signature']
  if use_bootstrap:np.savez_compressed(target,clip=clip_bootstrap[i],dino=dino_bootstrap[i],signature=bootstrap['signature'],previewSha256=row['previewSha256'],sourceKey=row['sourceKey'])
probe_ids=['cma-94979','met-10062'] if args.verify_portability else []
for id in probe_ids:
 target=FEATURES/(id+'.npz')
 if target.exists():target.unlink()
# The legacy bootstrap arrays are empty: full-features already contains verified cached vectors.
for name,dimension in [('clip-vectors.npy',512),('dinov2-vectors.npy',384)]:
 if not (WORK/name).exists():np.save(WORK/name,np.empty((0,dimension),dtype=np.float32))
def run(name):subprocess.run(([os.environ.get('NODE_BINARY','node')] if name.endswith('.cjs') else [sys.executable])+[str(WORK/name)],check=True)
if not args.offline and not args.skip_collection:
 try:run('collect-full-catalog.py')
 except subprocess.CalledProcessError:
  if not (WORK/'full-cleveland.json').exists():raise
  status_path=WORK/'full-collection-report.json'
  status=json.loads(status_path.read_text(encoding='utf8')) if status_path.exists() else {}
  status.update({'result':'INCOMPLETE','collectionError':'Current museum API collection failed; cached records retained'})
  status_path.write_text(json.dumps(status),encoding='utf8')
  print('Museum collection incomplete; preserving cached metadata and reporting coverage.',flush=True)
run('normalize-full-catalog.cjs');run('build-full-index.py')
first=json.loads((WORK/'full-index-report.json').read_text(encoding='utf8'))
if first['failed'] and not args.offline and time.time()<float(os.environ['ART_INDEX_RUN_DEADLINE']):
 run('resolve-missing-previews.py');run('build-full-index.py')
payload=json.loads((WORK/'full-index.json').read_text(encoding='utf8'))
for id in probe_ids:
 row=next(i for i,a in enumerate(bootstrap['entries']) if a['id']==id)
 with np.load(FEATURES/(id+'.npz')) as actual:
  clip_cos=float(actual['clip']@clip_bootstrap[row]);dino_cos=float(actual['dino']@dino_bootstrap[row])
  clip_error=float(np.max(np.abs(actual['clip']-clip_bootstrap[row])));dino_error=float(np.max(np.abs(actual['dino']-dino_bootstrap[row])))
  check={'id':id,'clipCosine':clip_cos,'dinoCosine':dino_cos,'clipMaxError':clip_error,'dinoMaxError':dino_error,'previewHashMatches':str(actual['previewSha256'])==bootstrap['entries'][row]['previewSha256']}
  print('PORTABILITY_CHECK '+json.dumps(check),flush=True)
  if clip_cos<.99999 or dino_cos<.99999 or clip_error>.0001 or dino_error>.0001:raise RuntimeError('Independent CPU portability check failed: '+id)
collection=json.loads((WORK/'full-collection-report.json').read_text(encoding='utf8')) if (WORK/'full-collection-report.json').exists() else {'result':'INCOMPLETE','metEnumeratedIDs':14423}
inference=json.loads((WORK/'full-index-report.json').read_text(encoding='utf8'))
from scripts.desktop_publication import coverage_for, legacy_record, select_legacy_vectors
from scripts.style_metadata import enrich_styles
from publish_index import load_previous,publish_index

# Encode the entire catalogue for the expanded desktop channel.
payload['updateManifest']='https://raw.githubusercontent.com/alexmolty/Art-Gallery-for-Wallpaper-Engine/art-index/manifest.json'
(WORK/'full-index.json').write_text(json.dumps(payload,ensure_ascii=True,separators=(',',':')),encoding='utf8')
run('build-vector-index.py')
vector_payload=json.loads((WORK/'vector-index.json').read_text(encoding='utf8'))
destination=args.publish_dir.resolve();destination.mkdir(parents=True,exist_ok=True)
_, previous_vector = load_previous(destination/'desktop'/'v2')
if not previous_vector:
 _, previous_vector = load_previous(destination/'v2')
inventory=json.loads((WORK/'full-catalog.json').read_text(encoding='utf8'))
style_report=enrich_styles(vector_payload, inventory, WORK, previous_vector or {},
 not args.offline, min(time.time()+420,float(os.environ['ART_INDEX_RUN_DEADLINE'])))
failures=inference['failed']
desktop_coverage=coverage_for(inventory,vector_payload['records'],failures,collection)
desktop_coverage.update({key:value for key,value in style_report.items() if key not in ('styleTaggedWorks','styleUntaggedWorks')})
desktop_payload={**vector_payload,'coverage':desktop_coverage,
 'updateManifest':'https://raw.githubusercontent.com/alexmolty/Classical-Gallery/art-index/desktop/v2/manifest.json'}
result=publish_index(destination/'desktop'/'v2',desktop_payload)
print('DESKTOP_VECTOR_PUBLICATION '+json.dumps(result),flush=True)

# Keep the old desktop and Wallpaper Engine clients on compatible identifiers.
legacy_inventory=[art for art in inventory if legacy_record(art['metadata'])]
legacy_vector_records=[art for art in vector_payload['records'] if legacy_record(art)]
legacy_coverage=coverage_for(legacy_inventory,legacy_vector_records,failures,collection)
result=publish_index(destination/'v2',select_legacy_vectors(vector_payload,legacy_coverage))
print('VECTOR_PUBLICATION '+json.dumps(result),flush=True)
legacy_records=[{key:value for key,value in art.items() if key!='toneBrightness'}
 for art in payload['records'] if legacy_record(art) and not str(art['id']).startswith('masterpiece-')]
legacy_ids={str(art['id']) for art in legacy_records}
legacy_payload={**payload,'indexedCount':len(legacy_records),'records':legacy_records,
 'neighbors':{id:[row for row in rows if str(row[0]) in legacy_ids] for id,rows in payload['neighbors'].items() if id in legacy_ids},
 'coverage':coverage_for([art for art in legacy_inventory if not art['id'].startswith('masterpiece-')],legacy_records,failures,collection)}
legacy_bytes=len(json.dumps(legacy_payload,ensure_ascii=True,separators=(',',':')).encode('utf8'))
if legacy_bytes>32*1024*1024 and previous:
 print('LEGACY_PUBLICATION '+json.dumps({'result':'RETAINED','reason':'legacy size limit','indexedCount':previous.get('indexedCount'),'bytes':legacy_bytes}),flush=True)
else:
 result=publish_index(destination,legacy_payload)
 print('LEGACY_PUBLICATION '+json.dumps(result),flush=True)

