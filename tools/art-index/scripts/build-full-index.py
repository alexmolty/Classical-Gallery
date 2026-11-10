"""Resumable full-catalog image inference. No model or Python runtime goes into the wallpaper."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import concurrent.futures,hashlib,io,json,pathlib,sys,threading,time,urllib.error,urllib.request
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'python-libs'))
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageOps
from tone_brightness import measure_brightness,measure_legacy_preview
from desktop_publication import legacy_record
Image.MAX_IMAGE_PIXELS=48_000_000
PREVIEWS=HERE/'previews';PREVIEWS.mkdir(exist_ok=True);FEATURES=HERE/'full-features';FEATURES.mkdir(exist_ok=True)
catalog=json.loads((HERE/'full-catalog.json').read_text(encoding='utf8'))
override_path=HERE/'preview-overrides.json'
overrides=json.loads(override_path.read_text(encoding='utf8')) if override_path.exists() else {}
for art in catalog:
 if art['id'] in overrides:art['previewUrl']=overrides[art['id']]
dino_config=json.loads((HERE/'preprocessor_config.json').read_text())
padding=tuple(round(v*255) for v in dino_config['image_mean'])
clip_config=json.loads((HERE/'clip-preprocessor_config.json').read_text())
model_hashes=[hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['model.onnx','clip-vision_model.onnx']]
signature='full-composition-224-v1:'+':'.join(model_hashes)
deadline=float(os.environ.get('ART_INDEX_RUN_DEADLINE','inf'))
# Older vectors with no cached pixels remain usable. Limit extra downloads for
# their brightness so new visual-index work keeps its existing time budget.
brightness_backfill=threading.Semaphore(256)
prior=json.loads((HERE/'build-report.json').read_text(encoding='utf8'))
old_clip,old_dino=np.load(HERE/'clip-vectors.npy'),np.load(HERE/'dinov2-vectors.npy')
for i,record in enumerate(prior['provenance']):
 target=FEATURES/(record['id']+'.npz')
 if not target.exists():np.savez_compressed(target,clip=old_clip[i],dino=old_dino[i],signature=signature,previewSha256=record['previewSha256'])

def cached_brightness(target, meta):
 if 'toneBrightness' in meta or not target.exists():return
 try:
  data=target.read_bytes()
  if hashlib.sha256(data).hexdigest()!=meta['previewSha256']:return
  with Image.open(io.BytesIO(data)) as image:
   value=measure_legacy_preview(image,padding)
  if value is not None:meta['toneBrightness']=value
 except (OSError,ValueError):pass

def preview(art):
 target=PREVIEWS/(art['id']+'.png');meta_path=target.with_suffix('.json')
 source_key=hashlib.sha256((art['previewUrl']+'|'+art['imageUrl']+'|'+str(art['metadata'].get('metadataUpdatedAt',''))).encode()).hexdigest()
 feature_path=FEATURES/(art['id']+'.npz')
 stale_feature=False;cached_meta=None
 if feature_path.exists():
  with np.load(feature_path) as cached:
   if str(cached['signature'])==signature and 'sourceKey' in cached and str(cached['sourceKey'])==source_key:
    cached_meta={'previewSha256':str(cached['previewSha256']),'sourceKey':source_key}
    if 'toneBrightness' in cached:cached_meta['toneBrightness']=int(cached['toneBrightness'].item())
    if 'brightnessRetryAt' in cached:cached_meta['brightnessRetryAt']=int(cached['brightnessRetryAt'].item())
    cached_brightness(target,cached_meta)
    if 'toneBrightness' in cached_meta or time.time()>=deadline or cached_meta.get('brightnessRetryAt',0)>time.time() or os.environ.get('ART_INDEX_OFFLINE')=='1' or not brightness_backfill.acquire(blocking=False):
     return art,cached_meta
   stale_feature='sourceKey' in cached and str(cached['sourceKey'])!=source_key
 if time.time()>=deadline:return art,cached_meta or {'error':'Image time budget reached; deferred until a later build'}
 if cached_meta is None and target.exists() and meta_path.exists():
  previous=json.loads(meta_path.read_text())
  if previous.get('sourceKey')==source_key or not stale_feature and not previous.get('sourceKey') and previous.get('source') in [art['previewUrl'],art['imageUrl'],art.get('localPath')]:
   previous['sourceKey']=source_key
   cached_brightness(target,previous)
   meta_path.write_text(json.dumps(previous),encoding='utf8');return art,previous
 errors=[]
 for source in dict.fromkeys([art.get('localPath') or art['previewUrl'],art['imageUrl']]):
  try:
   if source==art.get('localPath'):data=pathlib.Path(source).read_bytes()
   else:
    if os.environ.get('ART_INDEX_OFFLINE')=='1':raise RuntimeError('Image not available in offline cache')
    request=urllib.request.Request(source,headers={'User-Agent':'ClassicalGallery-VisualIndex/2.0'})
    with urllib.request.urlopen(request,timeout=25) as response:data=response.read(32*1024*1024+1)
   if len(data)>32*1024*1024:raise ValueError('Image exceeds 32MiB')
   with Image.open(io.BytesIO(data)) as original:
    original.load();oriented=ImageOps.exif_transpose(original)
    value=measure_brightness(oriented)
    if cached_meta is not None:
     if value is not None:cached_meta['toneBrightness']=value
     else:cached_meta['brightnessRetryAt']=int(time.time())+3*86400
     return art,cached_meta
    image=oriented.convert('RGB');image.thumbnail((224,224),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(224,224),padding)
    canvas.paste(image,((224-image.width)//2,(224-image.height)//2));canvas.save(target)
   meta={'id':art['id'],'source':source,'sourceKey':source_key,'sourceSha256':hashlib.sha256(data).hexdigest(),'previewSha256':hashlib.sha256(target.read_bytes()).hexdigest(),'pageUrl':art['pageUrl']}
   if value is not None:meta['toneBrightness']=value
   meta_path.write_text(json.dumps(meta),encoding='utf8');return art,meta
  except Exception as error:
   errors.append(str(error))
   if isinstance(error,urllib.error.HTTPError) and error.code in (403,429):break
 if cached_meta is not None:
  cached_meta['brightnessRetryAt']=int(time.time())+3*86400
  return art,cached_meta
 return art,{'error':'; '.join(errors)}

options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
dino=ort.InferenceSession(str(HERE/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
clip=ort.InferenceSession(str(HERE/'clip-vision_model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
def feature(image,session,config,kind):
 pixels=(image/255.0-np.asarray(config['image_mean'],dtype=np.float32).reshape(1,1,3))/np.asarray(config['image_std'],dtype=np.float32).reshape(1,1,3)
 out=session.run(None,{session.get_inputs()[0].name:pixels.transpose(2,0,1)[None].copy()})
 vector=out[[v.name for v in session.get_outputs()].index('image_embeds')][0] if kind=='clip' else out[0][0,0]
 assert np.isfinite(vector).all();return vector/np.linalg.norm(vector)
indexed,clip_vectors,dino_vectors,failed=[],[],[],[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
 futures=[executor.submit(preview,art) for art in catalog]
 for n,future in enumerate(concurrent.futures.as_completed(futures)):
  art,meta=future.result()
  if 'error' in meta:failed.append({'id':art['id'],'error':meta['error']})
  else:
   feature_path=FEATURES/(art['id']+'.npz');cv=dv=None
   if feature_path.exists():
    with np.load(feature_path) as loaded:
     parts=str(loaded['signature']).split(':')
     if len(parts)==3 and parts[0]=='full-composition-224-v1' and str(loaded['previewSha256'])==meta['previewSha256']:
      if parts[1]==model_hashes[0]:dv=loaded['dino'].copy()
      if parts[2]==model_hashes[1]:cv=loaded['clip'].copy()
   if cv is None or dv is None:
    if time.time()>=deadline:
     failed.append({'id':art['id'],'error':'Image time budget reached; deferred until a later build'})
     continue
    with Image.open(PREVIEWS/(art['id']+'.png')) as image:pixels=np.asarray(image,dtype=np.float32)
    if cv is None:cv=feature(pixels,clip,clip_config,'clip')
    if dv is None:dv=feature(pixels,dino,dino_config,'dino')
   brightness={'toneBrightness':meta['toneBrightness']} if 'toneBrightness' in meta else {}
   brightness_cache=brightness or {key:meta[key] for key in ('brightnessRetryAt',) if key in meta}
   np.savez_compressed(feature_path,clip=cv,dino=dv,signature=signature,previewSha256=meta['previewSha256'],sourceKey=meta['sourceKey'],**brightness_cache)
   indexed.append({**art,'metadata':{**art['metadata'],**brightness}});clip_vectors.append(cv);dino_vectors.append(dv)
  if (n+1)%100==0:print(json.dumps({'stage':'full image inference','processed':n+1,'total':len(catalog),'indexed':len(indexed),'failed':len(failed)}),flush=True)
order=np.argsort([a['id'] for a in indexed]);indexed=[indexed[i] for i in order]
cv=np.stack(clip_vectors)[order];dv=np.stack(dino_vectors)[order]
np.save(HERE/'full-clip-vectors.npy',cv);np.save(HERE/'full-dino-vectors.npy',dv)
# Only the original clients consume precomputed neighbors. The expanded desktop
# catalogue uses its vector matrix and does not need an additional quadratic pass.
neighbors={};count=len(indexed)
legacy_positions=[i for i,art in enumerate(indexed) if legacy_record(art['metadata'])]
legacy_cv,legacy_dv=cv[legacy_positions],dv[legacy_positions]
legacy_count=len(legacy_positions)
for start in range(0,legacy_count,64):
 scores=.8*(legacy_cv[start:start+64]@legacy_cv.T)+.2*(legacy_dv[start:start+64]@legacy_dv.T)
 for offset,row in enumerate(scores):
  i=start+offset;row[i]=-1
  top=np.argpartition(-row,min(79,legacy_count-1))[:min(80,legacy_count-1)];top=top[np.argsort(-row[top])]
  neighbors[indexed[legacy_positions[i]]['id']]=[[indexed[legacy_positions[j]]['id'],round(float(row[j]),5)] for j in top if row[j]>=.45]
payload={'version':1,'model':'CLIP ViT-B/32 + DINOv2 ViT-S/14','indexedCount':count,'minimumScore':.45,
 'clip':{'repository':'Xenova/clip-vit-base-patch32','revision':'d15189d','sha256':model_hashes[1],'precision':'float32','weight':.8},
 'dino':{'repository':'Xenova/dinov2-small','revision':'c2bb04a','sha256':model_hashes[0],'weight':.2},
 'records':[{**a['metadata'],'indexPreviewUrl':a['previewUrl']} for a in indexed],'neighbors':neighbors,'generatedAt':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
target=HERE/'full-index.json';target.write_text(json.dumps(payload,ensure_ascii=True,separators=(',',':')),encoding='utf8')
report={'result':'PASS' if not failed else 'INCOMPLETE','catalogCount':len(catalog),'indexedCount':count,'failed':failed,'indexBytes':target.stat().st_size,'ids':[a['id'] for a in indexed]}
(HERE/'full-index-report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k not in ['ids','failed']}|{'failed':failed[:12],'failureCount':len(failed)}),flush=True)

