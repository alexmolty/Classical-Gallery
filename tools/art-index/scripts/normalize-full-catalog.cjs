const fs=require('fs'),path=require('path'),vm=require('vm');
const here=__dirname,root=path.resolve(here,'../../outputs/chicago-gallery'),C=require(path.join(root,'core.js'));
const window={GalleryCore:C};
vm.runInNewContext(fs.readFileSync(path.join(root,'museum-catalog.js'),'utf8'),{window,URL,URLSearchParams,AbortController,DOMException,setTimeout,clearTimeout,console});
const original=JSON.parse(fs.readFileSync(path.join(here,'gallery-records.json'),'utf8'));
const byId=new Map(original.map(art=>[String(art.id),art]));
for(const id of byId.keys())if(id.startsWith('cma-'))byId.delete(id);
const previews=new Map(),rejected={cleveland:0,met:0};
const artistReferences = new Map();
const cma=JSON.parse(fs.readFileSync(path.join(here,'full-cleveland.json'),'utf8'));
for(const record of cma){
 const art=window.MuseumCatalog.normalize('cleveland',record);
 if(!art){rejected.cleveland++;continue;}
 if(record.is_highlight===true)art.masterpiece=true;
 byId.set(String(art.id),art);previews.set(String(art.id),record.images.web?.url||record.images.print.url);
}
const metFiles=fs.readdirSync(path.join(here,'full-raw')).filter(name=>/^met-\d+\.json$/.test(name));
for(const file of metFiles){
 const record=JSON.parse(fs.readFileSync(path.join(here,'full-raw',file),'utf8'));
 const art=window.MuseumCatalog.normalize('met',record);
 if(!art){rejected.met++;byId.delete('met-'+record.objectID);continue;}
 if(record.isHighlight===true)art.masterpiece=true;
 byId.set(String(art.id),C.mergeArtworkMetadata(byId.get(String(art.id)),art));
 const reference = String(record.artistWikidata_URL || '').match(/^https?:\/\/(?:www\.)?wikidata\.org\/(?:wiki|entity)\/(Q[1-9]\d*)\/?$/i);
 if (reference) artistReferences.set(String(art.id), reference[1].toUpperCase());
 previews.set(String(art.id),record.primaryImageSmall||record.primaryImage);
}
const originalPreviews=new Map(JSON.parse(fs.readFileSync(path.join(here,'catalog.json'),'utf8')).map(a=>[a.id,a]));
// Curated files are manually checked for rights, size, CORS and source before
// they enter this list. Their actual images still undergo the same inference.
const curated=JSON.parse(fs.readFileSync(path.join(here,'masterpieces.json'),'utf8'));
for(const art of curated){
 const standalone=/^masterpiece-[a-z0-9-]+$/.test(art.id)&&art.sourceMuseum==='masterpieces'&&art.license==='Public domain';
 const metOverride=/^met-\d+$/.test(art.id)&&art.sourceMuseum==='met'&&art.imageLicense==='Public domain'&&
   art.pageUrl==='https://www.metmuseum.org/art/collection/search/'+art.id.slice(4)&&
   typeof art.commonsFile==='string'&&art.commonsFile.startsWith('File:')&&
   /^https:\/\/commons\.wikimedia\.org\/wiki\/File:/.test(art.commonsPageUrl||'')&&
   Number.isSafeInteger(art.width)&&Number.isSafeInteger(art.height)&&art.width>0&&art.height>0&&
   art.width*art.height<=48000000&&Math.max(art.width,art.height)<=12000;
 if(!(standalone||metOverride)||art.masterpiece!==true||!art.verifiedCommons||
    !/^https:\/\/(?:thumb|upload)\.wikimedia\.org\//.test(art.imageUrl)||!art.pageUrl)
  throw new Error('Invalid curated artwork: '+String(art.id));
 if(metOverride){
  const museum=byId.get(art.id);
  // Never resurrect a museum object removed by the current eligibility checks.
  if(!museum||museum.sourceMuseum!=='met')continue;
  byId.set(art.id,{...museum,imageUrl:art.imageUrl,width:art.width,height:art.height,pendingValidation:false,
   masterpiece:true,verifiedCommons:true,commonsFile:art.commonsFile,commonsPageUrl:art.commonsPageUrl,
   imageLicense:art.imageLicense,source:art.source});
 }else byId.set(art.id,art);
 previews.set(art.id,art.imageUrl);
}

// Import the desktop's reviewed public museum snapshot. Live Met and Cleveland
// records remain authoritative when the APIs remove or replace an image.
const desktopPath=path.join(here,'desktop-catalog.json.gz');
const desktop=fs.existsSync(desktopPath)
 ? JSON.parse(require('node:zlib').gunzipSync(fs.readFileSync(desktopPath), {maxOutputLength:32*1024*1024}))
 : {records:[], supplementalStyles:{}};
const desktopSources=new Map([
 ['nga',['api.nga.gov','/iiif/']],['rijks',['iiif.micr.io','/']],['getty',['media.getty.edu','/iiif/image/']],
 ['smk',['iip.smk.dk','/iiif/jp2/']],['stockholm',['thumb.wikimedia.org','/wikipedia/commons/thumb/']],
 ['ateneum',['kokoelma.kansallisgalleria.fi','/media-assets/']],['sinebrychoff',['kokoelma.kansallisgalleria.fi','/media-assets/']]
]);
function verifiedLink(value,host,prefix) {
 try { const u=new URL(value); return u.protocol==='https:'&&!u.username&&!u.password&&u.hostname===host&&u.pathname.startsWith(prefix); }
 catch { return false; }
}
for(const art of desktop.records) {
 const id=String(art.id), source=art.sourceMuseum;
 if(source==='met'||source==='cleveland')continue;
 const rule=desktopSources.get(source);
 if(rule) {
  const validId=source==='getty'?/^getty-[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/.test(id):id.startsWith(source+'-')&&/^\d+$/.test(id.slice(source.length+1));
  const licensed=source==='nga'?art.imageLicense==='CC0':/^https:\/\/creativecommons\.org\/publicdomain\/(zero|mark)\/1\.0\/$/.test(art.rightsUrl||'');
  if(!validId||!licensed||!verifiedLink(art.imageUrl,...rule)||art.indexPreviewUrl&&!verifiedLink(art.indexPreviewUrl,...rule))
   throw new Error('Invalid desktop artwork: '+id);
 }else if(source==='chicago'){
  const allowed=url=>['thumb.wikimedia.org','upload.wikimedia.org'].some(host=>verifiedLink(url,host,'/wikipedia/commons/'));
  if(!/^\d+$/.test(id)||!allowed(art.imageUrl)||art.indexPreviewUrl&&!allowed(art.indexPreviewUrl))
   throw new Error('Invalid Chicago artwork: '+id);
 }else throw new Error('Unexpected desktop museum: '+source);
 if(art.local||art.localImage||!art.pageUrl)throw new Error('Invalid shared artwork: '+id);
 byId.set(id,C.mergeArtworkMetadata(byId.get(id),art));
 if(art.indexPreviewUrl)previews.set(id,art.indexPreviewUrl);
}
for(const [id, art] of byId){
 const supplemental=desktop.supplementalStyles[id];
 if(supplemental&&!C.movementsFor(art).length)byId.set(id,{...art,...supplemental});
}
function hasStyle(art) {
 return C.movementsFor(art).length>0||
  !!art.workMovements?.length&&!!(art.workStyleWikidata||art.workStyleSourceUrl)||
  !!art.artistMovements?.length&&!!(art.artistStyleWikidata||art.artistStyleSourceUrl);
}
function tagCoverage(art) {
 return {
  title:!!art.title&&!/^(untitled|unknown)$/i.test(art.title),
  artist:!!art.artist&&!/^(unknown|unidentified|anonymous|unknown artist|artist unknown|unidentified artist)$/i.test(art.artist),
  origin:C.originsFor(art).some(origin=>origin!=='Unspecified'),
  style:hasStyle(art), genre:C.genresFor(art).length>0,
  period:C.periodsFor(art).some(period=>period!=='undated'),
  subject:[...(Array.isArray(art.subjects)?art.subjects:[]),...(Array.isArray(art.tags)?art.tags:[])].some(tag=>!!(typeof tag==='string'?tag.trim():tag?.term||tag?.name)),
  year:Number.isInteger(art.dateStart)||Number.isInteger(art.dateEnd),
  source:!!art.sourceMuseum, orientation:art.width>0&&art.height>0
 };
}
const inventory=[...byId.values()].filter(a=>C.eligible(a,C.settings({}))).map(art=>({
 id:String(art.id),title:art.title,artist:art.artist,artistKey:C.artistKey(art.artist),pageUrl:art.pageUrl,imageUrl:art.imageUrl,
 previewUrl:previews.get(String(art.id))||originalPreviews.get(String(art.id))?.previewUrl||art.imageUrl,
 ...(originalPreviews.get(String(art.id))?.localPath?{localPath:originalPreviews.get(String(art.id)).localPath}:{}),
 artistWikidataId: artistReferences.get(String(art.id)) || art.artistStyleWikidata,
 artistCountries: [...new Set([art.artistNationality, ...(art.artistNationalities || [])].flatMap(C.nationalityOriginsFor))],
 hasDocumentedStyle:hasStyle(art), tagCoverage:tagCoverage(art),
 metadata:art
}));
fs.writeFileSync(path.join(here,'full-catalog.json'),JSON.stringify(inventory));
console.log(JSON.stringify({accepted:inventory.length,clevelandRecords:cma.length,metObjectsAvailable:metFiles.length,rejected,bySource:inventory.reduce((r,a)=>(r[a.metadata.sourceMuseum]=(r[a.metadata.sourceMuseum]||0)+1,r),{})}));

