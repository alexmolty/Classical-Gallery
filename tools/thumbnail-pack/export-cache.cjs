'use strict';
// Development tool: reuse verified index pixels; no downloads, models or inference.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {createRequire}=require('node:module');
const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const validId=id=>typeof id==='string'&&/^[a-z0-9-]{1,160}$/i.test(id);
function decoder(){
  if(process.env.THUMBNAIL_SHARP_MODULE)return require(process.env.THUMBNAIL_SHARP_MODULE);
  for(const base of [path.resolve(__dirname,'../../tmp/subject-enrichment/tools/package.json'),path.resolve(__dirname,'../subject-index/package.json')]){
    try{return createRequire(base)('sharp');}catch(_){}
  }
  throw Error('Install development-only sharp or set THUMBNAIL_SHARP_MODULE.');
}
async function jpeg(bytes,{legacy=false,padding,sharp=decoder()}={}){
  let image=sharp(bytes,{limitInputPixels:48000000,failOn:'error'}).rotate().toColourspace('srgb');
  const meta=await image.metadata();
  if(!['jpeg','png','webp'].includes(meta.format)||!meta.width||!meta.height)throw Error('Unsupported cached image');
  // Only the known, checksum-verified 224px inference canvas has model padding.
  // Require symmetric full border rows/columns; never trim an ordinary artwork.
  if(legacy&&meta.width===224&&meta.height===224&&padding?.length===3){
    const {data,info}=await image.removeAlpha().raw().toBuffer({resolveWithObject:true});
    const pad=(x,y)=>padding.every((v,c)=>data[(y*info.width+x)*info.channels+c]===v);
    let l=0,r=224,t=0,b=224;
    while(l<224&&Array.from({length:224},(_,y)=>pad(l,y)).every(Boolean))l++;
    while(r>l&&Array.from({length:224},(_,y)=>pad(r-1,y)).every(Boolean))r--;
    while(t<224&&Array.from({length:224},(_,x)=>pad(x,t)).every(Boolean))t++;
    while(b>t&&Array.from({length:224},(_,x)=>pad(x,b-1)).every(Boolean))b--;
    if(l<r&&t<b){
      if(l>0&&224-r>0&&Math.abs(l-(224-r))<=1&&t===0&&b===224)image=image.extract({left:l,top:0,width:r-l,height:224});
      else if(t>0&&224-b>0&&Math.abs(t-(224-b))<=1&&l===0&&r===224)image=image.extract({left:0,top:t,width:224,height:b-t});
    }
  }
  const result=await image.resize(384,384,{fit:'inside',withoutEnlargement:true}).flatten({background:'#ffffff'})
    .jpeg({quality:82,mozjpeg:true}).toBuffer({resolveWithObject:true});
  return {bytes:result.data,width:result.info.width,height:result.info.height};
}
async function exportCache(work,output,{sharp=decoder()}={}){
  const previews=path.join(work,'previews'),files=fs.readdirSync(previews).filter(n=>/\.(png|jpe?g|webp)$/i.test(n)).sort();
  fs.mkdirSync(path.join(output,'images'),{recursive:true});
  let padding;try{padding=JSON.parse(fs.readFileSync(path.join(work,'preprocessor_config.json'),'utf8')).image_mean.map(v=>Math.round(v*255));}catch(_){}
  const records=[],failed=[],seen=new Set();
  for(const name of files){
    const id=path.parse(name).name;if(!validId(id)||seen.has(id))continue;
    try{
      const file=path.join(previews,name),metaFile=path.join(previews,id+'.json');
      if(fs.statSync(file).size>32*1024*1024||fs.statSync(metaFile).size>128*1024)throw Error('Invalid cache sizes');
      const bytes=fs.readFileSync(file),meta=JSON.parse(fs.readFileSync(metaFile,'utf8'));
      if(meta.id&&String(meta.id)!==id||!meta.previewSha256||hash(bytes)!==meta.previewSha256||typeof meta.source!=='string')throw Error('Unverified cached pixels');
      const result=await jpeg(bytes,{legacy:path.extname(name)==='.png',padding,sharp});
      const target=path.join(output,'images',id+'.jpg');fs.writeFileSync(target,result.bytes);
      records.push({id,source:meta.source,sourceKey:meta.sourceKey||'',inputSha256:meta.previewSha256,
        sha256:hash(result.bytes),width:result.width,height:result.height,legacyCanvas:path.extname(name)==='.png'});
      seen.add(id);
    }catch(error){failed.push({id,error:String(error.message).slice(0,200)});}
    if((records.length+failed.length)%1000===0)console.log(JSON.stringify({exported:records.length,failed:failed.length}));
  }
  const report={version:1,encoder:'jpeg-384-q82-v1',records,failed};
  fs.writeFileSync(path.join(output,'export.json'),JSON.stringify(report));
  console.log(JSON.stringify({exported:records.length,failed:failed.length,networkRequests:0,inferenceRuns:0}));
  return report;
}
module.exports={decoder,jpeg,exportCache,hash,validId};
if(require.main===module){
  const args=process.argv.slice(2),get=name=>args[args.indexOf(name)+1];
  if(!args.includes('--work')||!args.includes('--output'))throw Error('Use --work <cached ai-index directory> --output <export directory>.');
  exportCache(path.resolve(get('--work')),path.resolve(get('--output'))).catch(e=>{console.error(e);process.exitCode=1;});
}
