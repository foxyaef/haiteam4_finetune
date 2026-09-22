import argparse,concurrent.futures,hashlib,json,math,os,threading,time,urllib.request,msvcrt,tempfile,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/downloads'
CACHE=Path(os.environ.get('BDD100K_DOWNLOAD_CACHE',str(Path(tempfile.gettempdir())/'rtdetr_bdd100k_download_cache')))
BLOCK=64*1024*1024

def status(value):
    value['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    temp=DEST/'download-status.tmp'
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp.replace(DEST/'download-status.json')
    print(json.dumps(value),flush=True)

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def download(url):
    DEST.mkdir(parents=True,exist_ok=True)
    name=url.rsplit('/',1)[1]
    target=DEST/name
    for attempt in range(30):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as r:
                total=int(r.headers['Content-Length'])
                signature={'url':url,'bytes':total,'etag':r.headers.get('ETag'),'last_modified':r.headers.get('Last-Modified')}
            break
        except Exception as e:
            status({'phase':'waiting_for_network','file':name,'attempt':attempt+1,'error':str(e)})
            if attempt==29:raise
            time.sleep(20)
    record=target.with_suffix('.download.json')
    if target.exists() and record.exists():
        old=json.loads(record.read_text())
        if target.stat().st_size==total and old['sha256']==sha256(target):
            print('Already verified:',name,flush=True)
            return
        raise RuntimeError('Existing archive hash mismatch: '+name)
    CACHE.mkdir(parents=True,exist_ok=True)
    parts=CACHE/(name+'.parts')
    parts.mkdir(exist_ok=True)
    lock=parts/'source.json'
    if lock.exists() and json.loads(lock.read_text())!=signature:
        raise RuntimeError('Remote source changed; keep old parts separately.')
    lock.write_text(json.dumps(signature,indent=2))
    count=math.ceil(total/BLOCK)
    def task(i):
        start=i*BLOCK; end=min(total,(i+1)*BLOCK)-1
        piece=parts/f'{i:04}.part'
        if piece.exists() and piece.stat().st_size==end-start+1:return
        for attempt in range(30):
            try:
                have=piece.stat().st_size if piece.exists() else 0
                if have>end-start+1:raise RuntimeError('Chunk larger than expected')
                if have==end-start+1:return
                resume=start+have
                req=urllib.request.Request(url,headers={'Range':f'bytes={resume}-{end}','User-Agent':'BDD100K-research-downloader/1.0'})
                with urllib.request.urlopen(req,timeout=90) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {resume}-{end}/{total}':
                        raise RuntimeError('Unexpected range response')
                    with piece.open('ab') as f:
                        while b:=r.read(1024*1024):f.write(b)
                if piece.stat().st_size!=end-start+1:raise IOError('Incomplete chunk')
                return
            except Exception:
                if attempt==29:raise
                time.sleep(min(20,2**attempt))
    began=time.monotonic()
    def downloaded_bytes():
        return sum((parts/f'{i:04}.part').stat().st_size for i in range(count) if (parts/f'{i:04}.part').exists())
    initial=downloaded_bytes()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(task,i) for i in range(count)]
        while any(not f.done() for f in futures):
            done=downloaded_bytes()
            elapsed=time.monotonic()-began
            status({'phase':'downloading','file':name,'downloaded_gb':round(done/1e9,3),'total_gb':round(total/1e9,3),'percent':round(100*done/total,1),'elapsed_seconds':round(elapsed),'mb_per_second':round((done-initial)/max(1,elapsed)/1e6,2)})
            time.sleep(15)
        for f in futures:f.result()
    status({'phase':'assembling_and_hashing','file':name,'percent':100})
    temp=CACHE/(name+'.assembling')
    h=hashlib.sha256()
    with temp.open('wb') as out:
        for i in range(count):
            with (parts/f'{i:04}.part').open('rb') as f:
                for b in iter(lambda:f.read(4*1024*1024),b''):out.write(b);h.update(b)
    if temp.stat().st_size!=total:raise IOError('Assembled size mismatch')
    # Only finalized archives are published into the OneDrive project.
    shutil.copyfile(temp,target)
    signature.update(sha256=h.hexdigest(),retrieved_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    record.write_text(json.dumps(signature,indent=2))
    temp.unlink()
    # Delete only our exact numbered chunk files after a verified assembly.
    for i in range(count):(parts/f'{i:04}.part').unlink()
    status({'phase':'complete','file':name,**signature})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('urls',nargs='+');args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    CACHE.mkdir(parents=True,exist_ok=True)
    with (CACHE/'downloader.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        for url in args.urls:download(url)
