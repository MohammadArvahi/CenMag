"""CPU-only, hash-locked upstream provisioning; never launches research training."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.pending')
    with temp.open('w') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def checked_target(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise RuntimeError('Unsafe provision destination')
    target = root/relative
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Provision destination escapes input root')
    return target


def verify_record(path, record):
    return path.is_file() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256']


def download(record, root):
    target = checked_target(root, record['relative_path'])
    if target.exists():
        if not verify_record(target, record):
            raise RuntimeError('Existing destination differs; preserved for inspection: '+record['relative_path'])
        return dict(path=record['relative_path'], status='REUSED_VERIFIED', network_bytes=0)
    url = record['url']
    if urllib.parse.urlparse(url).scheme != 'https':
        raise RuntimeError('HTTPS source required')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name+'.download.part')
    received = 0; started = time.monotonic()
    for attempt in range(1, 4):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > record['bytes']:
            raise RuntimeError('Oversized partial file; preserved for inspection')
        if offset == record['bytes']:
            if verify_record(partial, record): break
            raise RuntimeError('Complete partial hash mismatch; preserved for inspection')
        headers = {'User-Agent':'CenMag-hash-locked-preparation/1'}
        if offset: headers['Range'] = 'bytes=%d-' % offset
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers), timeout=90) as response:
                resumed = offset and response.status == 206
                if resumed and not response.headers.get('Content-Range','').startswith('bytes %d-' % offset):
                    raise RuntimeError('Incorrect upstream range response')
                mode = 'ab' if resumed else 'wb'
                current = offset if resumed else 0
                with partial.open(mode) as stream:
                    while block := response.read(4*1024**2):
                        current += len(block); received += len(block)
                        if current > record['bytes']: raise RuntimeError('Upstream file exceeds pinned size')
                        stream.write(block)
                    stream.flush(); os.fsync(stream.fileno())
            if not verify_record(partial, record):
                raise RuntimeError('Downloaded size/SHA256 differs from original publisher manifest: '+record['relative_path'])
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code in [400,401,403,404,410]: raise
            if attempt == 3: raise
            time.sleep(attempt)
    if not verify_record(partial, record): raise RuntimeError('Unverified download')
    partial.replace(target)
    return dict(path=record['relative_path'],status='DOWNLOADED_VERIFIED',network_bytes=received,
                source_url=url,sha256=record['sha256'],bytes=record['bytes'],seconds=time.monotonic()-started)


def provision(plan_path, root, groups):
    plan = json.loads(Path(plan_path).read_text()); root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = [r for r in plan['download_records'] if r['group'] in groups]
    missing = sum(r['bytes'] for r in records if not (root/r['relative_path']).is_file())
    if shutil.disk_usage(root).free < missing+1024**3:
        raise RuntimeError('Insufficient disk capacity for pinned inputs')
    started = time.monotonic(); report = dict(status='RUNNING',scope='CPU_ONLY_NO_TRAINING',
        plan_sha256=sha(plan_path),groups=groups,records=[],failures=[],started_unix=time.time())
    path = root/('download_'+('_'.join(groups))+'.json')
    atomic_json(path,report)
    # Embedded ensurepip is extracted only from the original verified CPython tar.
    direct = [r for r in records if 'url' in r]
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = {pool.submit(download,r,root):r for r in direct}
        for future in as_completed(tasks):
            r=tasks[future]
            try: report['records'].append(future.result())
            except Exception as error:
                report['failures'].append(dict(path=r['relative_path'],type=type(error).__name__,error=str(error).split('?')[0]))
            atomic_json(path,report)
    if not report['failures']:
        for r in records:
            if 'embedded_member' not in r:continue
            destination=checked_target(root,r['relative_path']); container=checked_target(root,r['container'])
            parent_record=next(v for v in plan['download_records'] if v['relative_path']==r['container'])
            if not verify_record(container,parent_record):raise RuntimeError('Embedded artifact container identity mismatch')
            if destination.exists():
                if not verify_record(destination,r):raise RuntimeError('Embedded destination mismatch')
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                with tarfile.open(container,'r:gz') as archive:
                    stream=archive.extractfile(r['embedded_member'])
                    if stream is None:raise RuntimeError('Missing embedded ensurepip wheel')
                    partial=destination.with_name(destination.name+'.download.part')
                    with stream, partial.open('xb') as target:shutil.copyfileobj(stream,target)
                if not verify_record(partial,r):raise RuntimeError('Embedded ensurepip hash mismatch')
                partial.replace(destination)
            report['records'].append(dict(path=r['relative_path'],status='EXTRACTED_VERIFIED',network_bytes=0,sha256=r['sha256'],bytes=r['bytes']))
    report.update(status='FAIL' if report['failures'] else 'PASS',seconds=time.monotonic()-started,
                  total_network_bytes=sum(r['network_bytes'] for r in report['records']),
                  verified_bytes=sum(r['bytes'] for r in records if verify_record(root/r['relative_path'],r)))
    atomic_json(path,report)
    if report['failures']:raise RuntimeError('Pinned provisioning failed; inspect '+str(path))
    return {k:v for k,v in report.items() if k!='records'}


def install(release, root, prefix):
    root, prefix, release = Path(root).resolve(),Path(prefix).resolve(),Path(release).resolve()
    if prefix.exists():raise RuntimeError('New isolated environment prefix required')
    command=[sys.executable,'-B',str(release/'runtime_setup.py'),'--input',str(root/'runtime'),'--prefix',str(prefix),'--execute']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    log=root/'environment_setup.log';started=time.monotonic()
    with log.open('x') as stream:
        result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Isolated setup failed; inspect '+str(log))
    python=prefix/'env/bin/python'
    probe="import json,torch,numpy,transformers,peft,bitsandbytes,pyarrow,datasets,evaluate,accelerate;print(json.dumps(dict(torch=str(torch.__version__),cuda=torch.version.cuda,numpy=numpy.__version__,transformers=transformers.__version__,peft=peft.__version__,bitsandbytes=bitsandbytes.__version__,pyarrow=pyarrow.__version__,datasets=datasets.__version__,evaluate=evaluate.__version__,accelerate=accelerate.__version__,visible_gpus=torch.cuda.device_count())))"
    completed=subprocess.run([str(python),'-B','-c',probe],env=env,capture_output=True,text=True,timeout=120)
    report=dict(status='PASS' if completed.returncode==0 else 'FAIL',seconds=time.monotonic()-started,
                python=str(python),imports_stdout=completed.stdout,imports_stderr=completed.stderr,
                cuda_probe=False,training=False)
    atomic_json(root/'environment_validation.json',report)
    if completed.returncode:raise RuntimeError('Pinned dependency imports failed')
    return report


def seal(root, release, plan_path):
    root, release = Path(root).resolve(),Path(release).resolve()
    plan=json.loads(Path(plan_path).read_text())
    failures=[r['relative_path'] for r in plan['download_records'] if not verify_record(root/r['relative_path'],r)]
    if failures:raise RuntimeError('Input verification failed: '+str(failures))
    for report_name in ['environment_validation.json','cache/rebuild_validation.json']:
        if json.loads((root/report_name).read_text())['status']!='PASS':raise RuntimeError('CPU gate did not pass: '+report_name)
    files={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='prepared_input_manifest.json'}
    manifest=dict(schema=1,status='CPU_VERIFIED_NOT_EXTERNALLY_DURABLE',source_code_manifest_sha256=sha(release/'RELEASE_MANIFEST.json'),
                  plan_sha256=sha(plan_path),revisions=plan['revisions'],files=files,
                  total_bytes=sum(v['bytes'] for v in files.values()),gpu_validation=False,external_verified=False)
    atomic_json(root/'prepared_input_manifest.json',manifest)
    return dict(status='PASS_LOCAL_CPU_ONLY',manifest_sha256=sha(root/'prepared_input_manifest.json'),files=len(files),bytes=manifest['total_bytes'],external_verified=False)


def verify_import(root, expected):
    root=Path(root).resolve();path=root/'prepared_input_manifest.json'
    if sha(path)!=expected:raise RuntimeError('Published input manifest differs')
    manifest=json.loads(path.read_text())
    for name,r in manifest['files'].items():
        if not verify_record(checked_target(root,name),r):raise RuntimeError('Published input content mismatch: '+name)
    return dict(status='PASS',manifest_sha256=expected,files=len(manifest['files']),scope='INDEPENDENT_INPUT_CONTENT_VERIFICATION_NOT_GPU_EXECUTION')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    q=s.add_parser('download');q.add_argument('--plan',required=True);q.add_argument('--root',required=True);q.add_argument('--groups',nargs='+',choices=['runtime','model','dataset','metric'],required=True)
    q=s.add_parser('install');q.add_argument('--release',required=True);q.add_argument('--root',required=True);q.add_argument('--prefix',required=True)
    q=s.add_parser('seal');q.add_argument('--root',required=True);q.add_argument('--release',required=True);q.add_argument('--plan',required=True)
    q=s.add_parser('verify-import');q.add_argument('--root',required=True);q.add_argument('--expected-manifest-sha256',required=True)
    a=p.parse_args()
    try:
        if a.action=='download':r=provision(a.plan,a.root,a.groups)
        elif a.action=='install':r=install(a.release,a.root,a.prefix)
        elif a.action=='seal':r=seal(a.root,a.release,a.plan)
        else:r=verify_import(a.root,a.expected_manifest_sha256)
        print(json.dumps(r,sort_keys=True,indent=2))
    except Exception as e:
        print(json.dumps(dict(status='FAIL',type=type(e).__name__,error=str(e).split('?')[0])),file=sys.stderr);sys.exit(1)
