'PEC sphere and particle-cloud example: campaign tools. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from pathlib import Path
import csv,datetime,hashlib,io,json,math,os,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parent
SOURCE_ROOT=ROOT.parents[1]
__import__('sys').path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def verify_package(root=ROOT):
    from release_checks import verify_case
    verify_case(SOURCE_ROOT, 'examples/sphere/revision_manifest.json')
    cpu=json.loads((root/'validation/cpu_qualification.json').read_text())
    if cpu['status']!='CPU_QUALIFICATION_PASSED_NATIVE_CUDA_GATE_REQUIRED':
        raise RuntimeError('No passing scoped sphere CPU qualification')
    print('PASSED current sphere manifest and CPU qualification.',flush=True)
    return json.loads((root/'revision_manifest.json').read_text())


def apps(uuid):
    p=subprocess.run(['nvidia-smi','-i',uuid,'--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    if p.returncode:raise RuntimeError(p.stderr or p.stdout)
    rows=[]
    for row in csv.reader(io.StringIO(p.stdout)):
        if not row or not row[0].strip() or row[0].strip().startswith('No running'):continue
        if len(row)!=3 or not row[0].strip().isdigit():raise RuntimeError('Unrecognized process row: '+repr(row))
        rows.append(dict(pid=int(row[0]),uuid=row[1].strip(),MiB=row[2].strip()))
    return rows

def descendants(pid):
    parent={}
    # Enumerate only /proc itself. A recursive glob can raise ESRCH while
    # entering a PID directory, before the per-file exception handler runs.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        p=entry/'status'
        try:
            for line in p.read_text().splitlines():
                if line.startswith('PPid:'):parent[int(p.parent.name)]=int(line.split()[1]);break
        except (OSError,ValueError):pass
    owned={pid}
    while True:
        found=owned|{p for p,pp in parent.items() if pp in owned}
        if found==owned:return owned
        owned=found

def wait_idle(uuid,mode,minutes):
    if mode not in ('share','wait','fail'):raise ValueError('Unknown GPU occupancy policy: '+str(mode))
    if mode=='share':
        busy=apps(uuid)
        if busy:print('Shared GPU allowed for accuracy run; proceeding without waiting. Existing PIDs:',[x['pid'] for x in busy],flush=True)
        return
    stop=time.monotonic()+minutes*60
    while True:
        busy=apps(uuid)
        if not busy:return
        if mode=='fail' or time.monotonic()>=stop:raise RuntimeError('GPU is busy; no new GPU phase started: '+json.dumps(busy))
        print('Selected GPU busy; waiting 15 s. PIDs:',[x['pid'] for x in busy],flush=True)
        time.sleep(15)

def phase(name,cmd,folder,env,uuid=None):
    folder.mkdir(parents=True,exist_ok=True);metadata=dict(name=name,command=[str(x) for x in cmd],start_utc=now(),monitor=[])
    started=time.monotonic();foreign_seen=False;proc=None;reader=None
    print('\nSTART',name,'log:',folder/'run.log',flush=True)
    log=(folder/'run.log').open('w',encoding='utf-8')
    try:
        proc=subprocess.Popen(metadata['command'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        def tee():
            for line in proc.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
        reader=threading.Thread(target=tee,daemon=True);reader.start();heartbeat=0
        while proc.poll() is None:
            elapsed=time.monotonic()-started
            if uuid:
                try:
                    own=descendants(proc.pid)
                    active=apps(uuid);own|=descendants(proc.pid)
                    foreign=[x for x in active if x['pid'] not in own]
                    metadata['monitor'].append(dict(utc=now(),elapsed_s=elapsed,processes=active,foreign=foreign))
                    if foreign and not foreign_seen:
                        foreign_seen=True
                        print('Foreign GPU process observed. Accuracy run continues; timing is not a performance result.',flush=True)
                except Exception as e:
                    errors=metadata.setdefault('monitor_errors',[]);errors.append(repr(e))
                    if len(errors)==1:print('GPU monitoring unavailable for this sample; accuracy calculation continues:',repr(e),flush=True)
            if elapsed-heartbeat>=60:
                print(name,'running;',round(elapsed/60,1),'min elapsed',flush=True);heartbeat=elapsed
            if proc.poll() is None:time.sleep(2)
        reader.join();metadata['returncode']=proc.returncode
        if proc.returncode:raise RuntimeError(name+' failed; read '+str(folder/'run.log'))
    except BaseException as e:
        metadata['error']=repr(e)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        if reader:reader.join(timeout=15)
        raise
    finally:
        if reader and reader.is_alive():reader.join(timeout=15)
        log.close()
        metadata.update(end_utc=now(),elapsed_wall_s=time.monotonic()-started,foreign_GPU_process_observed=foreign_seen,
            timing_scope='Monitored accuracy run only; not an A800/H100 performance measurement.')
        (folder/'phase.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('PASSED',name,flush=True)

def accept_coupled(folder,smoke=False):
    r=json.loads((folder/'report.json').read_text())
    if r['status']!='complete' or not r['no_particle_collection_or_outer_crossing']:raise RuntimeError('Incomplete/invalid particle run.')
    if not smoke:
        if not r['complete_to_common_end']:raise RuntimeError('Required common duration not reached.')
        policy=json.loads((ROOT/'acceptance_policy.json').read_text())
        for key,limit in policy['coupled_gates'].items():
            if not math.isfinite(r[key]) or r[key]>limit:raise RuntimeError(f'{key}={r[key]:.6g} exceeds predeclared gate {limit}; stop before next expensive run.')
    return r
