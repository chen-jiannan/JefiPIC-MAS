'PEC sphere and particle-cloud example: run. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from pathlib import Path
import argparse,csv,datetime,io,json,os,shutil,subprocess,sys,zipfile
from campaign_tools import verify_package,phase,wait_idle
ROOT=Path(__file__).resolve().parent
SOURCE_ROOT=ROOT.parents[1]
__import__('sys').path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]

def accept(folder,smoke=False):
    r=json.loads((folder/'report.json').read_text())
    if r['status']!='complete' or not r['no_particle_collection_or_outer_crossing']:
        raise RuntimeError('Incomplete run or particle crossing event; retain diagnostic output.')
    if not smoke and (not r['complete_to_common_end'] or r['samples_completed']!=87):
        raise RuntimeError('Common physical end time not reached')
    if r['metal'] and (r['boundary_revision']!='normalized-mixed-hermite-mot-v2' or r['boundary_theta']!=.6):
        raise RuntimeError('Unexpected boundary version')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',default='0')
    ap.add_argument('--busy',choices=['share','wait','fail'],default='share')
    ap.add_argument('--gpu-wait-minutes',type=float,default=60)
    ap.add_argument('--verify-only',action='store_true');a=ap.parse_args()
    verify_package()
    if a.verify_only:return 0
    out=ROOT/('sphere_mixed_return_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    summary={'status':'running','stages':[],'gpu_policy':a.busy,'scope':'accuracy only; unchanged unsplit CC PIC sources; normalized mixed Hermite MOT theta=0.6'}
    env=dict(os.environ);env['JEFIPIC_PROFILE']='0'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):env[key]='1'
    code=0
    try:
        phase('CPU qualification',[sys.executable,ROOT/'qualify_cpu.py','--out',out/'cpu_qualification'],out/'cpu_qualification',env)
        info=subprocess.run(['nvidia-smi','-i',a.gpu,'--query-gpu=uuid,name,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        row=next(csv.reader(io.StringIO(info.stdout)));uuid=row[0].strip();summary['GPU']=row;env['CUDA_VISIBLE_DEVICES']=uuid
        wait_idle(uuid,a.busy,a.gpu_wait_minutes)
        phase('Native CUDA gate',[sys.executable,ROOT/'validate_gpu.py','--gpu','0','--out',out/'gpu_qualification','--fixture',out/'cpu_qualification/gpu_replay_fixture.npz'],out/'gpu_qualification',env,uuid)
        if json.loads((out/'gpu_qualification/gpu_qualification.json').read_text())['status']!='NATIVE_CUDA_QUALIFICATION_PASSED':
            raise RuntimeError('Native CUDA gate did not pass')
        for name,grid,metal,limit in [('smoke',8,True,3),('jefi_n36_metal',36,True,None),('jefi_n36_no_metal',36,False,None)]:
            wait_idle(uuid,a.busy,a.gpu_wait_minutes)
            cmd=[sys.executable,ROOT/'simulate.py','--repo',SOURCE_ROOT,'--case',ROOT/'input','--out',out/name,'--gpu','0','--grid',str(grid),'--sources','192','--dt-r-over-c','2.2','--metal' if metal else '--no-metal']
            if limit is not None:cmd+=['--max-steps',str(limit)]
            phase(name,cmd,out/name,env,uuid);accept(out/name,smoke=limit is not None);summary['stages'].append(name)
        summary['status']='JEFIPIC_MIXED_SPHERE_COMPLETE_EXISTING_WARPX_COMPARISON_PENDING'
    except KeyboardInterrupt:summary['status']='interrupted';code=130
    except Exception as exc:summary.update(status='stopped_retain_outputs',error=repr(exc));print('STOP:',exc,flush=True);code=1
    finally:
        (out/'campaign.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        dst=out/'provenance';dst.mkdir(exist_ok=True)
        for name in ('revision_manifest.json','calibration.json','source_preservation.json','simulate.py','boundary.py','qualify_cpu.py','run.py'):
            shutil.copy2(ROOT/name,dst/name)
        shutil.copy2(ROOT/'input/case.json',dst/'case.json')
        shutil.copy2(SOURCE_ROOT/'EMsolver/mas_mixed_mot.py',dst/'mas_mixed_mot.py')
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in sorted(out.rglob('*')):
                if path.is_file() and path.suffix!='.zip':z.write(path,path.relative_to(out.parent))
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise RuntimeError('Return archive CRC failure')
        print('RETURN THIS ZIP (also on failure):',archive,flush=True)
    return code

if __name__=='__main__':sys.exit(main())
