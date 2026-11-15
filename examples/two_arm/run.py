'Driven two-arm example: run. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from pathlib import Path
import argparse,csv,datetime,hashlib,io,json,os,shutil,subprocess,sys,zipfile
from campaign_tools import phase,wait_idle,apps
ROOT=Path(__file__).resolve().parent
SOURCE_ROOT=ROOT.parents[1]
__import__('sys').path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]

def verify():
    from release_checks import verify_case
    verify_case(SOURCE_ROOT, 'examples/two_arm/revision_manifest.json')
    cert=json.loads((ROOT/'validation/CPU_candidate_certificate.json').read_text())
    arithmetic=json.loads((ROOT/'validation/feedback_CPU_arithmetic.json').read_text())
    fixture=hashlib.sha256((ROOT/'validation/em_feedback_fixture.npz').read_bytes()).hexdigest()
    if not all(cert['checks'].values()) or arithmetic['status']!='NATIVE_KERNEL_ARITHMETIC_CPU_PASSED_CUDA_GATE_REQUIRED' or arithmetic['fixture_sha256']!=fixture:
        raise RuntimeError('Passing scoped CPU evidence/fixture missing')
    if not arithmetic.get('source_sha256') or any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest for name,digest in arithmetic['source_sha256'].items()):
        raise RuntimeError('Current CPU arithmetic report does not match interface sources')
    print('PASSED current source manifest and scoped CPU checks. Any new GPU run still performs its native gates.',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',default='0');ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--mode',choices=('all','replay'),default='all');ap.add_argument('--busy',choices=('share','wait','fail'),default='share')
    ap.add_argument('--gpu-wait-minutes',type=float,default=60);ap.add_argument('--profile',action='store_true');ap.add_argument('--verify-only',action='store_true')
    a=ap.parse_args();verify()
    if a.threads<1 or a.gpu_wait_minutes<0:ap.error('Positive thread count and nonnegative wait time required')
    if a.verify_only:return 0
    out=ROOT/('compatible_return_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
    summary=dict(status='RUNNING',arguments=vars(a),stages=[],CPU_long_qualification='passed_scoped',CPU_arithmetic_qualification='pending',native_CUDA_qualification='not_run',full_coupled_qualification='not_run')
    env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):env[key]=str(a.threads)
    env['JEFIPIC_PROFILE']='1' if a.profile else '0';code=0;pmon=None;activity=None;stage='cpu'
    try:
        for name,file,folder in (
            ('CPU reflected CIC','qualify_reflective_cic_fix.py','cpu_CC'),
            ('CPU direct incident arithmetic','qualify_direct_incident_cpu.py','cpu_direct'),
            ('CPU electric+magnetic native-kernel arithmetic','qualify_feedback_cpu.py','cpu_feedback')):
            dst=out/folder;phase(name,[sys.executable,ROOT/file,dst],dst,env);summary['stages'].append(name)
        summary['CPU_arithmetic_qualification']='passed';stage='native'
        info=subprocess.run(['nvidia-smi','-i',a.gpu,'--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],check=True,text=True,capture_output=True)
        gpu=next(csv.reader(io.StringIO(info.stdout)));index=gpu[0].strip();uuid=gpu[1].strip();summary['GPU']=[x.strip() for x in gpu];env['CUDA_VISIBLE_DEVICES']=uuid
        summary['resident_processes_before']=apps(uuid)
        print('Physical GPU',index,'->',uuid,'; child CUDA 0 maps to this GPU.',flush=True)
        try:
            activity=(out/'gpu_process_activity.log').open('w',encoding='utf-8')
            pmon=subprocess.Popen(['nvidia-smi','pmon','-i',index,'-s','um','-d','1','-o','DT'],stdout=activity,stderr=subprocess.STDOUT)
        except Exception as exc:summary['activity_monitor_error']=repr(exc)
        for name,file,folder in (
            ('Native direct incident gate','validate_direct_gpu.py','gpu_direct'),
            ('Native electric+magnetic recorded-state feedback','validate_feedback_gpu.py','gpu_feedback')):
            wait_idle(uuid,a.busy,a.gpu_wait_minutes);dst=out/folder
            phase(name,[sys.executable,ROOT/file,'--out',dst],dst,env,uuid);summary['stages'].append(name)
        gate_path=out/'gpu_feedback/report.json';gate=json.loads(gate_path.read_text())
        if gate['status']!='NATIVE_EM_FEEDBACK_PASSED':raise RuntimeError('Native feedback report did not pass')
        summary['native_CUDA_qualification']='passed'
        if a.mode=='replay':summary['status']='NATIVE_REPLAY_PASSED_FULL_NOT_RUN'
        else:
            stage='full';wait_idle(uuid,a.busy,a.gpu_wait_minutes);dst=out/'coupled_RF'
            phase('One original physical RF-PIC case, 4001 nodes',[sys.executable,ROOT/'run_simulation.py','--out',dst,'--calibration',ROOT/'calibration.json','--native-gate-report',gate_path],dst,env,uuid)
            r=json.loads((dst/'report.json').read_text())
            if r['status']!='qualified' or r['steps_completed']!=4001 or not r['precision_qualified'] or not r['late_window_precision_qualified'] or not r['two_way_execution_valid']:
                raise RuntimeError('Full 4001-node coupled report is not qualified')
            summary['full_coupled_qualification']='passed';summary['status']='FULL_4001_NODE_COMPATIBLE_RF_QUALIFIED';summary['stages'].append('Full coupled RF')
    except KeyboardInterrupt:
        summary.update(status='INTERRUPTED',interrupted_stage=stage);code=130
    except Exception as exc:
        summary.update(status='STOPPED_NOT_QUALIFIED',failed_stage=stage,error=repr(exc));code=1
        key={'cpu':'CPU_arithmetic_qualification','native':'native_CUDA_qualification','full':'full_coupled_qualification'}[stage];summary[key]='failed'
        print('STOP:',exc,flush=True)
    finally:
        if pmon is not None:
            if pmon.poll() is None:
                pmon.terminate()
                try:pmon.wait(timeout=5)
                except subprocess.TimeoutExpired:pmon.kill();pmon.wait()
            summary['pmon_returncode']=pmon.returncode
        if activity:activity.close()
        reports=[json.loads(p.read_text()) for p in out.glob('*/phase.json')]
        summary['profile_timing_eligible']=bool(a.profile and summary['full_coupled_qualification']=='passed' and a.busy in ('wait','fail') and 'activity_monitor_error' not in summary and summary.get('pmon_returncode') in (0,-15) and not any(x.get('foreign_GPU_process_observed') or x.get('monitor_errors') for x in reports))
        summary['timing_scope']='Synchronized diagnostic profile, not repeated uninstrumented throughput. Share mode does not qualify timings. pmon dash means unavailable, not zero.'
        (out/'campaign.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        prov=out/'provenance';prov.mkdir(exist_ok=True)
        for name in ('revision_manifest.json','geometry_provenance.json','actual_geometry.npz','calibration.json','two_arm_model.py','electric_magnetic_sources.py','mixed_boundary.py','magnetic_feedback.py','feedback_reference.py','reflecting_sources.py','incident_fields.py','run.py','run_simulation.py','validate_feedback_gpu.py'):
            shutil.copy2(ROOT/name,prov/name)
        shutil.copy2(SOURCE_ROOT/'EMsolver/mas_mixed_mot.py',prov/'mas_mixed_mot.py')
        shutil.copytree(ROOT/'validation',prov/'CPU_release_evidence',dirs_exist_ok=True)
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(out.parent).as_posix())
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise RuntimeError('Return archive CRC failed')
        print('STATUS:',summary['status'],flush=True);print('RETURN THIS ZIP (also on failure):',archive,flush=True)
    return code

if __name__=='__main__':sys.exit(main())
