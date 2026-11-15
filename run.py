"""Run current JefiPIC-MAS examples and reproduction commands."""
from pathlib import Path
import argparse,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['verify','cpu','sphere-cpu','figures','rf','sphere','scuff-mas','warpx-inputs'])
    args,extra=ap.parse_known_args();env=dict(os.environ)
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:env[key]='4'
    def run(p,a=()):subprocess.run([sys.executable,str(ROOT/p),*map(str,a)],cwd=ROOT,env=env,check=True)
    if args.command in ['verify','cpu']:
        if extra:ap.error('This command takes no additional arguments')
        from release_checks import verify_all
        verify_all(ROOT)
        if args.command=='verify':return 0
    if args.command=='cpu':
        for script,name in [('qualify_reflective_cic_fix.py','reflected_CC'),('qualify_direct_incident_cpu.py','direct_incident'),('qualify_feedback_cpu.py','EM_feedback')]:
            dst=ROOT/'outputs/checks'/name;dst.mkdir(parents=True,exist_ok=True)
            run('examples/two_arm/'+script,[dst])
        print('PASSED current CPU arithmetic checks; native GPU gates remain separate.')
    elif args.command=='sphere-cpu':
        dst=ROOT/'outputs/checks/sphere';dst.mkdir(parents=True,exist_ok=True)
        run('examples/sphere/qualify_cpu.py',['--out',dst,*extra])
    else:
        paths={'figures':'scripts/plot_figures.py','rf':'examples/two_arm/run.py','sphere':'examples/sphere/run.py','scuff-mas':'scripts/reproduce_scuff_mas.py','warpx-inputs':'scripts/prepare_warpx_inputs.py'}
        run(paths[args.command],extra)
    return 0
if __name__=='__main__':sys.exit(main())
