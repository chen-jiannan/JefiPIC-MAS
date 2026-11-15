"""Matched analytic-sphere component comparison using real Gmsh/GetDP executables.
No PIC or GPU run. MAS calls are the unmodified released static-sphere routines.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, platform, re, subprocess, sys, time

def run(cmd,cwd,log,env):
    t=time.perf_counter()
    p=subprocess.run([str(x) for x in cmd],cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    wall=time.perf_counter()-t
    (cwd/log).write_text(p.stdout,encoding='utf-8')
    if p.returncode:raise RuntimeError(f'{cmd[0]} rc={p.returncode}; inspect {cwd/log}')
    return wall,p.stdout

def load_source(path):
    spec=importlib.util.spec_from_file_location('released_sphere_case',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def worker(a):
    import numpy as np
    mod=load_source(a.mas_source)
    t=time.perf_counter()
    surf,normals,areas,src,q,vf,res,cond,pcond,soft=mod.solve_case_arrays(a.mas_worker,.004,1.,.35*.004,0.,0.)
    solve_s=time.perf_counter()-t
    npz={};metrics={}
    t=time.perf_counter()
    for fac in (1.05,1.1,2.):
        pts,_,_=mod.fibonacci_sphere_points(a.probes,fac*.004)
        e=mod.point_charge_field(pts,src,q);e[:,2]+=1.
        phi=mod.potential_matrix(pts,src)@q-pts[:,2]
        exact=mod.analytic_conducting_sphere_field(pts,.004,1.)
        phiex=mod.analytic_conducting_sphere_potential(pts,.004,1.)
        key=f'{fac:.2f}'
        metrics[key]=dict(E_rel_l2=float(np.linalg.norm(e-exact)/np.linalg.norm(exact)),Phi_rel_l2=float(np.linalg.norm(phi-phiex)/np.linalg.norm(phiex)))
        npz['points_'+key]=pts;npz['E_'+key]=e;npz['Phi_'+key]=phi
    query_s=time.perf_counter()-t
    dipole=np.sum(q[:,None]*src,axis=0)
    exactp=4*np.pi*mod.EPS0*.004**3
    r=dict(method='MAS',n_sources=a.mas_worker,unknowns=a.mas_worker+1,
        solve_case_arrays_wall_s=solve_s,query_wall_s=query_s,
        timed_scope='Unmodified solve_case_arrays includes geometry, matrix, linear solve, and two condition-number diagnostics; query uses common independent probes.',
        metrics=metrics,dipole_rel_error=float(np.linalg.norm(dipole-np.array([0,0,exactp]))/exactp),
        net_charge_C=float(np.sum(q)),floating_potential_V=float(vf),KKT_condition=cond,potential_matrix_condition=pcond,
        source_sha256=hashlib.sha256(a.mas_source.read_bytes()).hexdigest())
    np.savez_compressed(a.out/'fields.npz',**npz)
    (a.out/'result.json').write_text(json.dumps(r,indent=2),encoding='utf-8')

def fem_geometry(h,outer):
    return f'''SetFactory("OpenCASCADE");
R=0.004; B={outer:.17g}; h={h*.004:.17g};
Sphere(1)={{0,0,0,R}}; Sphere(2)={{0,0,0,B*R}};
air()=BooleanDifference{{Volume{{2}};Delete;}}{{Volume{{1}};Delete;}};
inner()=Surface In BoundingBox{{-1.01*R,-1.01*R,-1.01*R,1.01*R,1.01*R,1.01*R}};
outer()=Boundary{{Volume{{air()}};}}; outer()-=inner();
Physical Volume(1)={{air()}}; Physical Surface(2)={{inner()}}; Physical Surface(3)={{outer()}};
Field[1]=MathEval; Field[1].F=Sprintf("%g+0.3*(Sqrt(x*x+y*y+z*z)-%g)",h,R);
Background Field=1;
Mesh.MeshSizeMin=h; Mesh.MeshSizeMax=2*R;
Mesh.MeshSizeFromPoints=0; Mesh.MeshSizeFromCurvature=0; Mesh.MeshSizeExtendFromBoundary=0;
Mesh.MshFileVersion=2.2; Mesh.Binary=0; Mesh.ElementOrder=2;
Mesh.MaxNumThreads3D=1; General.NumThreads=1; Mesh.RandomSeed=1;
'''

def fem_problem(outer,probes):
    base=(Path(__file__).parent/'smoke/sphere.pro').read_text()
    base=re.sub(r'R=0.004; B=8; E0=1.; N=128;',f'R=0.004; B={outer:.17g}; E0=1.; N={probes};',base)
    base=base[:base.index('PostOperation {')]
    base+='PostOperation {\n { Name Sample; NameOfPostProcessing Fields; Operation {\n'
    for fac in (1.05,1.1,2.):
        point=f'{{{fac:.17g}*R*Sqrt[1-(1-2*($A+0.5)/N)^2]*Cos[Pi*(3-Sqrt[5])*$A],{fac:.17g}*R*Sqrt[1-(1-2*($A+0.5)/N)^2]*Sin[Pi*(3-Sqrt[5])*$A],{fac:.17g}*R*(1-2*($A+0.5)/N)}}'
        for quantity in ('E','Phi'):
            base+=f' Print[{quantity}, OnGrid {point} {{0:N-1,0,0}}, Format Table, File "{quantity}_{fac:.2f}.txt"];\n'
    return base+' } }\n}\n'

def parse_fem(folder,mod,probes):
    import numpy as np
    result={};data={}
    for fac in (1.05,1.1,2.):
        key=f'{fac:.2f}';ef=np.loadtxt(folder/f'E_{key}.txt');pf=np.loadtxt(folder/f'Phi_{key}.txt')
        pts,_,_=mod.fibonacci_sphere_points(probes,fac*.004)
        assert len(ef)==probes and len(pf)==probes
        assert np.max(abs(ef[:,2:5]-pts))<1e-13
        assert np.max(abs(pf[:,2:5]-pts))<1e-13
        e=ef[:,-3:];phi=pf[:,-1]
        assert np.isfinite(e).all() and np.isfinite(phi).all()
        ex=mod.analytic_conducting_sphere_field(pts,.004,1.);px=mod.analytic_conducting_sphere_potential(pts,.004,1.)
        result[key]=dict(E_rel_l2=float(np.linalg.norm(e-ex)/np.linalg.norm(ex)),Phi_rel_l2=float(np.linalg.norm(phi-px)/np.linalg.norm(px)))
        data['points_'+key]=pts;data['E_'+key]=e;data['Phi_'+key]=phi
    pts=data['points_2.00'];rr=np.linalg.norm(pts,axis=1);nn=pts/rr[:,None]
    scattered=data['Phi_2.00']+pts[:,2]
    coeff=np.linalg.lstsq(nn,scattered,rcond=None)[0]
    dipole=4*np.pi*mod.EPS0*(.008**2)*coeff
    exactp=4*np.pi*mod.EPS0*.004**3
    np.savez_compressed(folder/'fields.npz',**data)
    return result,float(np.linalg.norm(dipole-np.array([0,0,exactp]))/exactp)

def main(a):
    import numpy as np
    a.out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
        env[key]='1'
    env['MPLCONFIGDIR']=str((a.out/'mpl_cache').resolve())
    Path(env['MPLCONFIGDIR']).mkdir()
    mod=load_source(a.mas_source)
    metadata=dict(python=sys.version,platform=platform.platform(),cpu=platform.processor(),logical_cpu_count=os.cpu_count(),numpy=np.__version__,
        gmsh_executable=str(a.gmsh),getdp_executable=str(a.getdp),threads={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        geometry='R=4mm, uniform E0=1 V/m z, induced neutral-sphere solution; independent Fibonacci probes at 1.05R,1.10R,2R.',
        FEM='GetDP scalar scattered potential, curved P2 tetrahedra, inner v=E0*z, outer partial_r(v)+2*v/(B*R)=0. Robin condition is exact for the continuous dipole solution.',
        comparison_scope='Static boundary component only; not RF-PIC throughput or a complete-code GPU comparison.',arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    for exe,key in ((a.gmsh,'gmsh'),(a.getdp,'getdp')):
        p=subprocess.run([str(exe),'-version'],capture_output=True,text=True)
        metadata[key+'_version']=(p.stdout+p.stderr).strip();metadata[key+'_exe_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest()
    try:
        from threadpoolctl import threadpool_info
        metadata['parent_threadpools']=threadpool_info()
    except ImportError:metadata['parent_threadpools']='unavailable; environment limits are set before worker processes'
    (a.out/'environment.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    results=[]
    for h in [float(s) for s in a.h.split(',')]:
        case=a.out/f'fem_h{h:.3f}';case.mkdir()
        (case/'sphere.geo').write_text(fem_geometry(h,a.outer))
        (case/'sphere.pro').write_text(fem_problem(a.outer,a.probes))
        mt,mlog=run([a.gmsh,'sphere.geo','-3','-format','msh2','-o','sphere.msh','-v','4','-nopopup'],case,'mesh.log',env)
        nm=re.search(r'(\d+) nodes (\d+) elements',mlog)
        for trial in range(a.trials):
            trialdir=case/f'trial_{trial}';trialdir.mkdir()
            shutil_mesh=case/'sphere.msh'
            wall,log=run([a.getdp,case/'sphere.pro','-msh',shutil_mesh,'-name',trialdir/'sphere','-solve','SolveSphere','-pos','Sample','-v','4'],trialdir,'solve.log',env)
            # GetDP resolves post-processing file names relative to its working directory.
            metrics,dipole=parse_fem(trialdir,mod,a.probes)
            dof=re.search(r'System 1/1: (\d+) Dofs',log)
            residuals=re.findall(r'KSP Residual norm ([\d.eE+-]+)',log)
            row=dict(method='GetDP_P2',h_over_R=h,outer_radius_over_R=a.outer,trial=trial,n_mesh_nodes=int(nm[1]) if nm else None,
                unknowns=int(dof[1]) if dof else None,mesh_wall_s=mt,process_solve_and_query_wall_s=wall,
                process_scope='GetDP process: imports/native startup, mesh read, preassembly, matrix assembly, sparse solve, save, common field/potential queries; mesh generation separate.',
                metrics=metrics,dipole_rel_error=dipole,linear_residual_norms=[float(x) for x in residuals])
            results.append(row)
            (trialdir/'result.json').write_text(json.dumps(row,indent=2))
            (a.out/'results.json').write_text(json.dumps(results,indent=2))
            print('FEM',h,trial,'DOF',row['unknowns'],'E110',metrics['1.10']['E_rel_l2'],'wall',wall,flush=True)
    for n in [int(s) for s in a.mas_points.split(',')]:
        for trial in range(a.trials):
            trialdir=a.out/f'mas_N{n}'/f'trial_{trial}';trialdir.mkdir(parents=True)
            wall,_=run([sys.executable,Path(__file__).resolve(),'--mas-worker',n,'--mas-source',a.mas_source,'--probes',a.probes,'--out',trialdir],trialdir,'solve.log',env)
            row=json.loads((trialdir/'result.json').read_text());row.update(trial=trial,process_solve_and_query_wall_s=wall)
            results.append(row)
            (a.out/'results.json').write_text(json.dumps(results,indent=2))
            print('MAS',n,trial,'E110',row['metrics']['1.10']['E_rel_l2'],'wall',wall,flush=True)
    print('COMPLETE',a.out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gmsh',type=Path)
    p.add_argument('--getdp',type=Path)
    p.add_argument('--mas-source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--h',default='.4,.2,.1')
    p.add_argument('--outer',type=float,default=8.)
    p.add_argument('--probes',type=int,default=1024)
    p.add_argument('--mas-points',default='48,96,192,320')
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--mas-worker',type=int)
    a=p.parse_args()
    a.mas_source=a.mas_source.resolve();a.out=a.out.resolve()
    if a.mas_worker:worker(a)
    else:
        a.gmsh=a.gmsh.resolve();a.getdp=a.getdp.resolve();main(a)
