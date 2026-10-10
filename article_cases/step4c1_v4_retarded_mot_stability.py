#!/usr/bin/env python3
"""Step 4C1 V4 — one-step retarded-MOT long-horizon stability validation."""
from __future__ import annotations
import argparse, gc, json, math, time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from EMsolver.mas_preprocessor import generate_rounded_two_arm_dipole_points

EPS0=8.8541878128e-12; C0=299792458.0; DT=1e-9; TINY=1e-300
DEPTH_FRAC=0.80; COLLOCATION_FACTOR=2.0; LAMBDA_REL=1e-4
POWER_DEFAULT=300; FREE_DEFAULT=200; TAIL_WINDOW=30
SEEDS_DEFAULT=(20260929,20260930,20261001); GATE=1.001

def settings(quick):
    return dict(mode='quick' if quick else 'formal', total_length_m=16e-3 if quick else 60e-3,
                sources_per_arm=180 if quick else 900, n_phi=28 if quick else 72,
                n_cyl=28 if quick else 96, n_hemi=10 if quick else 24,
                obs_chunk=16 if quick else 8)

def geometry(s):
    return generate_rounded_two_arm_dipole_points(radius=1e-3,total_length=s['total_length_m'],
        feed_gap=2e-3,depth_frac=DEPTH_FRAC,sources_per_arm=s['sources_per_arm'],
        collocation_factor=COLLOCATION_FACTOR,n_phi=s['n_phi'],n_cyl=s['n_cyl'],n_hemi=s['n_hemi'])

def pair_blocks(obs,src,w):
    r=obs[:,None,:]-src[None,:,:]; R=np.linalg.norm(r,axis=2)
    if np.any(R<=0): raise RuntimeError('zero source-observation distance')
    rr=r[:,:,:,None]*r[:,:,None,:]; I=np.eye(3)[None,None,:,:]; W=w[None,:,None,None]
    ce=1/(4*np.pi*EPS0)
    Gp=ce*W*(3*rr/R[:,:,None,None]**5-I/R[:,:,None,None]**3)
    Gk=ce*W*(3*rr/R[:,:,None,None]**4-I/R[:,:,None,None]**2)/C0
    Gd=ce*W*(rr/R[:,:,None,None]**3-I/R[:,:,None,None])/(C0*C0)
    return Gp,Gk,Gd,R

def project(ot,T,st):
    return np.einsum('mua,mjab,jvb->mujv',ot,T,st,optimize=True).reshape(2*len(ot),2*len(st))

def assemble(g,scheme,chunk):
    obs=np.asarray(g.solve_obs_points,float); src=np.asarray(g.ant_src_points,float); w=np.asarray(g.source_weights,float)
    ot=np.stack([g.solve_tangent1,g.solve_tangent2],axis=1); st=np.stack([g.source_tangent1,g.source_tangent2],axis=1)
    Hp=np.empty((2*len(obs),2*len(src))); Hk=np.empty_like(Hp); An=np.empty_like(Hp)
    rmin=math.inf; rmax=0.; amin=math.inf; amax=-math.inf
    for a in range(0,len(obs),chunk):
        b=min(len(obs),a+chunk); Gp,Gk,Gd,R=pair_blocks(obs[a:b],src,w); rmin=min(rmin,float(R.min())); rmax=max(rmax,float(R.max()))
        if scheme=='be':
            hp=Gp; hk=-Gd/DT; an=DT*Gp+Gk+Gd/DT
        else:
            alpha=1-R/(C0*DT); amin=min(amin,float(alpha.min())); amax=max(amax,float(alpha.max()))
            if np.any(alpha<0) or np.any(alpha>1): raise RuntimeError(f'one-step MOT invalid: alpha={alpha.min()}..{alpha.max()}')
            A=alpha[:,:,None,None]; A2=(alpha*alpha)[:,:,None,None]
            hp=Gp
            hk=Gp*(DT*(A-.5*A2))+Gk*(1-A)-Gd/DT
            an=Gp*(.5*DT*A2)+Gk*A+Gd/DT
        sl=slice(2*a,2*b); Hp[sl]=project(ot[a:b],hp,st); Hk[sl]=project(ot[a:b],hk,st); An[sl]=project(ot[a:b],an,st)
        del Gp,Gk,Gd,R,hp,hk,an; gc.collect()
    d=dict(min_distance_m=rmin,max_distance_m=rmax,min_delay_s=rmin/C0,max_delay_s=rmax/C0,max_delay_over_dt=(rmax/C0)/DT)
    if scheme!='be': d.update(alpha_min=amin,alpha_max=amax)
    return Hp,Hk,An,d

def smax(A,n=8):
    rng=np.random.default_rng(1234); x=rng.normal(size=A.shape[1]); x/=max(np.linalg.norm(x),TINY)
    for _ in range(n):
        y=A@x; z=A.T@y; nz=np.linalg.norm(z)
        if nz<=TINY:return 1.
        x=z/nz
    return float(np.linalg.norm(A@x))

def prepare(A,w):
    rs=np.repeat(np.sqrt(np.asarray(w)/max(float(np.mean(w)),TINY)),2); A*=rs[:,None]
    cs=np.linalg.norm(A,axis=0); cs=np.where(cs>TINY,cs,1.); A/=cs[None,:]
    sm=max(smax(A),TINY); alpha=LAMBDA_REL*sm; H=A.T@A; H.flat[::H.shape[0]+1]+=alpha*alpha
    cf=cho_factor(H,lower=True,overwrite_a=True,check_finite=False)
    return dict(rs=rs,cs=cs,As=A,cf=cf,smax=sm,ridge_alpha=alpha)

def solve(prep,rhs):
    bw=np.asarray(rhs)*prep['rs']; y=cho_solve(prep['cf'],prep['As'].T@bw,check_finite=False); return y/prep['cs']

def wrms(g,q):
    q=np.asarray(q).reshape(-1,2); w=np.asarray(g.source_weights); return float(np.sqrt(np.sum(w*np.sum(q*q,axis=1))/max(float(np.sum(w)),TINY)))

def snorm(g,q,p): return math.sqrt(wrms(g,q)**2+wrms(g,np.asarray(p)/DT)**2)

def init(g,seed):
    rng=np.random.default_rng(seed); q=rng.standard_normal(2*g.num_sources); p=DT*rng.standard_normal(2*g.num_sources); n=snorm(g,q,p); return q/n,p/n

def step(g,scheme,Hp,Hk,prep,q,p):
    qn=solve(prep,-(Hp@p+Hk@q)); pn=p+DT*qn if scheme=='be' else p+.5*DT*(q+qn); return qn,pn

def run_seed(g,scheme,Hp,Hk,prep,seed,npi,nfree):
    q,p=init(g,seed); growth=[]; secs=0.
    for _ in range(npi):
        t=time.perf_counter(); qn,pn=step(g,scheme,Hp,Hk,prep,q,p); secs+=time.perf_counter()-t; gn=snorm(g,qn,pn); growth.append(float(gn))
        if not np.isfinite(gn) or gn<=0:return dict(seed=seed,finite=False,power_tail_median_growth=float('inf'),free_tail_median_growth=float('inf'),apply_seconds_mean=secs/max(len(growth),1))
        q,p=qn/gn,pn/gn
    pt=float(np.median(growth[-TAIL_WINDOW:])); norms=[snorm(g,q,p)]; ratios=[]; finite=True
    for _ in range(nfree):
        t=time.perf_counter(); qn,pn=step(g,scheme,Hp,Hk,prep,q,p); secs+=time.perf_counter()-t; nn=snorm(g,qn,pn)
        if not np.isfinite(nn) or nn<=0: finite=False; break
        ratios.append(float(nn/max(norms[-1],TINY))); norms.append(float(nn)); q,p=qn,pn
    ft=float(np.median(ratios[-TAIL_WINDOW:])) if ratios else float('inf'); applies=len(growth)+len(ratios)
    idx=np.linspace(0,len(growth)-1,min(20,len(growth)),dtype=int)
    return dict(seed=seed,finite=finite,power_iterations_completed=len(growth),power_tail_median_growth=pt,
      power_growth_samples=[dict(iteration=int(i+1),growth=float(growth[i])) for i in idx],free_steps_completed=len(norms)-1,
      free_initial_state_norm=float(norms[0]),free_final_state_norm=float(norms[-1]),free_max_state_norm=float(np.max(norms)),
      free_tail_median_growth=ft,apply_seconds_total=float(secs),apply_seconds_mean=float(secs/max(applies,1)))

def run_scheme(g,scheme,chunk,seeds,npi,nfree):
    print(f'  assembling {scheme} ...',flush=True); t=time.perf_counter(); Hp,Hk,An,delay=assemble(g,scheme,chunk); ass=time.perf_counter()-t
    mem=(Hp.nbytes+Hk.nbytes+An.nbytes)/1024**3; print(f'    assembly={ass:.3f}s matrices={mem:.3f}GiB',flush=True)
    t=time.perf_counter(); prep=prepare(An,g.solve_weights); fac=time.perf_counter()-t
    reps=[]
    for sd in seeds:
        print(f'    seed={sd}: {npi} power + {nfree} free',flush=True); r=run_seed(g,scheme,Hp,Hk,prep,sd,npi,nfree); reps.append(r)
        print(f"      power={r['power_tail_median_growth']:.12g} free={r['free_tail_median_growth']:.12g} finite={r['finite']}",flush=True)
    out=dict(scheme=scheme,assembly_seconds=ass,ridge_factor_seconds=fac,stored_operator_matrix_GiB=mem,
      mean_recurrence_apply_seconds=float(np.mean([r['apply_seconds_mean'] for r in reps])),delay_geometry=delay,
      all_seeds_finite=all(r['finite'] for r in reps),max_power_tail_growth_across_seeds=max(r['power_tail_median_growth'] for r in reps),
      max_free_tail_growth_across_seeds=max(r['free_tail_median_growth'] for r in reps),seeds=reps)
    del Hp,Hk,An,prep; gc.collect(); return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--quick',action='store_true'); p.add_argument('--out',default='step4c1_v4_retarded_mot_stability')
    p.add_argument('--power-iterations',type=int,default=POWER_DEFAULT); p.add_argument('--free-steps',type=int,default=FREE_DEFAULT)
    p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS_DEFAULT)); p.add_argument('--skip-be-control',action='store_true'); a=p.parse_args()
    s=settings(a.quick); out=Path('article_cases_output')/a.out; out.mkdir(parents=True,exist_ok=True)
    print('Step 4C1 V4 retarded-MOT long-horizon stability'); print('  mode:',s['mode']); print('  sources/arm:',s['sources_per_arm']); print('  power/free:',a.power_iterations,a.free_steps); print('  seeds:',a.seeds)
    g=geometry(s)
    obs=np.asarray(g.solve_obs_points); src=np.asarray(g.ant_src_points); rmin=math.inf; rmax=0.
    for i in range(0,len(obs),s['obs_chunk']):
        R=np.linalg.norm(obs[i:i+s['obs_chunk'],None,:]-src[None,:,:],axis=2); rmin=min(rmin,float(R.min())); rmax=max(rmax,float(R.max()))
    md=(rmax/C0)/DT; one=md<1.; print(f'  max delay/dt: {md:.9g}',flush=True)
    if not one: raise RuntimeError('one-step MOT invalid: max delay >= dt')
    be=None if a.skip_be_control else run_scheme(g,'be',s['obs_chunk'],a.seeds,a.power_iterations,a.free_steps)
    mot=run_scheme(g,'mot',s['obs_chunk'],a.seeds,a.power_iterations,a.free_steps)
    motpass=mot['all_seeds_finite'] and mot['max_power_tail_growth_across_seeds']<=GATE and mot['max_free_tail_growth_across_seeds']<=GATE
    bebad=None if be is None else (be['max_power_tail_growth_across_seeds']>1.001 or be['max_free_tail_growth_across_seeds']>1.001)
    runtime=dict(same_number_of_history_dense_matvecs_per_step=2,same_fixed_regularized_solve_class_per_step=True)
    if be is not None:
        runtime.update(mot_over_be_operator_assembly_time_ratio=mot['assembly_seconds']/max(be['assembly_seconds'],TINY),
          mot_over_be_mean_recurrence_apply_time_ratio=mot['mean_recurrence_apply_seconds']/max(be['mean_recurrence_apply_seconds'],TINY),
          mot_over_be_stored_operator_matrix_ratio=mot['stored_operator_matrix_GiB']/max(be['stored_operator_matrix_GiB'],TINY))
    accepted=bool(one and motpass)
    rep=dict(step='4C1-V4',purpose='one-step retarded-MOT long-horizon stability',mode=s['mode'],production_settings=dict(total_length_m=s['total_length_m'],sources_per_arm=s['sources_per_arm'],collocation_points_per_arm=2*s['sources_per_arm'],depth_frac=DEPTH_FRAC,lambda_rel=LAMBDA_REL,operator_softening_distance_m=0.,dt_s=DT),stability_protocol=dict(power_iterations=a.power_iterations,free_steps=a.free_steps,tail_window=TAIL_WINDOW,seeds=a.seeds),one_step_retardation_precheck=dict(min_distance_m=rmin,max_distance_m=rmax,min_delay_s=rmin/C0,max_delay_s=rmax/C0,max_delay_over_dt=md,one_step_MOT_valid=one),time_local_be_control=be,time_local_be_weak_instability_reproduced=bebad,retarded_mot_candidate=mot,retarded_growth_gate=GATE,runtime_overhead=runtime,accepted=accepted,decision='STEP4C1_V4_PASS_proceed_to_STEP4C2_V4_retarded_frequency_regression' if accepted else 'STEP4C1_V4_FAIL_do_not_return_to_PIC_replay',scope_note='PASS establishes homogeneous long-horizon stability only; C2 V4 must validate 3/6/9 MHz response.')
    (out/'step4c1_v4_report.json').write_text(json.dumps(rep,indent=2))
    lines=['Step 4C1 V4 retarded-MOT long-horizon stability',f'accepted: {accepted}',f'mode: {s["mode"]}',f'max delay/dt: {md:.9g}']
    if be is not None: lines += [f'BE max power growth: {be["max_power_tail_growth_across_seeds"]:.12g}',f'BE max free growth: {be["max_free_tail_growth_across_seeds"]:.12g}']
    lines += [f'MOT max power growth: {mot["max_power_tail_growth_across_seeds"]:.12g}',f'MOT max free growth: {mot["max_free_tail_growth_across_seeds"]:.12g}']
    if be is not None: lines += [f'MOT/BE assembly ratio: {runtime["mot_over_be_operator_assembly_time_ratio"]:.6g}',f'MOT/BE recurrence-apply ratio: {runtime["mot_over_be_mean_recurrence_apply_time_ratio"]:.6g}',f'MOT/BE stored-matrix ratio: {runtime["mot_over_be_stored_operator_matrix_ratio"]:.6g}']
    lines.append('decision: '+rep['decision']); (out/'step4c1_v4_summary.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
if __name__=='__main__': main()
