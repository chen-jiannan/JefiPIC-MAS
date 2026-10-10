#!/usr/bin/env python3
"""Step 4C2 V4 — causal retarded-MOT physical frequency-response regression."""
from __future__ import annotations
import argparse, gc, json, math, time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, gmres
import step2mb1_incident_normalization_feed_localization as mb1
from EMsolver.mas_preprocessor import generate_rounded_two_arm_dipole_points

EPS0=8.8541878128e-12; C0=299792458.0; DT=1e-9; TINY=1e-300
DEPTH_FRAC=0.80; COLLOCATION_FACTOR=2.0; LAMBDA_REL=1e-4
FORMAL_FREQS=(3e6,6e6,9e6); STRESS_FREQ=15e6
BOUNDARY_GATE=0.05; AMP_GATE=0.05; PHASE_GATE_DEG=5.; FIELD_DIFF_GATE=0.05; WEAK=0.20
GATE_REGIONS=("global","feed_rounded","feed_tips","feed_transition","outer_rounded","outer_tips","all_transition","side_mid")

def settings(quick):
    return dict(mode='quick' if quick else 'formal', total_length_m=16e-3 if quick else 60e-3,
                sources_per_arm=180 if quick else 900, n_phi=28 if quick else 72,
                n_cyl=28 if quick else 96, n_hemi=10 if quick else 24,
                obs_chunk=16 if quick else 8)

def geometry(s):
    return generate_rounded_two_arm_dipole_points(radius=1e-3,total_length=s['total_length_m'],feed_gap=2e-3,
        depth_frac=DEPTH_FRAC,sources_per_arm=s['sources_per_arm'],collocation_factor=COLLOCATION_FACTOR,
        n_phi=s['n_phi'],n_cyl=s['n_cyl'],n_hemi=s['n_hemi'])

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

def assemble_mot(g,chunk):
    obs=np.asarray(g.solve_obs_points,float); src=np.asarray(g.ant_src_points,float); w=np.asarray(g.source_weights,float)
    ot=np.stack([g.solve_tangent1,g.solve_tangent2],axis=1); st=np.stack([g.source_tangent1,g.source_tangent2],axis=1)
    Hp=np.empty((2*len(obs),2*len(src))); Hk=np.empty_like(Hp); An=np.empty_like(Hp)
    rmin=math.inf; rmax=0.; amin=math.inf; amax=-math.inf
    for a in range(0,len(obs),chunk):
        b=min(len(obs),a+chunk); Gp,Gk,Gd,R=pair_blocks(obs[a:b],src,w)
        alpha=1-R/(C0*DT); rmin=min(rmin,float(R.min())); rmax=max(rmax,float(R.max())); amin=min(amin,float(alpha.min())); amax=max(amax,float(alpha.max()))
        if np.any(alpha<0) or np.any(alpha>1): raise RuntimeError(f'one-step MOT invalid: alpha={alpha.min()}..{alpha.max()}')
        A=alpha[:,:,None,None]; A2=(alpha*alpha)[:,:,None,None]
        hp=Gp; hk=Gp*(DT*(A-.5*A2))+Gk*(1-A)-Gd/DT; an=Gp*(.5*DT*A2)+Gk*A+Gd/DT
        sl=slice(2*a,2*b); Hp[sl]=project(ot[a:b],hp,st); Hk[sl]=project(ot[a:b],hk,st); An[sl]=project(ot[a:b],an,st)
        del Gp,Gk,Gd,R,hp,hk,an,A,A2; gc.collect()
    return Hp,Hk,An,dict(min_distance_m=rmin,max_distance_m=rmax,max_delay_over_dt=(rmax/C0)/DT,alpha_min=amin,alpha_max=amax)

def smax_real(A,n=8):
    rng=np.random.default_rng(1234); x=rng.normal(size=A.shape[1]); x/=max(float(np.linalg.norm(x)),TINY)
    for _ in range(n):
        y=A@x; z=A.T@y; nz=float(np.linalg.norm(z))
        if nz<=TINY:return 1.
        x=z/nz
    return float(np.linalg.norm(A@x))

def smax_complex(A,n=8):
    rng=np.random.default_rng(1234); x=rng.normal(size=A.shape[1])+1j*rng.normal(size=A.shape[1]); x/=max(float(np.linalg.norm(x)),TINY)
    for _ in range(n):
        y=A@x; z=A.conj().T@y; nz=float(np.linalg.norm(z))
        if nz<=TINY:return 1.
        x=z/nz
    return float(np.linalg.norm(A@x))

def prepare_real(A,w):
    rs=np.repeat(np.sqrt(np.asarray(w,float)/max(float(np.mean(w)),TINY)),2); A*=rs[:,None]
    cs=np.linalg.norm(A,axis=0); cs=np.where(cs>TINY,cs,1.); A/=cs[None,:]
    sm=max(smax_real(A),TINY); alpha=LAMBDA_REL*sm; H=A.T@A; H.flat[::H.shape[0]+1]+=alpha*alpha
    return dict(rs=rs,cs=cs,As=A,cf=cho_factor(H,lower=True,overwrite_a=True,check_finite=False),smax=sm,ridge_alpha=alpha)

def inv_complex(prep,b):
    b=np.asarray(b,np.complex128)
    def one(x):
        bw=np.asarray(x,float)*prep['rs']; y=cho_solve(prep['cf'],prep['As'].T@bw,check_finite=False); return y/prep['cs']
    return one(b.real)+1j*one(b.imag)

def prepare_complex(A,w):
    rs=np.repeat(np.sqrt(np.asarray(w,float)/max(float(np.mean(w)),TINY)),2); Aw=A*rs[:,None]
    cs=np.linalg.norm(Aw,axis=0); cs=np.where(cs>TINY,cs,1.); As=Aw/cs[None,:]
    sm=max(smax_complex(As),TINY); alpha=LAMBDA_REL*sm; H=As.conj().T@As; H.flat[::H.shape[0]+1]+=alpha*alpha
    return dict(rs=rs,cs=cs,As=As,cf=cho_factor(H,lower=True,overwrite_a=True,check_finite=False),smax=sm,ridge_alpha=alpha)

def solve_complex(prep,b):
    bw=np.asarray(b,np.complex128)*prep['rs']; y=cho_solve(prep['cf'],prep['As'].conj().T@bw,check_finite=False); return y/prep['cs']

def incident_coeff(E,g):
    E=np.asarray(E,np.complex128); out=np.empty(2*len(E),np.complex128)
    out[0::2]=np.sum(E*g.solve_tangent1,axis=1); out[1::2]=np.sum(E*g.solve_tangent2,axis=1); return out

def coeff_to_current(g,q):
    q=np.asarray(q,np.complex128).reshape(-1,2); return q[:,0,None]*g.source_tangent1+q[:,1,None]*g.source_tangent2

def assemble_retarded_harmonic(g,omega,chunk):
    obs=np.asarray(g.solve_obs_points,float); src=np.asarray(g.ant_src_points,float); w=np.asarray(g.source_weights,float)
    ot=np.stack([g.solve_tangent1,g.solve_tangent2],axis=1); st=np.stack([g.source_tangent1,g.source_tangent2],axis=1)
    Aret=np.empty((2*len(obs),2*len(src)),np.complex128); diff2=0.; local2=0.; maxphase=0.
    for a in range(0,len(obs),chunk):
        b=min(len(obs),a+chunk); Gp,Gk,Gd,R=pair_blocks(obs[a:b],src,w)
        Tlocal=Gp/(1j*omega)+Gk+1j*omega*Gd; phase=np.exp(-1j*omega*R/C0)[:,:,None,None]; Tret=phase*Tlocal
        Pret=project(ot[a:b],Tret,st); Plocal=project(ot[a:b],Tlocal,st); Aret[2*a:2*b]=Pret
        diff2+=float(np.sum(np.abs(Pret-Plocal)**2)); local2+=float(np.sum(np.abs(Plocal)**2)); maxphase=max(maxphase,float(np.max(omega*R/C0)))
        del Gp,Gk,Gd,R,Tlocal,Tret,Pret,Plocal,phase; gc.collect()
    return Aret,dict(operator_retarded_minus_local_rel_fro=math.sqrt(diff2/max(local2,TINY)),max_pair_phase_delay_deg=maxphase*180/math.pi)

def solve_mot_z(Hp,Hk,prep,e,omega,rtol,maxiter):
    z=np.exp(1j*omega*DT); zi=1/z
    # nodal P^n = P^(n-1)+dt/2(K^(n-1)+K^n)
    P_over_Q=.5*DT*(1+zi)/(1-zi); pprev_over_Q=zi*P_over_Q
    def mv(q):
        # q = R[-e - Hp pprev - Hk qprev]
        Bq=(-pprev_over_Q)*(Hp@q)+(-zi)*(Hk@q)
        return q-inv_complex(prep,Bq)
    n=Hp.shape[1]; op=LinearOperator((n,n),matvec=mv,dtype=np.complex128); rhs=inv_complex(prep,-e)
    q,info=gmres(op,rhs,rtol=rtol,atol=0.,restart=min(80,n),maxiter=maxiter)
    return q,int(info),P_over_Q,pprev_over_Q

def make_holdouts(quick):
    R=1e-3; total=16e-3 if quick else 60e-3; gap=2e-3; Larm=.5*(total-gap); zb=-.5*total; zt=.5*gap
    ncyl,nphi,nhemi=(28,28,10) if quick else (96,72,24)
    T0,_,_,_,L0=mb1.make_arm(R,R,Larm,zb,ncyl,nphi,nhemi); T1,_,_,_,L1=mb1.make_arm(R,R,Larm,zt,ncyl,nphi,nhemi)
    out=[]
    for name,bary in (("holdout",[.19,.37,.44]),("spot",[.53,.29,.18])):
        H0,N0=mb1.bary_arm(L0,bary,R,Larm,zb); H1,N1=mb1.bary_arm(L1,bary,R,Larm,zt)
        P=np.concatenate((H0,H1)); N=np.concatenate((N0,N1)); aid=np.concatenate((np.zeros(len(H0),int),np.ones(len(H1),int)))
        out.append((name,P,N,mb1.classify(P,aid,R,Larm,zb,zt)))
    return out

def eval_retarded_field(points,g,q,omega,obs_chunk=256,src_chunk=64):
    K=coeff_to_current(g,q); P=np.asarray(points,float); SP=np.asarray(g.ant_src_points,float); SW=np.asarray(g.source_weights,float)
    E=np.zeros((len(P),3),np.complex128); ce=1/(4*np.pi*EPS0)
    for oa in range(0,len(P),obs_chunk):
        ob=min(len(P),oa+obs_chunk); X=P[oa:ob]; Eb=np.zeros((len(X),3),np.complex128)
        for sa in range(0,len(SP),src_chunk):
            sb=min(len(SP),sa+src_chunk); r=X[:,None,:]-SP[None,sa:sb,:]; R=np.maximum(np.linalg.norm(r,axis=2),1e-15)
            Ks=K[sa:sb]; rk=np.sum(r*Ks[None,:,:],axis=2); w=SW[None,sa:sb,None]; ph=np.exp(-1j*omega*R/C0)[:,:,None]
            gp=3*r*rk[:,:,None]/R[:,:,None]**5-Ks[None,:,:]/R[:,:,None]**3
            gk=(3*r*rk[:,:,None]/R[:,:,None]**4-Ks[None,:,:]/R[:,:,None]**2)/C0
            gd=(r*rk[:,:,None]/R[:,:,None]**3-Ks[None,:,:]/R[:,:,None])/(C0*C0)
            Eb+=np.sum(ce*w*ph*(gp/(1j*omega)+gk+1j*omega*gd),axis=1)
        E[oa:ob]=Eb
    return E

def tangential_vectors(E,N):
    E=np.asarray(E,np.complex128); N=np.asarray(N,float); return E-N*np.sum(E*N,axis=1)[:,None]

def trms(E,N,mask=None):
    Et=tangential_vectors(E,N); Et=Et if mask is None else Et[mask]
    return 0. if len(Et)==0 else float(np.sqrt(np.mean(np.sum(np.abs(Et)**2,axis=1))))

def metrics(total,inc,N,masks):
    ig=trms(inc,N); out={}
    for region,mask in masks.items():
        if not np.any(mask): continue
        rr=trms(total,N,mask); ii=trms(inc,N,mask); out[region]=dict(residual_rms=rr,incident_rms=ii,local_rel=rr/max(ii,TINY),global_norm=rr/max(ig,TINY),incident_ratio=ii/max(ig,TINY))
    return out

def gate_value(region,m):
    return ('local_rel',float(m['local_rel'])) if region=='global' or m['incident_ratio']>=WEAK else ('global_norm',float(m['global_norm']))

def worst_gate(ms):
    rows=[]
    for r in GATE_REGIONS:
        if r in ms:
            metric,v=gate_value(r,ms[r]); rows.append(dict(region=r,metric=metric,value=v,incident_ratio=float(ms[r]['incident_ratio'])))
    return max(rows,key=lambda x:x['value']) if rows else None

def compare_fields(A,B,N):
    a=tangential_vectors(A,N).reshape(-1); b=tangential_vectors(B,N).reshape(-1); na=float(np.linalg.norm(a)); nb=float(np.linalg.norm(b))
    if na<TINY or nb<TINY:return dict(amplitude_relative_error=float('inf'),phase_error_deg=float('inf'))
    return dict(amplitude_relative_error=abs(nb/na-1),phase_error_deg=abs(float(np.angle(np.vdot(a,b),deg=True))))

def run_freq(f,g,Hp,Hk,motprep,holds,quick,chunk,rtol,maxiter):
    om=2*np.pi*f; inc=mb1.inc_feed_localized(g.solve_obs_points,2e-3,1.).astype(np.complex128); e=incident_coeff(inc,g)
    t=time.perf_counter(); Aref,diag=assemble_retarded_harmonic(g,om,chunk); ref_ass=time.perf_counter()-t
    refprep=prepare_complex(Aref,g.solve_weights); qref=solve_complex(refprep,-e)
    qc,info,PQ,pPQ=solve_mot_z(Hp,Hk,motprep,e,om,rtol,maxiter)
    ref_coll=float(np.linalg.norm(Aref@qref+e)/max(float(np.linalg.norm(e)),TINY)); cand_coll=float(np.linalg.norm(Aref@qc+e)/max(float(np.linalg.norm(e)),TINY))
    rows=[]; comps=[]
    for name,HP,HN,masks in holds:
        iH=mb1.inc_feed_localized(HP,2e-3,1.).astype(np.complex128); Er=eval_retarded_field(HP,g,qref,om); Ec=eval_retarded_field(HP,g,qc,om)
        mr=metrics(iH+Er,iH,HN,masks); mc=metrics(iH+Ec,iH,HN,masks); wr=worst_gate(mr); wc=worst_gate(mc); cp=compare_fields(Er,Ec,HN); dif=trms(Ec-Er,HN)/max(trms(iH,HN),TINY)
        rows.append(dict(pattern=name,reference_worst_gate=wr,candidate_worst_gate=wc)); comps.append(dict(pattern=name,**cp,candidate_minus_reference_global_norm=float(dif)))
    worst_ref=max(r['reference_worst_gate']['value'] for r in rows); worst_c=max(r['candidate_worst_gate']['value'] for r in rows)
    wa=max(r['amplitude_relative_error'] for r in comps); wp=max(r['phase_error_deg'] for r in comps); wd=max(r['candidate_minus_reference_global_norm'] for r in comps)
    gates=dict(gmres_converged=info==0,reference_collocation_lt_5pct=ref_coll<BOUNDARY_GATE,candidate_collocation_lt_5pct=cand_coll<BOUNDARY_GATE,
        reference_holdout_lt_5pct=worst_ref<BOUNDARY_GATE,candidate_holdout_lt_5pct=worst_c<BOUNDARY_GATE,field_amplitude_error_lt_5pct=wa<AMP_GATE,
        field_phase_error_lt_5deg=wp<PHASE_GATE_DEG,field_difference_global_norm_lt_5pct=wd<FIELD_DIFF_GATE)
    if quick: passkeys=('gmres_converged','reference_collocation_lt_5pct','candidate_collocation_lt_5pct','field_amplitude_error_lt_5pct','field_phase_error_lt_5deg','field_difference_global_norm_lt_5pct')
    else: passkeys=tuple(gates)
    passed=all(gates[k] for k in passkeys)
    out=dict(frequency_Hz=float(f),gmres_info=info,reference=dict(collocation_rel_l2=ref_coll,worst_holdout_gate=worst_ref),mot_candidate=dict(continuous_physics_collocation_rel_l2=cand_coll,worst_holdout_gate=worst_c),
        holdouts=rows,field_comparison=comps,worst_field_amplitude_relative_error=wa,worst_field_phase_error_deg=wp,worst_candidate_minus_reference_global_norm=wd,
        retardation_diagnostic=diag,reference_operator_assembly_seconds=ref_ass,gates=gates,passed=bool(passed),quick_holdout_absolute_gates_are_descriptive=bool(quick))
    del Aref,refprep,qref,qc; gc.collect(); return out

def self_test():
    s=settings(True); s=dict(s); s.update(sources_per_arm=16,n_phi=16,n_cyl=16,n_hemi=6,obs_chunk=8); g=geometry(s)
    Hp,Hk,An,d=assemble_mot(g,s['obs_chunk']); prep=prepare_real(An,g.solve_weights); f=3e6; om=2*np.pi*f
    inc=mb1.inc_feed_localized(g.solve_obs_points,2e-3,1.).astype(np.complex128); e=incident_coeff(inc,g)
    q,info,PQ,pPQ=solve_mot_z(Hp,Hk,prep,e,om,1e-11,500); z=np.exp(1j*om*DT)
    # Put the phasor back into the actual real-valued recurrence at several phases.
    errs=[]
    for n in range(16):
        qn=np.real(q*z**n); qp=np.real(q*z**(n-1)); pn=np.real((PQ*q)*z**n); pp=np.real((PQ*q)*z**(n-1)); en=np.real(e*z**n)
        qcalc=inv_complex(prep,-(en+Hp@pp+Hk@qp)).real
        errs.append(float(np.linalg.norm(qcalc-qn)/max(float(np.linalg.norm(q)),TINY)))
    report=dict(self_test_pass=bool(info==0 and max(errs)<1e-8 and d['max_delay_over_dt']<1),gmres_info=info,max_real_step_reconstruction_relative_error=max(errs),max_delay_over_dt=d['max_delay_over_dt'])
    print(json.dumps(report,indent=2));
    if not report['self_test_pass']: raise RuntimeError('Step 4C2 V4 self-test failed')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--quick',action='store_true'); p.add_argument('--self-test',action='store_true'); p.add_argument('--include-15mhz-stress',action='store_true')
    p.add_argument('--gmres-rtol',type=float,default=1e-9); p.add_argument('--gmres-maxiter',type=int,default=300); p.add_argument('--out',default='step4c2_v4_retarded_frequency_regression'); a=p.parse_args()
    if a.self_test: self_test(); return
    s=settings(a.quick); out=Path('article_cases_output')/a.out; out.mkdir(parents=True,exist_ok=True); freqs=list(FORMAL_FREQS)+([STRESS_FREQ] if a.include_15mhz_stress else [])
    print('Step 4C2 V4 causal retarded-MOT frequency regression'); print('  mode:',s['mode']); print('  sources/arm:',s['sources_per_arm']); print('  frequencies MHz:',[f/1e6 for f in freqs],flush=True)
    g=geometry(s); t=time.perf_counter(); Hp,Hk,An,delay=assemble_mot(g,s['obs_chunk']); mot_ass=time.perf_counter()-t
    if delay['max_delay_over_dt']>=1: raise RuntimeError('one-step MOT invalid')
    motprep=prepare_real(An,g.solve_weights); del An; gc.collect(); holds=make_holdouts(a.quick)
    results=[]
    for f in freqs:
        print(f'  solving {f/1e6:.1f} MHz ...',flush=True); r=run_freq(f,g,Hp,Hk,motprep,holds,a.quick,s['obs_chunk'],a.gmres_rtol,a.gmres_maxiter); results.append(r)
        print(f"    ref_coll={r['reference']['collocation_rel_l2']:.6g} mot_coll={r['mot_candidate']['continuous_physics_collocation_rel_l2']:.6g} amp={r['worst_field_amplitude_relative_error']:.6g} phase={r['worst_field_phase_error_deg']:.6g}deg diff={r['worst_candidate_minus_reference_global_norm']:.6g} pass={r['passed']}",flush=True)
    gating=[r for r in results if r['frequency_Hz'] in FORMAL_FREQS]
    if len(gating) != 3:
        raise RuntimeError(f'Expected 3 gating frequencies, got {len(gating)}')
    finite_keys = (
        'worst_field_amplitude_relative_error',
        'worst_field_phase_error_deg',
        'worst_candidate_minus_reference_global_norm',
    )
    for rr in gating:
        for kk in finite_keys:
            if not np.isfinite(rr[kk]):
                raise RuntimeError(f'Non-finite Step 4C2 V4 metric: {kk}={rr[kk]}')
        if not np.isfinite(rr['reference']['collocation_rel_l2']):
            raise RuntimeError('Non-finite reference collocation residual')
        if not np.isfinite(rr['mot_candidate']['continuous_physics_collocation_rel_l2']):
            raise RuntimeError('Non-finite MOT collocation residual')
    accepted=all(r['passed'] for r in gating)
    rep=dict(step='4C2-V4',purpose='causal retarded-MOT physical frequency-response regression',mode=s['mode'],previous_C2_V4_stub_status='invalid_no_physical_calculation_performed',production_settings=dict(total_length_m=s['total_length_m'],sources_per_arm=s['sources_per_arm'],collocation_points_per_arm=2*s['sources_per_arm'],depth_frac=DEPTH_FRAC,lambda_rel=LAMBDA_REL,operator_softening_distance_m=0.,dt_s=DT),one_step_delay=delay,mot_operator_assembly_seconds=mot_ass,
        frequencies_Hz=freqs,gates=dict(boundary=BOUNDARY_GATE,amplitude_relative_error=AMP_GATE,phase_error_deg=PHASE_GATE_DEG,candidate_reference_global_difference=FIELD_DIFF_GATE,weak_region_threshold=WEAK),results=results,accepted=accepted,
        decision='STEP4C2_V4_PASS_proceed_to_STEP4C3_V2_retarded_MOT_broadband_replay' if accepted else 'STEP4C2_V4_FAIL_do_not_replay_PIC_forcing',scope_note='Vacuum frequency-response validation only; no PIC or nonlinear feedback.')
    (out/'step4c2_v4_report.json').write_text(json.dumps(rep,indent=2),encoding='utf-8')
    lines=['Step 4C2 V4 causal retarded-MOT frequency regression',f'accepted: {accepted}',f"mode: {s['mode']}",f"max delay/dt: {delay['max_delay_over_dt']:.9g}"]
    for r in results: lines.append(f"{r['frequency_Hz']/1e6:.1f} MHz: ref_coll={r['reference']['collocation_rel_l2']:.6g}, mot_coll={r['mot_candidate']['continuous_physics_collocation_rel_l2']:.6g}, amp={r['worst_field_amplitude_relative_error']:.6g}, phase={r['worst_field_phase_error_deg']:.6g} deg, diff={r['worst_candidate_minus_reference_global_norm']:.6g}, pass={r['passed']}")
    lines.append('decision: '+rep['decision']); (out/'step4c2_v4_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines))

if __name__=='__main__': main()
