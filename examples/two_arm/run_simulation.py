'Driven two-arm example: run simulation. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
import argparse,gc,json,math,os,time
from pathlib import Path
import numpy as np
import two_arm_model as ab
from two_arm_model import c3,record,old

def run(out,smoke=False,vacuum_only=False,calibration=None,native_gate_report=None):
    if not vacuum_only:
        if native_gate_report is None:raise RuntimeError('Passing native electric+magnetic gate report is required before PIC setup')
        from validate_feedback_gpu import fingerprints
        gate=json.loads(Path(native_gate_report).read_text())
        if gate.get('status')!='NATIVE_EM_FEEDBACK_PASSED' or not gate.get('native_CUDA') or gate.get('source_sha256')!=fingerprints():
            raise RuntimeError('Native gate absent, failed, or bound to different source files')
        if gate.get('visible_devices')!=os.environ.get('CUDA_VISIBLE_DEVICES'):raise RuntimeError('Native gate and full run use different visible GPU selections')
    from EMsolver.revision_profile import StageProfile,environment
    s=ab.settings(smoke);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();g=ab.frozen_geometry()
    print('Building mixed PEC matrices and fixed ridge factor ...',flush=True)
    t0=time.perf_counter();beta=None if calibration is None else json.loads(Path(calibration).read_text())['beta']
    m=ab.model(g,magnetic_weight=beta);setup_s=time.perf_counter()-t0
    if not vacuum_only:
        from validate_operator_cpu import run_check
        run_check(m,out/'installed_operator_CPU.json')
    h=ab.frozen_holdout();hp=np.asarray(h['points']);hn=np.asarray(h['normals']);regions=list(h['region_names'])
    hm=np.asarray(h['mask_matrix'],bool);analysis=s['analysis_start_cycles']/c3.F_DRIVE
    if not vacuum_only:
        import cupy as cp
        import _case_common as common
        cfg=record.make_pic_config(str(out),s);cfg.seed=s['seed'];cfg.quasi_electrostatic=False;cfg.jefipic_kernel_mode='atomic'
        prior=common.build_mas_device
        common.build_mas_device=lambda *args,**kwargs:record.lightweight_probe_holder(g)
        try:bundle=common.build_pic_solver_mas(cfg,out/'UNUSED.stl')
        finally:common.build_mas_device=prior
        pic=bundle['PIC'];sol=bundle['solver'];bridge=bundle['unit_bridge'];rng=bundle['rng']
        from reflecting_sources import install
        from incident_fields import sample_solver
        install(pic)
        initial_fingerprint=c3.array_sha256([bundle['initial_positions_si'][0],bundle['initial_positions_si'][1]])
        common.save_config(out,cfg)
        (out/'physical_parameters.json').write_text(json.dumps(common.physical_summary(cfg),indent=2),encoding='utf-8')
        if calibration and initial_fingerprint!=json.loads(Path(calibration).read_text())['initial_position_fingerprint']:
            raise RuntimeError('Reference/full-run initial particle positions differ; no particle push started')
        if sol.kernel_mode!='atomic':raise RuntimeError('Expected native full-EM atomic path')
        sol.set_antenna_temporal_basis('hermite_P')
        from magnetic_feedback import MagneticFeedback,add_feedback_fields
        magnetic_feedback=MagneticFeedback(sol,m.ga.ant_src_points,m.ga.source_weights,bridge)
        nx,ny,nz=bundle['shape'];dx,dy,dz=bundle['spacing'];Ex,Ey,Ez,Bx,By,Bz=bundle['fields']
        obs=cp.asarray(bridge.length_si_to_code(g.solve_obs_points));hobs=cp.asarray(bridge.length_si_to_code(hp))
        fgrid=cp.asarray(old.build_feed_grid_template_code(bundle));steps=int(cfg.steps)
        prof=StageProfile(enabled=os.environ.get('JEFIPIC_PROFILE','0')=='1',sync=lambda:cp.cuda.Device().synchronize(),warmup_steps=10)
    else:
        cp=None;steps=record.steps_from_cycles(s['cycles']);prof=StageProfile(False)
    p=np.zeros(len(m.cs));k=p.copy();times=[];moments=[];polmom=[];norms=[];closures=[];ec=[];br=[];plasma_rms=[];pic_cont=[]
    er={name:0. for name in regions};ei=er.copy();counts={name:0 for name in regions};bsum=bden=0.
    coll_e=coll_i=0.;coll_b=coll_bden=0.;surface_events=[0,0];diagnostics=[]
    sample_idx=np.linspace(0,g.num_collocation_points-1,96,dtype=int);sampleE=[];sampleB=[]
    surface_records=[];magnetic_moments=[];valid_cc_count=0;
    windows=[dict(cycles=[lo,hi],er={x:0. for x in regions},ei={x:0. for x in regions},counts={x:0 for x in regions},br=0.,bi=0.) for lo,hi in ((.75,4.),(4.,8.),(8.,12.01))]
    harmonic=ab.HarmonicAccumulator(analysis);completed=0;runaway=False;qualification_stop=None;max_cc=0.
    (out/'running.json').write_text(json.dumps(dict(status='running',settings=s,backend=ab.VERSION)),encoding='utf-8')
    for step in range(steps):
        t=step*c3.DT;prof.begin_step(step)
        if not vacuum_only:
            with prof.stage('particle_push_and_surface_events'):
                if step>0:
                    pic.proceed_one_step(Ex,Ey,Ez,Bx,By,Bz,dx,dy,dz,nx,ny,nz,cfg.field_order)
                    hits=common.absorb_particles_on_antenna(pic,cfg,rng)
                    surface_events[0]+=int(hits.get(0,0));surface_events[1]+=int(hits.get(1,0))
            with prof.stage('deposition_including_transfers_and_continuity'):
                rho,jx,jy,jz=pic.evaluate_rho_J(dx,dy,dz,nx,ny,nz,cfg.particle_order)
            with prof.stage('plasma_field_pass1_and_history_upload'):
                sol.Jefimenko_solver(rho.flatten(),include_antenna=False,face_currents=(jx,jy,jz))
            with prof.stage('surface_EB_sample_download_and_units'):
                direct_E,direct_B=sample_solver(sol,obs)
                pe=np.asarray(bridge.efield_code_to_si(direct_E)).reshape(-1,3)
                pb=np.asarray(bridge.bfield_code_to_si(direct_B)).reshape(-1,3)
        else:pe=np.zeros((g.num_collocation_points,3));pb=pe.copy()
        pr=float(np.sqrt(np.mean(np.sum(pe*pe,axis=1))))
        if not np.isfinite(pe).all() or not np.isfinite(pb).all() or pr>1e12:
            runaway=True;break
        with prof.stage('cpu_mot_rhs_solve_and_state_update'):
            ie=pe+c3.feed_at_time(g.solve_obs_points,t);ib=pb
            rhs=m.rhs(ie,ib);history=m.Hp@p+m.Hk@k
            q=m.solve(-rhs-history);pn=p+c3.DT*((1-ab.THETA)*k+ab.THETA*q)
        if not np.isfinite(q).all() or not np.isfinite(pn).all():runaway=True;break
        with prof.stage('collocation_and_state_diagnostics'):
            residual=rhs+history+ab.now_coeff(m,q);nr=2*g.num_collocation_points
            closure=float(np.linalg.norm((pn-p)/c3.DT-((1-ab.THETA)*k+ab.THETA*q))/max(np.linalg.norm(q),1e-300))
            e_res,e_inc,bn,bnorm=ab.collocation_metrics(m,ie,ib,residual)
            ec.append(float(np.linalg.norm(e_res)/max(np.linalg.norm(e_inc),1e-300)))
            br.append(float(np.linalg.norm(bn)/max(bnorm,1e-300)))
            if t>=analysis:
                coll_e+=float(np.linalg.norm(e_res)**2);coll_i+=float(np.linalg.norm(e_inc)**2)
                coll_b+=float(np.linalg.norm(bn)**2);coll_bden+=float(bnorm**2)
        with prof.stage('independent_surface_EB_diagnostics'):
            if step%int(s['diag_stride'])==0 and t>=analysis:
                if vacuum_only:he=np.zeros_like(hp);hb=np.zeros_like(hp)
                else:
                    direct_E,direct_B=sample_solver(sol,hobs)
                    he=np.asarray(bridge.efield_code_to_si(direct_E)).reshape(-1,3)
                    hb=np.asarray(bridge.bfield_code_to_si(direct_B)).reshape(-1,3)
                he+=c3.feed_at_time(hp,t)
                se,sb=ab.field(hp,g,p,k,q)
                surface_records.append((t,he.copy(),hb.copy(),se.copy(),sb.copy(),p.copy(),k.copy(),q.copy()))
                rsq=c3.tangential_sq(he+se,hn);isq=c3.tangential_sq(he,hn)
                for ir,name in enumerate(regions):
                    mask=hm[ir];er[name]+=float(np.sum(rsq[mask]));ei[name]+=float(np.sum(isq[mask]));counts[name]+=int(mask.sum())
                rb=np.sum((hb+sb)*hn,axis=1);bsum+=float(np.linalg.norm(rb)**2);bden+=float(np.linalg.norm(hb)**2)
                diagnostics.append([t,float(np.sqrt(np.sum(rsq))),float(np.linalg.norm(rb)),float(np.linalg.norm(hb))])
                for window in windows:
                    if window['cycles'][0]<=t*c3.F_DRIVE<window['cycles'][1]:
                        for ir,name in enumerate(regions):
                            mask=hm[ir];window['er'][name]+=float(np.sum(rsq[mask]));window['ei'][name]+=float(np.sum(isq[mask]));window['counts'][name]+=int(mask.sum())
                        window['br']+=float(np.linalg.norm(rb)**2);window['bi']+=float(np.linalg.norm(hb)**2)
                harmonic.add(t,collocation_incident_E=ie,collocation_incident_B=ib,holdout_incident_E=he,holdout_incident_B=hb)
        with prof.stage('source_moments_and_units'):
            M,D,K,P=old.source_moments(g,q[:m.ne],pn[:m.ne])
            MP=pn[m.ne:].reshape(-1,3);MK=q[m.ne:].reshape(-1,3)
            magnetic_moments.append(c3.C0*np.sum(m.ga.source_weights[:,None]*MP,axis=0))
            if not vacuum_only:Kcode=bridge.surface_current_density_si_to_code(K);Pcode=bridge.surface_polarization_density_si_to_code(P)
        if not vacuum_only:
            with prof.stage('source_upload_and_total_field_pass2'):
                sol.recompute_current_fields(antenna_surface_currents=Kcode,antenna_polarization=Pcode,include_antenna=True)
                magnetic_EB=magnetic_feedback.compute(sol,MP,MK)
                Ex0,Ey0,Ez0,Bx,By,Bz=add_feedback_fields(sol.acquire_EB_field(),magnetic_EB)
            with prof.stage('feed_grid_addition'):
                ramp,_=c3.smoothstep_ramp_and_derivative(t);scale=ramp*math.cos(2*np.pi*c3.F_DRIVE*t)
                Ex=Ex0+scale*fgrid[:,0];Ey=Ey0+scale*fgrid[:,1];Ez=Ez0+scale*fgrid[:,2]
                # Next push uses Numba CUDA: make CuPy additions ready even when
                # intrusive stage profiling is disabled or PTDS is enabled.
                cp.cuda.get_current_stream().synchronize()
            with prof.stage('PIC_continuity_diagnostics'):
                cd=pic.get_continuity_diagnostics();pic_cont.append(cd)
                if cd.get('valid'):
                    valid_cc_count+=1
                    max_cc=max(max_cc,float(cd['face_current']['linf_rel']))
                    if max_cc>1e-9:qualification_stop='PIC source continuity exceeds 1e-9'
        times.append(t);moments.append(M);polmom.append(D);closures.append(closure);norms.append(ab.state_norm(m,q,pn));plasma_rms.append(pr)
        sampleE.append(pe[sample_idx].copy());sampleB.append(pb[sample_idx].copy())
        p,k=pn,q;completed+=1;prof.end_step()
        if step%100==0 or step==steps-1:
            print(f'{"vacuum" if vacuum_only else "coupled"}: {completed}/{steps}; Et={ec[-1]:.5g}; plasma E={pr:.5g} V/m; state={norms[-1]:.5g}',flush=True)
        if not smoke and step>=333 and step%20==0 and hold_count_ready(counts):
            gate_rows,gate_worst=old.aggregate_holdout_rows(er,ei,counts,regions)
            if gate_worst and gate_worst['gate_value']>.2:qualification_stop='Independent Et gross-error guard exceeds 20% after one cycle; final acceptance remains 5%'
            if bden>0 and np.sqrt(bsum/bden)>.2:qualification_stop='Independent Bn/full incoming B gross-error guard exceeds 20%; final acceptance remains 5%'
        if qualification_stop:
            print('QUALIFICATION STOP:',qualification_stop,flush=True);break
        if step%500==0:
            np.savez_compressed(out/'partial_checkpoint.npz',completed=completed,p=p,k=k,time_s=times,state_norm=norms)
    full_loop_end=time.perf_counter()
    if prof.enabled:
        prof.write(out,dict(environment=environment(cp),settings=s,backend=ab.VERSION,theta=ab.THETA,steps_completed=completed,steps_requested=steps,
            n_electric_sources=g.num_sources,n_magnetic_sources=len(m.ga.ant_src_points),total_dofs=len(m.cs),magnetic_test_rank=m.magnetic_test_rank,n_collocation=g.num_collocation_points,n_holdout=len(hp),plasma_electric_history_length=sol.len_time_snapshots,magnetic_history_length=magnetic_feedback.history_nodes,
            grid_kernel_mode=sol.kernel_mode,pool_used_bytes_snapshot=cp.get_default_memory_pool().used_bytes(),pool_reserved_bytes_snapshot=cp.get_default_memory_pool().total_bytes(),
            memory_scope='CuPy pool snapshots, not peak RAM/VRAM; synchronized diagnostic profile, not uninstrumented wall benchmark'))
    tail=[ab.state_norm(m,k,p)]
    for _ in range(500):
        q=m.solve(-m.Hp@p-m.Hk@k);p=p+c3.DT*((1-ab.THETA)*k+ab.THETA*q);k=q
        tail.append(ab.state_norm(m,k,p))
        if not np.isfinite(tail[-1]):runaway=True;break
    growth=float(np.median(np.array(tail[-50:])/np.maximum(np.array(tail[-51:-1]),1e-300))) if len(tail)>50 else float('inf')
    rows,worst=old.aggregate_holdout_rows(er,ei,counts,regions)
    response=old.fit_response(times,moments,polmom,analysis) if completed>=20 else None
    harmonic_report=harmonic.save(out/'incident_harmonics.npz')
    closuremax=max(closures,default=float('inf'))
    window_reports=[]
    for window in windows:
        wr,ww=old.aggregate_holdout_rows(window['er'],window['ei'],window['counts'],regions)
        bb=float(np.sqrt(window['br']/window['bi'])) if window['bi']>0 else None
        passed=ww is not None and ww['gate_value']<=.05 and (vacuum_only or (bb is not None and bb<=.05))
        window_reports.append(dict(cycles=window['cycles'],Et_regions=wr,worst=ww,Bn_over_full_B=bb,passed=bool(passed)))
    window_valid=smoke or all(x['passed'] for x in window_reports)
    execution_valid=completed==steps and not runaway and closuremax<1e-10 and growth<=1.0001
    precision_valid=(vacuum_only or valid_cc_count>=max(1,completed-1)) and max_cc<=1e-9 and (smoke or (worst is not None and worst['gate_value']<=.05 and (vacuum_only or (bden>0 and np.sqrt(bsum/bden)<=.05)) and window_valid))
    valid=execution_valid and precision_valid and qualification_stop is None
    report=dict(status=('smoke_only' if smoke else 'qualified') if valid else 'not_qualified',backend=ab.VERSION,theta=ab.THETA,settings=s,
        magnetic_weight=m.magnetic_weight,magnetic_weight_source='frozen area-weighted Et/full-B input energy from separate 64-node source calibration, no fit to boundary residuals' if calibration else 'same frozen article Et/full-B reference; smoke/control only',
        geometry_fingerprint=c3.geometry_fingerprint(g),steps_requested=steps,steps_completed=completed,full_two_way=not vacuum_only,
        deposition='native unsplit first-order charge-conserving face currents',drive='original curl-free localized E_z(z,t); external B=0; same feed in PIC and PEC RHS',
        all_states_finite=not runaway,coefficient_closure_max_relative=closuremax,homogeneous_tail_growth=growth,two_way_execution_valid=execution_valid,precision_qualified=precision_valid,qualification_stop=qualification_stop,source_continuity_max_relative=max_cc,source_continuity_valid_steps=valid_cc_count,n_electric_sources=g.num_sources,n_magnetic_sources=len(m.ga.ant_src_points),total_dofs=len(m.cs),magnetic_test_rank=m.magnetic_test_rank,n_collocation=g.num_collocation_points,independent_windows=window_reports,late_window_precision_qualified=bool(window_valid),native_gate_report=None if vacuum_only else str(native_gate_report),
        initial_position_fingerprint=None if vacuum_only else initial_fingerprint,incident_interface='direct retarded rho/face-J at actual PEC points, no E/B grid interpolation',
        source_boundary='reflected CIC shape at the unchanged physical reflective box',
        mixed_operator_setup_s=setup_s,elapsed_until_main_loop_end_s=full_loop_end-started,elapsed_scope='includes setup; synchronized profile if enabled; homogeneous tail excluded; not repeated wall throughput',
        boundary_Et_collocation=float(np.sqrt(coll_e/max(coll_i,1e-300))),boundary_Et_holdout=rows,worst_Et_holdout=worst,
        projected_Bn_collocation_over_area_weighted_full_B=float(np.sqrt(coll_b/max(coll_bden,1e-300))) if coll_bden else None,
        boundary_Bn_holdout_over_incident_full_B=float(np.sqrt(bsum/max(bden,1e-300))) if bden else None,
        magnetic_metric_note='Only holdout Bn is the full, unprojected physical residual divided by full incident B. Collocation Bn is a projected area-weighted internal diagnostic and is not an acceptance substitute. Vacuum B=0 has no relative B metric.',
        surface_absorption_counts=surface_events,response=response,response_scope='Electric auxiliary moments only, not port impedance or plasma/vacuum ratio. Magnetic auxiliary moment is saved separately; neither is asserted to be the total physical conductor moment.',
        harmonic_fit=harmonic_report,smoke_only=smoke)
    np.savez_compressed(out/'diagnostics.npz',time_s=times,current_moment_A_m=moments,polarization_moment_C_m=polmom,state_norm=norms,closure_relative=closures,
        collocation_Et_relative=ec,collocation_projected_Bn_over_area_weighted_full_B=br,magnetic_aux_moment_A_m2=magnetic_moments,plasma_surface_E_rms=plasma_rms,tail_norm=tail,
        sample_indices=sample_idx,sample_points_m=g.solve_obs_points[sample_idx],sample_plasma_E_V_m=sampleE,sample_plasma_B_T=sampleB,holdout_diagnostics=diagnostics)
    if surface_records:
        np.savez_compressed(out/'independent_surface_fields.npz',time_s=[x[0] for x in surface_records],points_m=hp,normals=hn,region_names=regions,mask_matrix=hm,incident_E=[x[1] for x in surface_records],incident_B=[x[2] for x in surface_records],scattered_E=[x[3] for x in surface_records],scattered_B=[x[4] for x in surface_records],P_prev_coeff=[x[5] for x in surface_records],K_prev_coeff=[x[6] for x in surface_records],K_now_coeff=[x[7] for x in surface_records])
    (out/'PIC_continuity.json').write_text(json.dumps(pic_cont,indent=2),encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'running.json').write_text(json.dumps(dict(status='finished',qualification=report['status'],steps_completed=completed)),encoding='utf-8')
    if valid:(out/'partial_checkpoint.npz').unlink(missing_ok=True)
    print(json.dumps(report,indent=2),flush=True)
    if not valid:raise RuntimeError('Full-run completion/stability/source-continuity/independent-E/B gate failed; diagnostic outputs retained')

def hold_count_ready(counts):return counts.get('global',0)>0

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--vacuum-only',action='store_true');ap.add_argument('--calibration');ap.add_argument('--native-gate-report')
    a=ap.parse_args();run(a.out,a.smoke,a.vacuum_only,a.calibration,a.native_gate_report)
