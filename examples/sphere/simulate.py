'PEC sphere and particle-cloud example: simulate. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json, math, os, sys, time
import numpy as np
import boundary as boundary

def fibonacci(n, phase=0.):
    k = np.arange(n)+.5
    z = 1-2*k/n
    phi = 2*np.pi*k/((1+np.sqrt(5))/2)+phase
    return np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z))

def tangents(normals):
    ref = np.tile([0.,0.,1.],(len(normals),1))
    ref[np.abs(normals[:,2])>.9] = [1.,0.,0.]
    a = np.cross(normals,ref); a /= np.linalg.norm(a,axis=1)[:,None]
    return a, np.cross(normals,a)

def geometry(R, count):
    return boundary.geometry(R, count)


def body_force(t,cfg):
    T=cfg['pulse_duration_s']
    return cfg['force_equivalent_E_V_m']*math.sin(math.pi*t/T)**2 if 0<t<T else 0.

def segment_distance(a,b):
    v=b-a; den=np.sum(v*v,axis=1)
    lam=np.clip(-np.sum(a*v,axis=1)/np.maximum(den,1.e-300),0.,1.)
    return np.linalg.norm(a+lam[:,None]*v,axis=1)

def run(args):
    # The shared article helper defaults the visibility mask to card 0. Set it
    # before importing CUDA modules so that --gpu also works on an unmasked host.
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
        logical_gpu=0
    else:
        logical_gpu=args.gpu
    repo=Path(args.repo).resolve()
    sys.path[:0]=[str(repo),str(repo/'support')]
    import cupy as cp
    from Particle_transport.main import pic_transport
    from EMsolver.solver import EMsolver
    from _case_common import JefiPICUnitBridge
    mot = boundary
    cp.cuda.Device(logical_gpu).use()
    cfg=json.loads((Path(args.case)/'case.json').read_text())
    particle_path=Path(args.case)/'particles.npz'
    if hashlib.sha256(particle_path.read_bytes()).hexdigest()!=cfg['particle_file_sha256']:
        raise RuntimeError('Shared particle file hash mismatch')
    xyz=np.load(particle_path)['positions_m']
    # Use the existing explicit SI mode. The legacy FU conversion table uses
    # rounded constants; a native cross-code comparison should share exact SI
    # constants rather than import those small conversion inconsistencies.
    bridge=JefiPICUnitBridge(mode='SI')
    R=cfg['radius_m']; half=cfg['base_box_halfwidth_m']; n=args.grid
    dt=args.dt_r_over_c*R/cfg['c_m_s']
    total=int(math.ceil(cfg['stop_time_s']/dt))
    if args.max_steps is not None: total=min(total,args.max_steps)
    # Common comparison ends at cfg.stop_time_s; retain final bracketing sample.
    if n%2: raise ValueError('Use an even particle-grid count (independent probes must avoid source centers)')
    # Only this compact cube contains particles/sources. Its boundary is a
    # computational guard, not an additional wall; any crossing invalidates
    # the run. Vacuum outside it is retained through the free-space kernel.
    origin_si=R*np.array([-1.5,-1.5,1.0])
    extent_si=3*R
    ds_si=extent_si/n; ds=float(bridge.length_si_to_code(ds_si))
    origin=np.asarray(bridge.length_si_to_code(origin_si))
    dt_code=float(bridge.time_si_to_code(dt))
    pos={s:bridge.length_si_to_code(xyz.copy()) for s in [0,1]}
    vel={s:np.zeros_like(xyz) for s in [0,1]}
    W=cfg['macro_weight']; Q=cfg['charge_C']
    charges=[float(bridge.charge_si_to_code(sign*Q*W)) for sign in [-1,1]]
    masses=[float(bridge.mass_si_to_code(cfg[k]*W)) for k in ['electron_mass_kg','ion_mass_kg']]
    pic=pic_transport(pos,vel,{s:vel[s].copy() for s in [0,1]},dt_code,charges,[W,W],masses,
                      *origin,n,n,n,ds,ds,ds,0.,0.,bridge.c_code,False,False)
    pic.set_boundary_mode('open')
    hist=max(16,int(math.ceil(2*half*math.sqrt(3)/cfg['c_m_s']/dt))+4)
    def make_solver(obs_n,obs_spacing,obs_origin,conductor=False):
        obj=EMsolver(hist,*obs_n,n,n,n,*obs_spacing,*obs_origin,
            ds,ds,ds,*origin,dt_code,bridge.epsilon0_code,bridge.c_code,0)
        obj.set_grid_current_layout('staggered_face'); obj.set_kernel_mode('atomic')
        if conductor:
            obj.set_antenna_temporal_basis('hermite_P')
            obj.set_antenna_sources(bridge.length_si_to_code(g.ant_src_points),
                source_weights=bridge.area_si_to_code(g.source_weights),softening_distance=0.)
        return obj
    g=geometry(R,args.sources)
    solver=make_solver((n,n,n),(ds,ds,ds),origin,args.metal)
    surface_n=24
    surface_spacing=float(bridge.length_si_to_code(2.4*R/surface_n))
    surface_origin=bridge.length_si_to_code(np.full(3,-1.2*R))
    surface_solver=make_solver((surface_n,)*3,(surface_spacing,)*3,surface_origin) if args.metal else None
    points=np.array(cfg['probes_m'])
    # The existing region constructor also gives exact independent point
    # observations: a 1x1x1 observation cell centered at the requested point.
    probe_solvers=[make_solver((1,1,1),(ds,)*3,bridge.length_si_to_code(point)-.5*ds,args.metal) for point in points]
    if args.metal:
        hp,hk,prep,delay=boundary.assemble(g,dt)
    else:
        delay={}
    pprev=np.zeros(2*g.num_sources); kprev=pprev.copy()
    fields=tuple(cp.zeros(n**3,dtype=cp.float64) for _ in range(6))
    hold_norm=fibonacci(512,.817); hold_points=R*hold_norm
    times=[]; probes=[]; bprobes=[]; grid_probe_E=[]; momenta=[]; kinetic=[]; centroids=[]; closures=[]
    coll_num=coll_den=hold_num=hold_den=0.
    coll_B_num=coll_B_den=hold_B_num=hold_B_den=0.
    p_hist=[]; k_hist=[]; incident_E_hist=[]; incident_B_hist=[]
    continuity_num=continuity_den=0.
    min_radius=float('inf'); max_coord=0.; max_continuity=0.; min_source_guard=float('inf')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); completed=False
    def record_report(error=None):
        report=dict(status='complete' if completed else 'incomplete', error=error,
            metal=args.metal, grid=n, sources=args.sources, dt_s=dt, history_nodes=hist,
            particle_grid_origin_m=origin_si.tolist(),particle_grid_extent_m=extent_si,
            particle_grid_spacing_m=ds_si,surface_observation_grid=surface_n,
            min_source_region_guard_m=min_source_guard,
            independent_probe_evaluation='Native original solver with 1x1x1 observation grids; no interpolation to probes',
            steps_requested=total, samples_completed=len(times), last_time_s=times[-1] if times else None,
            complete_to_common_end=bool(times and times[-1]>=cfg['stop_time_s']*(1-1e-12)),
            min_particle_radius_m=min_radius,max_abs_particle_coordinate_m=max_coord,
            no_particle_collection_or_outer_crossing=(min_radius>R and max_coord<half and min_source_guard>0),
            continuity_max_absolute_code_units=max_continuity, delay=delay,
            continuity_time_aggregated_rms_relative=math.sqrt(continuity_num/max(continuity_den,1.e-300)),
            full_electromagnetic=True,spatial_truncation=False,
            boundary_revision=boundary.VERSION, boundary_theta=boundary.THETA, boundary_ridge_rel=boundary.RIDGE_REL,
            auxiliary_geometry='Gauss 12x16 sources, 24x32 collocation; full 3D coefficients',
            antenna_time_basis='hermite_P with theta=0.6 endpoint integration', plasma_time_basis_unchanged=True,
            boundary_constraints='Et and fixed normalized beta*c*Bn; zero initial interior B',
            magnetic_weight=boundary.MAGNETIC_WEIGHT, normalization_reference=boundary.CALIBRATION_PROVENANCE,
            n_collocation=g.num_collocation_points,
            collocation_Bn_over_incident_B=math.sqrt(coll_B_num/max(coll_B_den,1.e-300)),
            holdout_Bn_over_incident_B=math.sqrt(hold_B_num/max(hold_B_den,1.e-300)),
            boundary_module_sha256=hashlib.sha256(Path(boundary.__file__).read_bytes()).hexdigest(),
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            collocation_time_L2_rel=math.sqrt(coll_num/max(coll_den,1e-300)),
            holdout_time_L2_rel=math.sqrt(hold_num/max(hold_den,1e-300)),
            wall_s=time.perf_counter()-started, units=bridge.summary(),
            particle_file_sha256=cfg['particle_file_sha256'], cuda_device=str(cp.cuda.runtime.getDeviceProperties(logical_gpu)['name']),
            case_file_sha256=hashlib.sha256((Path(args.case)/'case.json').read_bytes()).hexdigest(),
            requested_gpu=args.gpu,logical_gpu=logical_gpu,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
            source_hashes={p.relative_to(repo).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [repo/'EMsolver/solver.py',repo/'EMsolver/cuda_functions.py',repo/'Particle_transport/main.py',repo/'Particle_transport/cuda_kernels.py',repo/'support/retarded_history.py']})
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        np.savez_compressed(out/'diagnostics.npz',time_s=np.array(times),E_V_m=np.array(probes),B_T=np.array(bprobes),
            total_momentum_kg_m_s=np.array(momenta),kinetic_J=np.array(kinetic),
            centroids_m=np.array(centroids),closures=np.array(closures),probes_m=points,
            grid_interpolated_E_V_m=np.array(grid_probe_E),
            boundary_P_coeff=np.array(p_hist),boundary_K_coeff=np.array(k_hist),
            boundary_incident_E_V_m=np.array(incident_E_hist),boundary_incident_B_T=np.array(incident_B_hist),
            boundary_source_points_m=g.ant_src_points,boundary_source_weights=g.source_weights,
            boundary_collocation_points_m=g.solve_obs_points)
    try:
        for step in range(total+1):
            t=step*dt
            if step:
                old={s:bridge.length_code_to_si(cp.asnumpy(pic.pos[s])) for s in [0,1]}
                Ex,Ey,Ez,Bx,By,Bz=fields
                # Leapfrog velocities lag positions by dt/2. The Boris kick
                # uses the preceding integer-time field, as does WarpX.
                kick=float(bridge.efield_si_to_code(body_force(t-dt,cfg)))
                pic.proceed_one_step(Ex,Ey,Ez+kick,Bx,By,Bz,ds,ds,ds,n,n,n,1)
                for s in [0,1]:
                    now=bridge.length_code_to_si(cp.asnumpy(pic.pos[s]))
                    guard=float(np.min(np.minimum(now-origin_si,origin_si+extent_si-now)))
                    min_source_guard=min(min_source_guard,guard)
                    r=float(np.min(segment_distance(old[s],now)))
                    min_radius=min(min_radius,r); max_coord=max(max_coord,float(np.max(np.abs(now))))
                    if (args.metal and r<=R) or max_coord>=half or guard<=0:
                        raise RuntimeError('Particle event invalidates no-collection comparison; do not use this result')
            else:
                min_radius=float(np.min(np.linalg.norm(xyz,axis=1))); max_coord=float(np.max(np.abs(xyz)))
                min_source_guard=float(np.min(np.minimum(xyz-origin_si,origin_si+extent_si-xyz)))
            rho,jx,jy,jz=pic.evaluate_rho_J(ds,ds,ds,n,n,n,1)
            solver.Jefimenko_solver(rho.flatten(),include_antenna=False,face_currents=(jx,jy,jz))
            if args.metal:
                surface_solver.Jefimenko_solver(rho.flatten(),include_antenna=False,face_currents=(jx,jy,jz))
                incident=np.asarray(bridge.efield_code_to_si(surface_solver.sample_current_E_at_points(bridge.length_si_to_code(g.solve_obs_points))))
                if not np.all(np.isfinite(incident)):
                    raise RuntimeError('Nonfinite incident plasma field')
                incident_B=np.asarray(bridge.bfield_code_to_si(surface_solver.sample_current_B_at_points(bridge.length_si_to_code(g.solve_obs_points))))
                if not np.all(np.isfinite(incident_B)): raise RuntimeError('Nonfinite incident B')
                rhs=boundary.boundary_coeffs(incident,incident_B,g)
                history=hp@pprev+hk@kprev
                k=mot.solve_fixed_ridge(prep,-rhs-history)
                p=pprev+dt*((1-boundary.THETA)*kprev+boundary.THETA*k)
                if not (np.all(np.isfinite(k)) and np.all(np.isfinite(p))):
                    raise RuntimeError('Nonfinite MAS state')
                res=rhs+history+mot.operator_now_coeff(prep,k)
                et_rows=2*g.num_collocation_points
                coll_num+=float(res[:et_rows]@res[:et_rows]); coll_den+=float(rhs[:et_rows]@rhs[:et_rows])
                coll_B_num+=float(np.sum((res[et_rows:]/(boundary.MAGNETIC_WEIGHT*boundary.C0))**2))
                coll_B_den+=float(np.sum(incident_B**2))
                if step%4==0:
                    inc=bridge.efield_code_to_si(surface_solver.sample_current_E_at_points(bridge.length_si_to_code(hold_points)))
                    inc_B=bridge.bfield_code_to_si(surface_solver.sample_current_B_at_points(bridge.length_si_to_code(hold_points)))
                    scatter,scatter_B=boundary.field_interval(hold_points,g,pprev,kprev,k,dt)
                    hold_B_num+=float(np.sum(np.sum((inc_B+scatter_B)*hold_norm,axis=1)**2))
                    hold_B_den+=float(np.sum(inc_B**2))
                    hold_num+=float(np.sum(mot.tangential_sq(inc+scatter,hold_norm)))
                    hold_den+=float(np.sum(mot.tangential_sq(inc,hold_norm)))
                K=mot.coeff_to_vectors(g,k); P=mot.coeff_to_vectors(g,p)
                antenna_args=dict(antenna_surface_currents=bridge.surface_current_density_si_to_code(K),
                    antenna_polarization=bridge.surface_polarization_density_si_to_code(P))
                solver.recompute_current_fields(**antenna_args,include_antenna=True)
                closure=np.linalg.norm((p-pprev)/dt-((1-boundary.THETA)*kprev+boundary.THETA*k))/max(np.linalg.norm(k)+np.linalg.norm(kprev),1e-300)
                p_hist.append(p.copy()); k_hist.append(k.copy())
                incident_E_hist.append(incident.copy()); incident_B_hist.append(incident_B.copy())
                pprev,kprev=p,k
            else:
                closure=0.
                antenna_args={}
            all_fields=solver.acquire_EB_field()
            if not all(bool(cp.all(cp.isfinite(f))) for f in all_fields):
                raise RuntimeError('Nonfinite field')
            fields=all_fields
            es=[]; bs=[]
            for ps in probe_solvers:
                ps.Jefimenko_solver(rho.flatten(),face_currents=(jx,jy,jz),include_antenna=True,**antenna_args)
                pf=ps.acquire_EB_field()
                es.append([float(cp.asnumpy(f)[0]) for f in pf[:3]])
                bs.append([float(cp.asnumpy(f)[0]) for f in pf[3:]])
            E=bridge.efield_code_to_si(np.array(es)); B=bridge.bfield_code_to_si(np.array(bs))
            if not (np.all(np.isfinite(E)) and np.all(np.isfinite(B))):
                raise RuntimeError('Nonfinite independent probe field')
            grid_probe_E.append(bridge.efield_code_to_si(solver.sample_current_E_at_points(bridge.length_si_to_code(points))))
            # Only E has an existing sampler API. Independent B probes remain native.
            state_p=[]; state_ke=[]; center=[]
            for s,key in [(0,'electron_mass_kg'),(1,'ion_mass_kg')]:
                v=bridge.velocity_code_to_si(cp.asnumpy(pic.velocity[s]))
                if not np.all(np.isfinite(v)) or np.any(np.sum(v*v,axis=1)>=cfg['c_m_s']**2):
                    raise RuntimeError('Invalid particle velocity')
                gamma=1/np.sqrt(1-np.sum(v*v,axis=1)/cfg['c_m_s']**2)
                state_p.append(W*cfg[key]*np.sum(gamma[:,None]*v,axis=0))
                # Stable nonrelativistic-limit expression for gamma-1.
                state_ke.append(W*cfg[key]*np.sum(gamma*gamma*np.sum(v*v,axis=1)/(gamma+1)))
                center.append(np.mean(bridge.length_code_to_si(cp.asnumpy(pic.pos[s])),axis=0))
            max_continuity=max(max_continuity,float(getattr(pic,'last_charge_continuity_error_after',0.)))
            diag=getattr(pic,'last_continuity_diagnostics',{})
            if diag.get('valid',False):
                d=diag['face_current']
                continuity_num+=d['rms_abs']**2
                continuity_den+=(d['drho_dt_rms']+d['div_j_rms'])**2
            times.append(t); probes.append(E); bprobes.append(B); momenta.append(state_p); kinetic.append(state_ke); centroids.append(center); closures.append(closure)
            if step%10==0:
                print(f'step={step}/{total} t={t:.6e} Eprobe0={E[0,2]:.6e}',flush=True)
                record_report()
        completed=True
        np.savez_compressed(out/'final_state.npz',time_s=times[-1],
            E_V_m=np.stack([bridge.efield_code_to_si(cp.asnumpy(f)).reshape(n,n,n) for f in fields[:3]],axis=-1),
            B_T=np.stack([bridge.bfield_code_to_si(cp.asnumpy(f)).reshape(n,n,n) for f in fields[3:]],axis=-1),
            rho_C_m3=bridge.volume_charge_density_code_to_si(cp.asnumpy(rho)),
            electron_positions_m=bridge.length_code_to_si(cp.asnumpy(pic.pos[0])),
            ion_positions_m=bridge.length_code_to_si(cp.asnumpy(pic.pos[1])),
            origin_m=origin_si,spacing_m=np.full(3,ds_si))
        record_report()
    except Exception as e:
        record_report(str(e))
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); ap.add_argument('--case',required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--grid',type=int,default=24); ap.add_argument('--sources',type=int,default=192)
    ap.add_argument('--dt-r-over-c',type=float,default=2.2)
    ap.add_argument('--max-steps',type=int,help='Smoke test only; never qualifies the common-duration comparison')
    ap.add_argument('--metal',action=argparse.BooleanOptionalAction,default=True)
    run(ap.parse_args())
