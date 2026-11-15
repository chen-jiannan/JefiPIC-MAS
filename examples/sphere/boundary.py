'PEC sphere and particle-cloud example: boundary. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE_ROOT=ROOT.parents[1]
__import__('sys').path[:0]=[str(SOURCE_ROOT),str(SOURCE_ROOT/'support')]
sys.path.insert(0,str(SOURCE_ROOT))
from EMsolver.mas_mixed_mot import NormalizedMixedMOT,coefficients
import sphere_geometry as legacy
C0=legacy.C0;EPS0=legacy.EPS0
THETA=.6;RIDGE_REL=1e-4;VERSION='normalized-mixed-hermite-mot-v2'
CALIBRATION_PROVENANCE=json.loads((ROOT/'calibration.json').read_text())
MAGNETIC_WEIGHT=float(CALIBRATION_PROVENANCE['beta'])
geometry=legacy.geometry
coeff_to_vectors=legacy.coeff_to_vectors
tangential_sq=legacy.tangential_sq
field_history=legacy.field_history
field_from_states=legacy.field_from_states
hermite_state=legacy.hermite_state

def assemble(g,dt):
    s=NormalizedMixedMOT(g,dt,magnetic_weight=MAGNETIC_WEIGHT,ridge_rel=RIDGE_REL,theta=THETA)
    prep={k:getattr(s,k) for k in ['rs','cs','As','cf']}
    prep.update(ridge_alpha=s.ridge_alpha,ridge_rel=RIDGE_REL,theta=THETA)
    R=np.linalg.norm(g.solve_obs_points[:,None,:]-g.ant_src_points[None,:,:],axis=-1)
    return s.Hp,s.Hk,prep,dict(max_delay_over_dt=float(R.max()/(C0*dt)),magnetic_weight=MAGNETIC_WEIGHT)

def boundary_coeffs(E,B,g):return coefficients(g,E,B,MAGNETIC_WEIGHT)
def solve_fixed_ridge(prep,rhs):return legacy.solve_fixed_ridge(prep,rhs)
def operator_now_coeff(prep,q):return legacy.operator_now_coeff(prep,q)
def field_interval(points,g,p,k,q,dt):return legacy.field_interval(points,g,p,k,q,dt,theta=THETA)
def homogeneous_spectrum(hp,hk,prep,dt):return legacy.homogeneous_spectrum(hp,hk,prep,dt,theta=THETA)
