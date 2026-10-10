from pathlib import Path
import sys, numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"article_cases"))
from EMsolver.mas_dipole_basis import ContinuityConsistentElectricDipoleBasis
from article_cases.case2_sphere_uniform_field_induced_charge_test import run_one_case
from article_cases.step4d_a_two_way_retarded_mot_closure_full_em_atomic import self_test

def test_weak_continuity_identity():
    rng=np.random.default_rng(123)
    r=ContinuityConsistentElectricDipoleBasis.weak_continuity_residual(
        rng.normal(size=(100,3)),rng.normal(size=(100,3)))
    assert np.max(np.abs(r))<=1e-18

def test_conducting_sphere_small_case():
    r=run_one_case(96,0.004,1.0,0.35*0.004,[0.10],0.0,0.0)
    assert r["potential_rms_residual_V"]<1e-12
    assert r["surface_total_tangential_rel_to_incident"]<0.02
    assert r["dipole_z_relative_error"]<0.01

def test_retarded_mot_self_test():
    self_test()
