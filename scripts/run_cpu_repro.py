#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys, json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reproduced_results"; OUT.mkdir(parents=True,exist_ok=True)

def run(name,args):
    p=subprocess.run([sys.executable,*args],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (OUT/f"{name}.log").write_text(p.stdout,encoding="utf-8")
    print(p.stdout)
    if p.returncode:
        raise SystemExit(f"{name} failed; see reproduced_results/{name}.log")
    return p.stdout

a=run("01_mas_dipole_continuity",["article_cases/step2a_mas_dipole_source_closure.py","--out",str(OUT/"mas_dipole_continuity")])
b=run("02_conducting_sphere",["article_cases/case2_sphere_uniform_field_induced_charge_test.py","--mas-points","48,96","--source-offsets","0.35R","--eval-offset-fracs","0.10","--surface-plot-offset","0.35R","--publication-formats","png","--out",str(OUT/"conducting_sphere")])
c=run("03_retarded_mot_self_test",["article_cases/step4d_a_two_way_retarded_mot_closure_full_em_atomic.py","--self-test"])
s={
 "mas_dipole_continuity_completed":"OVERALL STEP 2-A: PASS" in a,
 "conducting_sphere_completed":"Finished:" in b,
 "retarded_mot_self_test_pass":'"self_test_pass": true' in c.lower(),
}
s["overall_pass"]=all(s.values())
(OUT/"cpu_reproduction_summary.json").write_text(json.dumps(s,indent=2)+"\n",encoding="utf-8")
print(json.dumps(s,indent=2))
if not s["overall_pass"]: raise SystemExit(2)
