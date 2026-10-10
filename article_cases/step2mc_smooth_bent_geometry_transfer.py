#!/usr/bin/env python3
"""
Step 2-MC: smooth bent rounded-dipole geometry-transfer validation.

NUMBERING
---------
Step 2-MB1 established an article-relevant PASS for the straight rounded
2-arm dipole with feed gap using the frozen Step 2-MA construction:
  d/R = 0.80, global FPS sources, tangent2 electric-dipole basis,
  900 sources/arm, 2x collocation, two independent holdouts.

The present Step 2-MC changes ONLY the conductor geometry of the upper arm.
The lower arm remains straight.  The upper arm uses a single gentle circular
bend with tangent-continuous (C1) joins to short straight segments.
Both ends remain hemispherically rounded.  No sharp metallic edge or corner is
introduced.

The upper-arm physical tip-to-tip path length remains 29 mm in the full case.
The bend radius defaults to 8R.  The coherent auxiliary surface uses the SAME
centerline and the frozen d/R=0.80, shrinking only the tube/cap radius from R
to R-d.  There is no retuning of source depth, source count, source basis,
regularization, or acceptance thresholds.

PURPOSE
-------
Test geometry transferability in the intended electrically-small, time-local,
near-field regime: can the straight-dipole MAS closure transfer without
retuning to a simple smooth/moderately-curved antenna?

Article-relevant gating incidents are inherited from Step 2-MB1:
  - uniform,
  - off-axis external electrostatic dipole,
  - bounded localized feed field sech^2(z/w).
The axial antisymmetric tanh field remains a NON-GATING stress/limitation case.

Every holdout region reports local-relative residual, global-normalized
regional residual, and incident-strength ratio eta_r.  The MB1 pre-declared
weak-region rule is unchanged.  MC adds bend-specific regions:
  bend_region, bend_inner, bend_outer, bend_transition.

This script is diagnostic only.  It does NOT modify production modules or the
manuscript.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    import step2kd1_coherent_closed_auxiliary_surface as kd1
    import step2ma_smooth_capsule_acceptance as ma
    import step2mb1_incident_normalization_feed_localization as mb1
except ImportError as exc:
    raise SystemExit(
        "Step 2-MC requires the prior diagnostic scripts: "
        "step2kd1_coherent_closed_auxiliary_surface.py, "
        "step2ma_smooth_capsule_acceptance.py, and "
        "step2mb1_incident_normalization_feed_localization.py in article_cases/."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
TINY = kd1.TINY


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="step2mc_smooth_bent_geometry_transfer")
    p.add_argument("--radius", type=float, default=1.0e-3)
    p.add_argument("--total-length", type=float, default=60.0e-3)
    p.add_argument("--feed-gap", type=float, default=2.0e-3)
    p.add_argument("--depth-frac", type=float, default=0.80)
    p.add_argument("--bend-radius-over-r", type=float, default=30.0)
    p.add_argument("--bend-angle-deg", type=float, default=15.0)
    p.add_argument("--sources-per-arm", type=int, default=900)
    p.add_argument("--collocation-factor", type=float, default=2.0)
    p.add_argument("--lambda-rel", type=float, default=1.0e-4)
    p.add_argument("--dt", type=float, default=1.0e-9)
    p.add_argument("--holdout-gate", type=float, default=0.05)
    p.add_argument("--collocation-gate", type=float, default=0.05)
    p.add_argument("--weak-region-threshold", type=float, default=0.20)
    p.add_argument("--feed-width-over-gap", type=float, default=1.0)
    return p.parse_args()


def _tri_append(tris, meta, tri, **kw):
    tris.append(np.asarray(tri, float))
    meta.append(dict(kw))


def _tube_frame(theta):
    """Planar x-z elbow frame. theta=0 -> +z; theta=pi/2 -> +x."""
    t = np.array([math.sin(theta), 0.0, math.cos(theta)], float)
    n = np.array([math.cos(theta), 0.0, -math.sin(theta)], float)  # toward bend center
    b = np.array([0.0, 1.0, 0.0], float)
    return t, n, b


def bent_centerline_geometry(Rphysical, Larm, gap, Rbend, bend_angle):
    centerline_len = float(Larm - 2.0*Rphysical)
    arc_len = bend_angle*Rbend
    straight_each = 0.5*(centerline_len - arc_len)
    if straight_each <= 1.25*Rphysical:
        raise ValueError(
            "Bend radius is too large for the requested arm length. "
            "Need positive straight segments around the circular elbow."
        )
    p0 = np.array([0.0, 0.0, 0.5*gap + Rphysical], float)
    p_arc0 = p0 + np.array([0.0, 0.0, straight_each])
    center = p_arc0 + np.array([Rbend, 0.0, 0.0])
    p_arc1 = p_arc0 + np.array([Rbend*(1.0-math.cos(bend_angle)), 0.0, Rbend*math.sin(bend_angle)])
    t_end = np.array([math.sin(bend_angle),0.0,math.cos(bend_angle)])
    p1 = p_arc1 + straight_each*t_end
    return {
        "Rphysical": float(Rphysical),
        "Larm": float(Larm),
        "gap": float(gap),
        "Rbend": float(Rbend),
        "bend_angle": float(bend_angle),
        "centerline_len": centerline_len,
        "arc_len": arc_len,
        "straight_pre": straight_each,
        "straight_post": straight_each,
        "p0": p0,
        "p_arc0": p_arc0,
        "bend_center": center,
        "p_arc1": p_arc1,
        "p1": p1,
        "s_arc0": straight_each,
        "s_arc1": straight_each + arc_len,
        "s_end": centerline_len,
    }


def centerline_sample(geom, s):
    s = float(np.clip(s, 0.0, geom["s_end"]))
    if s <= geom["s_arc0"]:
        c = geom["p0"] + np.array([0.0, 0.0, s])
        t, n, b = _tube_frame(0.0)
        kind = "tube_pre"
        theta = 0.0
    elif s <= geom["s_arc1"]:
        u = s - geom["s_arc0"]
        theta = min(geom["bend_angle"], u / geom["Rbend"])
        c = geom["p_arc0"] + np.array([
            geom["Rbend"]*(1.0-math.cos(theta)),
            0.0,
            geom["Rbend"]*math.sin(theta),
        ])
        t, n, b = _tube_frame(theta)
        kind = "tube_arc"
    else:
        u = s - geom["s_arc1"]
        t, n, b = _tube_frame(geom["bend_angle"])
        c = geom["p_arc1"] + u*t
        kind = "tube_post"
        theta = geom["bend_angle"]
    return c, t, n, b, kind, theta


def generate_bent_capsule(radius, Rphysical, Larm, gap, Rbend, bend_angle,
                           n_pre, n_arc, n_post, nphi, nhemi):
    """Triangulated C1 bent capsule with metadata per triangle."""
    geom = bent_centerline_geometry(Rphysical, Larm, gap, Rbend, bend_angle)
    ph = 2.0*np.pi*np.arange(nphi)/nphi

    # Tube rings. Junction rings appear only once.
    spre = np.linspace(0.0, geom["s_arc0"], n_pre+1)
    sarc = np.linspace(geom["s_arc0"], geom["s_arc1"], n_arc+1)[1:]
    spost = np.linspace(geom["s_arc1"], geom["s_end"], n_post+1)[1:]
    svals = np.concatenate((spre, sarc, spost))

    rings = []
    ring_data = []
    for s in svals:
        c,t,n,b,kind,theta = centerline_sample(geom, s)
        ring = c[None,:] + radius*(np.cos(ph)[:,None]*n[None,:] + np.sin(ph)[:,None]*b[None,:])
        rings.append(ring)
        ring_data.append((float(s), kind, float(theta)))

    tris, meta = [], []
    for ir in range(len(rings)-1):
        a, b_ring = rings[ir], rings[ir+1]
        s0, kind0, th0 = ring_data[ir]
        s1, kind1, th1 = ring_data[ir+1]
        sm = 0.5*(s0+s1)
        _,_,_,_,kindm,thm = centerline_sample(geom, sm)
        for j in range(nphi):
            k=(j+1)%nphi
            # circular midpoint robust to wrap-around
            phim = (2.0*np.pi*(j+0.5)/nphi) % (2.0*np.pi)
            common = dict(kind=kindm, s=sm, theta=thm, phi=phim, beta=np.nan)
            _tri_append(tris, meta, [a[j], b_ring[j], b_ring[k]], **common)
            _tri_append(tris, meta, [a[j], b_ring[k], a[k]], **common)

    # Feed cap at p0, pole in -t0 direction.
    c0,t0,n0,b0,_,_ = centerline_sample(geom, 0.0)
    pole0 = c0 - radius*t0
    cap_rings=[]
    betas=np.linspace(0.0, 0.5*np.pi, nhemi+1)[1:]
    for beta in betas:
        cap_rings.append(
            c0[None,:] + radius*(
                -math.cos(beta)*t0[None,:]
                + math.sin(beta)*(np.cos(ph)[:,None]*n0[None,:]+np.sin(ph)[:,None]*b0[None,:])
            )
        )
    first=cap_rings[0]
    beta_mid=0.5*betas[0]
    for j in range(nphi):
        k=(j+1)%nphi
        phim=(2.0*np.pi*(j+0.5)/nphi)%(2.0*np.pi)
        _tri_append(tris, meta, [pole0, first[k], first[j]], kind="feed_cap", s=0.0, theta=0.0, phi=phim, beta=beta_mid)
    for ir in range(len(cap_rings)-1):
        a,b_ring=cap_rings[ir],cap_rings[ir+1]
        beta_mid=0.5*(betas[ir]+betas[ir+1])
        for j in range(nphi):
            k=(j+1)%nphi
            phim=(2.0*np.pi*(j+0.5)/nphi)%(2.0*np.pi)
            common=dict(kind="feed_cap",s=0.0,theta=0.0,phi=phim,beta=beta_mid)
            _tri_append(tris,meta,[a[j],b_ring[k],b_ring[j]],**common)
            _tri_append(tris,meta,[a[j],a[k],b_ring[k]],**common)

    # Outer cap at p1, pole in +t1 direction.
    c1,t1,n1,b1,_,_ = centerline_sample(geom, geom["s_end"])
    pole1 = c1 + radius*t1
    cap_rings=[]
    for beta in betas[::-1]:  # equator -> near pole for convenient connection
        cap_rings.append(
            c1[None,:] + radius*(
                math.cos(beta)*t1[None,:]
                + math.sin(beta)*(np.cos(ph)[:,None]*n1[None,:]+np.sin(ph)[:,None]*b1[None,:])
            )
        )
    # The first ring is equator (beta=pi/2), last is near pole.
    beta_desc=betas[::-1]
    for ir in range(len(cap_rings)-1):
        a,b_ring=cap_rings[ir],cap_rings[ir+1]
        beta_mid=0.5*(beta_desc[ir]+beta_desc[ir+1])
        for j in range(nphi):
            k=(j+1)%nphi
            phim=(2.0*np.pi*(j+0.5)/nphi)%(2.0*np.pi)
            common=dict(kind="outer_cap",s=geom["s_end"],theta=geom["bend_angle"],phi=phim,beta=beta_mid)
            _tri_append(tris,meta,[a[j],b_ring[j],b_ring[k]],**common)
            _tri_append(tris,meta,[a[j],b_ring[k],a[k]],**common)
    last=cap_rings[-1]
    beta_mid=0.5*beta_desc[-1]
    for j in range(nphi):
        k=(j+1)%nphi
        phim=(2.0*np.pi*(j+0.5)/nphi)%(2.0*np.pi)
        _tri_append(tris, meta, [last[j], last[k], pole1], kind="outer_cap", s=geom["s_end"], theta=geom["bend_angle"], phi=phim, beta=beta_mid)

    T=np.asarray(tris,float)
    return T, meta, geom


def project_bent_by_meta(P, meta, geom, radius):
    P=np.asarray(P,float)
    Q=np.empty_like(P)
    N=np.empty_like(P)
    for i,(p,m) in enumerate(zip(P,meta)):
        kind=m["kind"]
        if kind=="feed_cap":
            c=geom["p0"]
            v=p-c
            nv=np.linalg.norm(v)
            n=v/max(nv,TINY)
            Q[i]=c+radius*n; N[i]=n
        elif kind=="outer_cap":
            c=geom["p1"]
            v=p-c
            nv=np.linalg.norm(v)
            n=v/max(nv,TINY)
            Q[i]=c+radius*n; N[i]=n
        else:
            # Exact nearest point on the piecewise straight/circular centerline.
            # Using only a triangle midpoint s here biases the normal because the
            # two triangles of each quad have different barycentric s locations.
            cands=[]
            # feed-side straight segment (+z)
            u=float(np.clip(p[2]-geom["p0"][2],0.0,geom["straight_pre"]))
            c_pre=geom["p0"]+np.array([0.0,0.0,u])
            cands.append(c_pre)
            # circular arc in x-z plane
            qx=float(p[0]-geom["bend_center"][0])
            qz=float(p[2]-geom["bend_center"][2])
            th=float(np.clip(math.atan2(qz,-qx),0.0,geom["bend_angle"]))
            c_arc=geom["p_arc0"]+np.array([
                geom["Rbend"]*(1.0-math.cos(th)),0.0,geom["Rbend"]*math.sin(th)
            ])
            cands.append(c_arc)
            # outer straight segment along final tangent
            t_end=np.array([math.sin(geom["bend_angle"]),0.0,math.cos(geom["bend_angle"])])
            u=float(np.clip(np.dot(p-geom["p_arc1"],t_end),0.0,geom["straight_post"]))
            c_post=geom["p_arc1"]+u*t_end
            cands.append(c_post)
            dd=[float(np.dot(p-c,p-c)) for c in cands]
            c=cands[int(np.argmin(dd))]
            v=p-c
            nv=np.linalg.norm(v)
            n=v/max(nv,TINY)
            Q[i]=c+radius*n; N[i]=n
    return Q,N


def bent_candidates(T, meta, geom, radius):
    C0,_,A=kd1.tri_geom(T)
    C,N=project_bent_by_meta(C0,meta,geom,radius)
    return C,N,A


def bary_bent(T, weights, meta, geom, radius):
    w=np.asarray(weights,float)
    P0=np.einsum("k,nkj->nj",w,T)
    return project_bent_by_meta(P0,meta,geom,radius)


def upper_masks(meta, R, geom):
    kind=np.asarray([m["kind"] for m in meta],dtype=object)
    s=np.asarray([m["s"] for m in meta],float)
    phi=np.asarray([m["phi"] for m in meta],float)
    beta=np.asarray([m["beta"] for m in meta],float)
    tube=np.char.startswith(kind.astype(str),"tube_")
    feed_cap=kind=="feed_cap"
    outer_cap=kind=="outer_cap"
    cap_eq_beta=math.acos(0.75)  # equator-near band consistent with MB-style 0.75R join window
    feed_transition=(feed_cap & (beta>=cap_eq_beta)) | ((kind=="tube_pre") & (s<=0.75*R))
    outer_transition=(outer_cap & (beta>=cap_eq_beta)) | ((kind=="tube_post") & ((geom["s_end"]-s)<=0.75*R))
    bend_band=1.0*R
    bend_region=kind=="tube_arc"
    bend_transition=(
        ((kind=="tube_pre") & ((geom["s_arc0"]-s)<=bend_band))
        | ((kind=="tube_arc") & (((s-geom["s_arc0"])<=bend_band) | ((geom["s_arc1"]-s)<=bend_band)))
        | ((kind=="tube_post") & ((s-geom["s_arc1"])<=bend_band))
    )
    bend_inner=bend_region & (np.cos(phi)>=0.0)
    bend_outer=bend_region & (np.cos(phi)<0.0)
    tip_beta=math.pi/3.0
    feed_tips=feed_cap & (beta<=tip_beta)
    outer_tips=outer_cap & (beta<=tip_beta)
    all_transition=feed_transition|outer_transition|bend_transition
    side_mid=tube & (~bend_region) & (~all_transition)
    return {
        "global":np.ones(len(meta),bool),
        "feed_rounded":feed_cap,
        "feed_tips":feed_tips,
        "feed_transition":feed_transition,
        "outer_rounded":outer_cap,
        "outer_tips":outer_tips,
        "all_transition":all_transition,
        "side_mid":side_mid,
        "bend_region":bend_region,
        "bend_inner":bend_inner,
        "bend_outer":bend_outer,
        "bend_transition":bend_transition,
        "lower_arm":np.zeros(len(meta),bool),
        "upper_arm":np.ones(len(meta),bool),
    }


def lower_masks(P,Ntri,R,Larm,z_bottom):
    aid=np.zeros(len(P),int)
    base=mb1.classify(P,aid,R,Larm,z_bottom,0.0)
    for k in ("bend_region","bend_inner","bend_outer","bend_transition"):
        base[k]=np.zeros(len(P),bool)
    return base


def combine_masks(a,b):
    keys=sorted(set(a)|set(b))
    out={}
    for k in keys:
        av=a.get(k,np.zeros(len(next(iter(a.values()))),bool))
        bv=b.get(k,np.zeros(len(next(iter(b.values()))),bool))
        out[k]=np.concatenate((av,bv))
    return out


def feature_name(m):
    k=m["kind"]
    if k=="tube_arc":
        return "bend_inner" if math.cos(m["phi"])>=0 else "bend_outer"
    return str(k)


def write_centerline_csv(path,geom,n=241):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["s","x","y","z","tx","ty","tz","segment"])
        for s in np.linspace(0.0,geom["s_end"],n):
            c,t,_,_,kind,_=centerline_sample(geom,s)
            w.writerow([s,*c,*t,kind])


def main():
    a=parse_args()
    out=ROOT/"article_cases_output"/a.out
    out.mkdir(parents=True,exist_ok=True)

    R=float(a.radius); total=float(a.total_length); gap=float(a.feed_gap)
    Larm=0.5*(total-gap)
    frac=float(a.depth_frac); d=frac*R; Rs=(1.0-frac)*R
    Rbend=float(a.bend_radius_over_r)*R
    bend_angle=math.radians(float(a.bend_angle_deg))
    if R<=0 or gap<=0 or Larm<=2*R or not (0<frac<1):
        raise ValueError("Invalid R/gap/Larm/depth settings")
    geom_check=bent_centerline_geometry(R,Larm,gap,Rbend,bend_angle)

    z_bottom=-0.5*total
    if a.quick:
        nphi,ncyl,nhemi=28,36,10
        npre,narc,npost=16,28,16
        nsrc_arm=min(300,int(a.sources_per_arm))
    else:
        nphi,ncyl,nhemi=72,96,24
        npre,narc,npost=34,58,34
        nsrc_arm=int(a.sources_per_arm)
    ncoll_arm=max(nsrc_arm+1,int(math.ceil(float(a.collocation_factor)*nsrc_arm)))

    # Lower arm: identical straight rounded arm used in MB1.
    T0,C0,N0,A0,T0loc=mb1.make_arm(R,R,Larm,z_bottom,ncyl,nphi,nhemi)

    # Upper arm: simple smooth/tangent-continuous bent capsule.
    T1,meta1,geom=generate_bent_capsule(R,R,Larm,gap,Rbend,bend_angle,npre,narc,npost,nphi,nhemi)
    C1,N1,A1=bent_candidates(T1,meta1,geom,R)
    T=np.concatenate((T0,T1),axis=0)
    kd1.write_ascii_stl(out/"step2mc_physical_smooth_bent_dipole.stl",T)
    write_centerline_csv(out/"step2mc_upper_centerline.csv",geom)

    # Frozen physical collocation: same point budget per arm as MB1.
    i0=ma.select_fps(C0,ncoll_arm,seed=3101)
    i1=ma.select_fps(C1,ncoll_arm,seed=3102)
    OP=np.concatenate((C0[i0],C1[i1])); ON=np.concatenate((N0[i0],N1[i1]))
    W0,_=kd1.voronoi_weights(C0,A0,C0[i0]); W1,_=kd1.voronoi_weights(C1,A1,C1[i1])
    OW=np.concatenate((W0,W1))

    # Independent holdout pattern 1.
    H00,HN00=mb1.bary_arm(T0loc,[0.19,0.37,0.44],R,Larm,z_bottom)
    H01,HN01=bary_bent(T1,[0.19,0.37,0.44],meta1,geom,R)
    HP=np.concatenate((H00,H01)); HN=np.concatenate((HN00,HN01))
    masks=combine_masks(lower_masks(H00,HN00,R,Larm,z_bottom),upper_masks(meta1,R,geom))

    # Independent holdout pattern 2.
    H20,HN20=mb1.bary_arm(T0loc,[0.53,0.29,0.18],R,Larm,z_bottom)
    H21,HN21=bary_bent(T1,[0.53,0.29,0.18],meta1,geom,R)
    HP2=np.concatenate((H20,H21)); HN2=np.concatenate((HN20,HN21))
    masks2=combine_masks(lower_masks(H20,HN20,R,Larm,z_bottom),upper_masks(meta1,R,geom))

    # Coherent inner surfaces. Lower stays straight; upper follows same bent centerline.
    Ts0,SC0,SN0,SA0,_=mb1.make_arm(Rs,R,Larm,z_bottom,ncyl,nphi,nhemi)
    Ts1,smeta1,sgeom=generate_bent_capsule(Rs,R,Larm,gap,Rbend,bend_angle,npre,narc,npost,nphi,nhemi)
    SC1,SN1,SA1=bent_candidates(Ts1,smeta1,sgeom,Rs)
    j0=ma.select_fps(SC0,nsrc_arm,seed=4101)
    j1=ma.select_fps(SC1,nsrc_arm,seed=4102)
    SP0,SP1=SC0[j0],SC1[j1]; SNs0,SNs1=SN0[j0],SN1[j1]
    SW0,_=kd1.voronoi_weights(SC0,SA0,SP0); SW1,_=kd1.voronoi_weights(SC1,SA1,SP1)
    area0=float(np.sum(A0)); area1=float(np.sum(A1))
    SW0*=area0/max(float(np.sum(SW0)),TINY); SW1*=area1/max(float(np.sum(SW1)),TINY)
    SP=np.concatenate((SP0,SP1)); SN=np.concatenate((SNs0,SNs1)); SW=np.concatenate((SW0,SW1))
    src_arm_id=np.concatenate((np.zeros(len(SP0),int),np.ones(len(SP1),int)))
    src_feature=np.concatenate((np.array(["lower_straight"]*len(SP0),object),np.array([feature_name(smeta1[j]) for j in j1],object)))
    modes=np.full(len(SP),2,dtype=np.int32); off=kd1.offsets(modes)

    kd1.write_ascii_stl(out/"step2mc_inner_lower_arm.stl",Ts0)
    kd1.write_ascii_stl(out/"step2mc_inner_bent_upper_arm.stl",Ts1)

    nn0=ma.nn_stats(SP0,d); nn1=ma.nn_stats(SP1,d)
    cross_src=mb1.cross_min(SP0,SP1,d); cross_phys=mb1.cross_min(C0,C1,R)
    # True source-to-corresponding-physical radial offset is d by construction.

    print("Step 2-MC smooth bent rounded-dipole geometry-transfer validation")
    print("  Step 3 started                 : NO")
    print(f"  total length / gap / R         : {1e3*total:.3f} / {1e3*gap:.3f} / {1e3*R:.3f} mm")
    print(f"  arm length                     : {1e3*Larm:.3f} mm")
    print(f"  upper bend angle / radius-R    : {math.degrees(bend_angle):.1f} deg / {Rbend/R:.3f}")
    print(f"  upper straight-before/after    : {1e3*geom['straight_pre']:.3f} / {1e3*geom['straight_post']:.3f} mm")
    print(f"  upper circular-arc length      : {1e3*geom['arc_len']:.3f} mm")
    print(f"  fixed depth d/R                : {frac:.3f}")
    print(f"  sources per arm / total        : {nsrc_arm} / {len(SP)}")
    print(f"  unknowns                       : {off[-1]}")
    print(f"  collocation total              : {len(OP)} ({len(OP)/len(SP):.2f}x sources)")
    print(f"  holdout points                 : {len(HP)} + {len(HP2)}")
    print(f"  inner-source cross-arm min/d   : {cross_src:.4f}")

    print(f"\nassembling {2*len(OP)} x {off[-1]} operator ...",flush=True)
    Mat=kd1.assemble(OP,ON,SP,SN,SW,modes,off,a.dt)
    rs,_,cs,As=kd1.prepare_scaled_operator(Mat,OW)
    coldiag=kd1.operator_column_diagnostics(cs,modes,off,SN)
    print("factoring ridge normal matrix once for all incidents ...",flush=True)
    cf,damp,smax=mb1.factor_ridge(As,a.lambda_rel)

    acceptance_incidents=("uniform","external_dipole","feed_localized_sech2")
    stress_incidents=("axial_antisymmetric_stress",)
    incidents={
        "uniform":lambda P:kd1.inc_uniform(P),
        "external_dipole":lambda P:mb1.inc_external(P,total,R),
        "feed_localized_sech2":lambda P:mb1.inc_feed_localized(P,gap,a.feed_width_over_gap),
        "axial_antisymmetric_stress":lambda P:mb1.inc_axial_stress(P,gap),
    }
    gate_regions=(
        "global","feed_rounded","feed_tips","feed_transition",
        "outer_rounded","outer_tips","all_transition","side_mid",
        "bend_region","bend_inner","bend_outer","bend_transition",
    )

    rows=[]; gate_rows=[]; coeff_rows=[]
    for name,fn in incidents.items():
        b=kd1.tangent_rhs(fn(OP),ON)
        x,diag=mb1.solve_factored(As,cs,rs,b,cf,damp)
        coll=float(np.linalg.norm(Mat@x-b)/max(np.linalg.norm(b),TINY))

        E1,_=kd1.eval_field(HP,SP,SN,SW,x,modes,off,a.dt); I1=fn(HP)
        m1=mb1.region_metrics(I1+E1,I1,HN,masks)
        E2,_=kd1.eval_field(HP2,SP,SN,SW,x,modes,off,a.dt); I2=fn(HP2)
        m2=mb1.region_metrics(I2+E2,I2,HN2,masks2)

        row={
            "incident":name,"acceptance_incident":name in acceptance_incidents,
            "collocation_rel_l2":coll,
            "direct_weighted_residual_rel":diag["weighted_residual_rel"],
            "direct_damped_optimality_rel":diag["damped_optimality_rel"],
            "damp":damp,
        }
        for prefix,mm in (("holdout",m1),("spot",m2)):
            for region,vals in mm.items():
                row[f"{prefix}_{region}_rel_rms"]=vals["local_rel"]
                row[f"{prefix}_{region}_local_rel_rms"]=vals["local_rel"]
                row[f"{prefix}_{region}_global_norm_rms"]=vals["global_norm"]
                row[f"{prefix}_{region}_incident_ratio"]=vals["incident_ratio"]
                row[f"{prefix}_{region}_incident_rms"]=vals["incident_rms"]
                row[f"{prefix}_{region}_residual_rms"]=vals["residual_rms"]
        rows.append(row)

        if name in acceptance_incidents:
            gate_rows.append({
                "incident":name,"pattern":"collocation","region":"global",
                "incident_ratio":1.0,"gate_metric":"collocation_rel_l2",
                "gate_value":coll,"gate":a.collocation_gate,"passed":bool(coll<a.collocation_gate),
            })
            for pattern,mm in (("holdout",m1),("spot",m2)):
                for region in gate_regions:
                    vals=mm[region]
                    metric_name,metric_value=mb1.gate_metric(region,vals,a.weak_region_threshold)
                    gate_rows.append({
                        "incident":name,"pattern":pattern,"region":region,
                        "incident_ratio":vals["incident_ratio"],"gate_metric":metric_name,
                        "local_rel_rms":vals["local_rel"],"global_norm_rms":vals["global_norm"],
                        "gate_value":metric_value,"gate":a.holdout_gate,
                        "passed":bool(metric_value<a.holdout_gate),
                    })

        for j in range(len(SP)):
            aa,bb=int(off[j]),int(off[j+1])
            coeff_rows.append({
                "incident":name,"source_index":j,
                "arm":"lower" if src_arm_id[j]==0 else "upper_bent",
                "feature":str(src_feature[j]),"coeff_l2":float(np.linalg.norm(x[aa:bb])),
            })

        status="GATING" if name in acceptance_incidents else "STRESS"
        print(
            f"  {name:27s} [{status}] coll={100*coll:7.3f}% "
            f"global={100*m1['global']['local_rel']:7.3f}% "
            f"bend={100*m1['bend_region']['local_rel']:7.3f}% "
            f"eta_bend={m1['bend_region']['incident_ratio']:6.3f} "
            f"bend_in={100*m1['bend_inner']['local_rel']:7.3f}% "
            f"bend_out={100*m1['bend_outer']['local_rel']:7.3f}%"
        )

    accepted=all(bool(r["passed"]) for r in gate_rows)
    failed=[r for r in gate_rows if not r["passed"]]
    worst_gate=max(float(r["gate_value"]) for r in gate_rows) if gate_rows else None
    bend_gate_rows=[r for r in gate_rows if str(r["region"]).startswith("bend")]
    worst_bend_gate=max(float(r["gate_value"]) for r in bend_gate_rows) if bend_gate_rows else None
    weak_gate_rows=[r for r in gate_rows if r["pattern"]!="collocation" and float(r.get("incident_ratio",1.0))<a.weak_region_threshold]
    worst_local_all=max(float(r[k]) for r in rows for k in r if k.endswith("_local_rel_rms") and r[k] is not None)
    stress_rows=[r for r in rows if not bool(r["acceptance_incident"])]
    worst_stress_local=max([float(r[k]) for r in stress_rows for k in r if k.endswith("_local_rel_rms") and r[k] is not None],default=None)

    source_rows=[]
    for j in range(len(SP)):
        source_rows.append({
            "source_index":j,"arm":"lower" if src_arm_id[j]==0 else "upper_bent",
            "feature":str(src_feature[j]),"x":SP[j,0],"y":SP[j,1],"z":SP[j,2],
            "nx":SN[j,0],"ny":SN[j,1],"nz":SN[j,2],"weight":SW[j],"mode":"tangent2",
        })

    mb1.write_csv(out/"step2mc_incident_results.csv",rows)
    mb1.write_csv(out/"step2mc_gate_results.csv",gate_rows)
    mb1.write_csv(out/"step2mc_source_cloud.csv",source_rows)
    mb1.write_csv(out/"step2mc_coefficient_norms.csv",coeff_rows)

    summary={
        "experiment":"Step 2-MC","step3_started":False,
        "purpose":"smooth bent rounded-dipole geometry-transfer validation without MAS retuning",
        "geometry":"straight lower arm + tangent-continuous gentle circular-bend upper arm",
        "sharp_edges":False,"centerline_continuity":"C1 tangent-continuous; curvature changes at straight/arc joins",
        "radius":R,"total_length":total,"feed_gap":gap,"arm_length":Larm,
        "bend_radius":Rbend,"bend_radius_over_R":Rbend/R,"bend_angle_deg":math.degrees(bend_angle),
        "upper_centerline_length":geom["centerline_len"],"upper_arc_length":geom["arc_len"],
        "upper_straight_pre":geom["straight_pre"],"upper_straight_post":geom["straight_post"],
        "depth_over_R":frac,"sources_per_arm":nsrc_arm,"source_count":len(SP),
        "unknown_count":int(off[-1]),"collocation_points":len(OP),
        "holdout_points_primary":len(HP),"holdout_points_spot":len(HP2),
        "source_nn_lower_p50_over_d":nn0["p50"],"source_nn_upper_p50_over_d":nn1["p50"],
        "source_cross_arm_min_over_d":cross_src,"physical_cross_arm_min_over_R":cross_phys,
        "column_norm_spread":coldiag["all"]["max_over_min"],"smax_scaled_est":smax,"damp":damp,
        "acceptance_incidents":list(acceptance_incidents),"non_gating_stress_incidents":list(stress_incidents),
        "weak_region_threshold":float(a.weak_region_threshold),
        "feed_localized_definition":{"Ez":"sech^2(z/w)","width_over_gap":float(a.feed_width_over_gap),"width_m":float(a.feed_width_over_gap*gap)},
        "worst_acceptance_gate_value":worst_gate,"worst_bend_gate_value":worst_bend_gate,
        "weak_region_gate_count":len(weak_gate_rows),"formal_gate_failure_count":len(failed),
        "worst_local_relative_reported_all_incidents":worst_local_all,
        "worst_non_gating_stress_local_relative":worst_stress_local,
        "accepted":bool(accepted),"production_integration_authorized":False,
        "incidents":rows,"gate_results":gate_rows,
    }
    with open(out/"step2mc_report.json","w") as f: json.dump(summary,f,indent=2)

    lines=[
        "Step 2-MC smooth bent rounded-dipole geometry-transfer validation",
        "=================================================================","",
        "NUMBERING","  major step                         : Step 2","  branch                             : M","  diagnostic                         : MC","  Step 3 started                     : NO","",
        "FROZEN MAS / GEOMETRY-TRANSFER DESIGN",
        f"  physical total length / gap / R    : {1e3*total:.3f} / {1e3*gap:.3f} / {1e3*R:.3f} mm",
        f"  arm path length                    : {1e3*Larm:.3f} mm",
        f"  bend angle / radius-R              : {math.degrees(bend_angle):.1f} deg / {Rbend/R:.3f}",
        f"  upper straight pre/post            : {1e3*geom['straight_pre']:.3f} / {1e3*geom['straight_post']:.3f} mm",
        f"  upper circular-arc length            : {1e3*geom['arc_len']:.3f} mm",
        f"  source depth d/R                   : {frac:.3f}",
        "  source sampling                    : global FPS independently on each coherent inner arm",
        "  basis                              : tangent2 electric dipoles",
        f"  sources per arm / total            : {nsrc_arm} / {len(SP)}",
        f"  unknowns                           : {off[-1]}",
        f"  collocation points                 : {len(OP)}",
        f"  holdout points                     : {len(HP)} + {len(HP2)} spot-check","",
        "PRE-DECLARED ACCEPTANCE RULE (UNCHANGED FROM MB1)",
        f"  energized threshold eta_r          : {a.weak_region_threshold:.3f}",
        f"  global / energized-region gate     : local-relative < {100*a.holdout_gate:.2f}%",
        f"  weak-region gate                   : global-normalized residual < {100*a.holdout_gate:.2f}%",
        f"  collocation gate                   : < {100*a.collocation_gate:.2f}%",
        "  axial antisymmetric stress         : REPORTED, NON-GATING","",
        "GEOMETRY / SCALING",
        f"  lower-arm source NN p50/d          : {nn0['p50']:.4f}",
        f"  bent-upper source NN p50/d         : {nn1['p50']:.4f}",
        f"  source cross-arm min/d             : {cross_src:.4f}",
        f"  physical cross-arm min/R           : {cross_phys:.4f}",
        f"  column norm spread                 : {coldiag['all']['max_over_min']:.4e}","",
    ]
    for r in rows:
        status="GATING" if r["acceptance_incident"] else "NON-GATING STRESS"
        lines += [
            f"  {r['incident']} [{status}]",
            f"    collocation                      : {100*r['collocation_rel_l2']:.3f}%",
            f"    global local-relative            : {100*r['holdout_global_local_rel_rms']:.3f}% / spot {100*r['spot_global_local_rel_rms']:.3f}%",
            f"    bend local/global-norm           : {100*r['holdout_bend_region_local_rel_rms']:.3f}% / {100*r['holdout_bend_region_global_norm_rms']:.3f}%",
            f"    bend eta                         : {r['holdout_bend_region_incident_ratio']:.4f}",
            f"    bend-inner local/global-norm     : {100*r['holdout_bend_inner_local_rel_rms']:.3f}% / {100*r['holdout_bend_inner_global_norm_rms']:.3f}%",
            f"    bend-outer local/global-norm     : {100*r['holdout_bend_outer_local_rel_rms']:.3f}% / {100*r['holdout_bend_outer_global_norm_rms']:.3f}%",
            f"    bend-transition local/global-norm: {100*r['holdout_bend_transition_local_rel_rms']:.3f}% / {100*r['holdout_bend_transition_global_norm_rms']:.3f}%",
            f"    feed rounded local/global-norm   : {100*r['holdout_feed_rounded_local_rel_rms']:.3f}% / {100*r['holdout_feed_rounded_global_norm_rms']:.3f}%",
        ]
    lines += [
        "",f"worst formal acceptance gate value   : {100*worst_gate:.3f}%",
        f"worst bend-specific gate value       : {100*worst_bend_gate:.3f}%",
        f"weak-region gate evaluations         : {len(weak_gate_rows)}",
        f"formal gate failures                 : {len(failed)}",
        f"GEOMETRY-TRANSFER MC ACCEPTANCE      : {'PASS' if accepted else 'FAIL'}",
        f"worst local-relative incl. stress    : {100*worst_local_all:.3f}%",
        f"worst NON-GATING axial-stress local  : {100*worst_stress_local:.3f}%" if worst_stress_local is not None else "worst NON-GATING axial-stress local  : n/a",
        "","Step 2-MC PRODUCTION INTEGRATION     : NOT AUTHORIZED BY THIS DIAGNOSTIC","",
        "Interpretation:",
        "  MC is a geometry-transfer test, not a parameter search.",
        "  PASS supports the restricted claim: the frozen time-local MAS closure",
        "  transfers from a straight rounded dipole to a simple smooth/moderately",
        "  curved electrically-small conductor without retuning.",
        "  FAIL does NOT trigger bend-radius/depth/source-count optimization; instead",
        "  the bent geometry claim should be weakened or removed from the manuscript.",
    ]
    (out/"step2mc_summary.txt").write_text("\n".join(lines)+"\n")
    print("\n"+"\n".join(lines))
    if failed:
        print("\nFailed formal gates:")
        for r in failed:
            print(
                f"  {r['incident']:24s} {r['pattern']:10s} {r['region']:16s} "
                f"{r['gate_metric']:12s}={100*float(r['gate_value']):7.3f}% "
                f"eta={float(r.get('incident_ratio',1.0)):.4f}"
            )
    print(f"\nWrote results to: {out}")


if __name__=="__main__":
    main()
