'PEC sphere and particle-cloud example: sphere geometry. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
from types import SimpleNamespace
import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigvals

C0 = 299792458.0
EPS0 = 8.8541878128e-12
THETA = 0.6
RIDGE_REL = 1.e-3
VERSION = 'sphere-hermite-mixed-v1'


def geometry(radius, count=192):
    if count != 192:
        raise ValueError('This qualified sphere discretization has 192 sources.')
    def shell(ntheta, nphi, phase):
        mu, weights = np.polynomial.legendre.leggauss(ntheta)
        phi = np.arange(nphi)*2*np.pi/nphi + phase
        u, p = np.meshgrid(mu, phi, indexing='ij')
        u, p = u.ravel(), p.ravel()
        s = np.sqrt(1-u*u)
        normal = np.column_stack((s*np.cos(p), s*np.sin(p), u))
        t1 = np.column_stack((-np.sin(p), np.cos(p), np.zeros(len(p))))
        return normal, t1, np.cross(normal, t1), np.repeat(weights, nphi)*2*np.pi*radius**2/nphi
    sn, s1, s2, sw = shell(12, 16, 0.)
    cn, c1, c2, cw = shell(24, 32, .371)
    return SimpleNamespace(num_sources=192, num_collocation_points=768,
        ant_src_points=.65*radius*sn, source_weights=sw,
        source_tangent1=s1, source_tangent2=s2, solve_obs_points=radius*cn,
        solve_normals=cn, solve_tangent1=c1, solve_tangent2=c2, solve_weights=cw)


def basis(a, theta=THETA):
    f0 = (2*theta-1)*a**3 + (1-3*theta)*a*a + a
    f1 = (1-2*theta)*a**3 + (3*theta-1)*a*a
    d0 = 3*(2*theta-1)*a*a + 2*(1-3*theta)*a + 1
    d1 = 1-d0
    dd0 = 6*(2*theta-1)*a + 2*(1-3*theta)
    return f0, f1, d0, d1, dd0, -dd0


def hermite_state(P0, P1, K0, K1, alpha, dt):
    """Independent endpoint formulation; works on scalar or broadcast arrays."""
    a = alpha
    delta = (P1-P0)/dt
    P = P0+dt*((a**3-2*a*a+a)*K0+(-2*a**3+3*a*a)*delta+(a**3-a*a)*K1)
    K = (3*a*a-4*a+1)*K0+(-6*a*a+6*a)*delta+(3*a*a-2*a)*K1
    Kdot = ((6*a-4)*K0+(-12*a+6)*delta+(6*a-2)*K1)/dt
    return P, K, Kdot


def pair_blocks(points, sources, weights):
    r = np.asarray(points)[:, None, :]-np.asarray(sources)[None, :, :]
    R = np.linalg.norm(r, axis=-1)
    if np.any(R <= 0):
        raise ValueError('Coincident dipole observation/source.')
    rr = r[..., :, None]*r[..., None, :]
    I = np.eye(3)[None, None, :, :]
    w = np.asarray(weights)[None, :, None, None]/(4*np.pi*EPS0)
    gp = w*(3*rr/R[..., None, None]**5-I/R[..., None, None]**3)
    gk = w*(3*rr/R[..., None, None]**4-I/R[..., None, None]**2)/C0
    gd = w*(rr/R[..., None, None]**3-I/R[..., None, None])/(C0*C0)
    return gp, gk, gd, R


def project_tensor(g, pair):
    ot = np.stack((g.solve_tangent1, g.solve_tangent2), axis=1)
    st = np.stack((g.source_tangent1, g.source_tangent2), axis=1)
    return np.einsum('mua,mjab,jvb->mujv', ot, pair, st, optimize=True).reshape(2*g.num_collocation_points, 2*g.num_sources)


def tangent_coeffs(field, t1, t2):
    return np.stack((np.sum(field*t1, axis=-1), np.sum(field*t2, axis=-1)), axis=-1).ravel()


def coeff_to_vectors(g, q):
    q = np.asarray(q).reshape(-1, 2)
    return q[:, :1]*g.source_tangent1+q[:, 1:]*g.source_tangent2


def tangential_sq(field, normals):
    et = field-normals*np.sum(field*normals, axis=1)[:, None]
    return np.sum(np.abs(et)**2, axis=1)


def prepare_fixed_ridge(A, weights, ridge_rel=RIDGE_REL):
    point_scale = np.sqrt(weights/np.mean(weights))
    rs = np.r_[np.repeat(point_scale, 2), point_scale] if len(A)==3*len(weights) else np.repeat(point_scale, 2)
    A = np.asarray(A)*rs[:, None]
    cs = np.linalg.norm(A, axis=0)
    cs = np.where(cs > 1.e-300, cs, 1.)
    As = A/cs[None, :]
    rng = np.random.default_rng(1234)
    x = rng.normal(size=As.shape[1]); x /= np.linalg.norm(x)
    for _ in range(8):
        x = As.T@(As@x); x /= max(np.linalg.norm(x), 1.e-300)
    smax = np.linalg.norm(As@x)
    alpha = ridge_rel*smax
    H = As.T@As
    H.flat[::len(H)+1] += alpha**2
    return dict(rs=rs, cs=cs, As=As, cf=cho_factor(H, lower=True, check_finite=False),
                ridge_alpha=float(alpha), ridge_rel=ridge_rel)


def solve_fixed_ridge(prep, rhs):
    return cho_solve(prep['cf'], prep['As'].T@(prep['rs']*rhs), check_finite=False)/prep['cs']


def operator_now_coeff(prep, q):
    return (prep['As']@(q*prep['cs']))/prep['rs']


def assemble(g, dt, theta=THETA, ridge_rel=RIDGE_REL):
    gp, gk, gd, R = pair_blocks(g.solve_obs_points, g.ant_src_points, g.source_weights)
    a = 1-R/(C0*dt)
    if np.any(a < 0) or np.any(a > 1):
        raise ValueError('One-step boundary delay exceeds dt.')
    f0, f1, d0, d1, dd0, dd1 = basis(a, theta)
    expand = lambda x: x[..., None, None]
    hp = project_tensor(g, gp)
    hk = project_tensor(g, gp*expand(dt*f0)+gk*expand(d0)+gd*expand(dd0/dt))
    an = project_tensor(g, gp*expand(dt*f1)+gk*expand(d1)+gd*expand(dd1/dt))
    # PEC with zero initial interior B also has B_normal=0. Use c*Bn in
    # electric-field units; this controls magnetic modes weakly seen by Et.
    rv=g.solve_obs_points[:,None,:]-g.ant_src_points[None,:,:]
    n=rv/R[...,None]
    tangents=np.stack([g.source_tangent1,g.source_tangent2],axis=1)
    cross=np.cross(tangents[None,:,:,:],n[:,:,None,:])
    normal=np.einsum('mjva,ma->mjv',cross,g.solve_normals)*g.source_weights[None,:,None]/(4*np.pi*EPS0*C0)
    B0=normal/R[:,:,None]**2;B1=normal/(C0*R[:,:,None])
    hkB=(B0*d0[:,:,None]+B1*(dd0/dt)[:,:,None]).reshape(g.num_collocation_points,-1)
    anB=(B0*d1[:,:,None]+B1*(dd1/dt)[:,:,None]).reshape(g.num_collocation_points,-1)
    hp=np.r_[hp,np.zeros_like(hkB)];hk=np.r_[hk,hkB];an=np.r_[an,anB]
    prep = prepare_fixed_ridge(an, g.solve_weights, ridge_rel)
    delay = dict(max_delay_over_dt=float(np.max(R/(C0*dt))),
                 min_distance_m=float(R.min()), max_distance_m=float(R.max()))
    return hp, hk, prep, delay


def boundary_coeffs(E,B,g):
    return np.r_[tangent_coeffs(E,g.solve_tangent1,g.solve_tangent2),C0*np.sum(B*g.solve_normals,axis=1)]


def advance(p, k, incident_coeffs, hp, hk, prep, dt, theta=THETA):
    kn = solve_fixed_ridge(prep, -incident_coeffs-hp@p-hk@k)
    return p+dt*((1-theta)*k+theta*kn), kn


def field_from_states(points, g, P, K, Kdot):
    """Pairwise SI E/B for states already evaluated at their retarded times."""
    gp, gk, gd, R = pair_blocks(points, g.ant_src_points, g.source_weights)
    E = np.sum(np.einsum('mjab,mjb->mja', gp, P)+np.einsum('mjab,mjb->mja', gk, K)+np.einsum('mjab,mjb->mja', gd, Kdot), axis=1)
    n = (np.asarray(points)[:, None, :]-g.ant_src_points[None, :, :])/R[..., None]
    B = np.sum(g.source_weights[None, :, None]/(4*np.pi*EPS0*C0*C0)*
        (np.cross(K, n)/R[..., None]**2+np.cross(Kdot, n)/(C0*R[..., None])), axis=1)
    return E, B


def field_interval(points, g, p0, k0, k1, dt, theta=THETA):
    R = np.linalg.norm(np.asarray(points)[:, None, :]-g.ant_src_points[None, :, :], axis=-1)
    a = 1-R/(C0*dt)
    if np.any(a < 0) or np.any(a > 1):
        raise ValueError('field_interval is for one-step collocation/holdout only.')
    p1 = p0+dt*((1-theta)*k0+theta*k1)
    states = hermite_state(coeff_to_vectors(g, p0)[None, :, :], coeff_to_vectors(g, p1)[None, :, :],
        coeff_to_vectors(g, k0)[None, :, :], coeff_to_vectors(g, k1)[None, :, :], a[..., None], dt)
    return field_from_states(points, g, *states)


def field_history(points, g, P, K, current, dt, history_nodes=None):
    """Absolute full CPU histories; apply the GPU's finite retention and left-node convention."""
    R = np.linalg.norm(np.asarray(points)[:, None, :]-g.ant_src_points[None, :, :], axis=-1)
    tau = current-R/(C0*dt)
    lo = np.floor(tau).astype(int); a = tau-np.floor(tau)
    oldest = max(0, current-(history_nodes or (current+1))+1)
    node = a < 1.e-12
    # At integer retarded nodes use the left interval for Kdot.
    lo = np.where(node, lo-1, lo); a = np.where(node, 1., a)
    valid = (lo >= oldest)&(lo+1 <= current)
    start = node&(lo == -1)&(oldest == 0)
    safe = np.clip(lo, 0, current); hi = np.minimum(safe+1, current); js = np.arange(g.num_sources)[None, :]
    states = list(hermite_state(P[safe, js], P[hi, js], K[safe, js], K[hi, js], a[..., None], dt))
    for i in range(3):states[i] = states[i]*valid[..., None]
    states[0] += P[0][None, :, :]*start[..., None]
    states[1] += K[0][None, :, :]*start[..., None]
    return field_from_states(points, g, *states)


def homogeneous_spectrum(hp, hk, prep, dt, theta=THETA):
    def inv(rhs):
        return cho_solve(prep['cf'], prep['As'].T@(prep['rs'][:, None]*rhs), check_finite=False)/prep['cs'][:, None]
    X, Y = dt*inv(hp), inv(hk)
    I = np.eye(len(X))
    A = np.block([[I-theta*X, (1-theta)*I-theta*Y], [-X, -Y]])
    z = eigvals(A, check_finite=False)
    return dict(spectral_radius=float(np.max(abs(z))), growth_tolerance=1.e-8,
        modes_above_1_plus_tolerance=int(np.count_nonzero(abs(z)>1+1.e-8)))
