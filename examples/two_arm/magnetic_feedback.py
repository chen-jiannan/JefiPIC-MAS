'Driven two-arm example: magnetic feedback. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
import math
import numpy as np
C_SI=299792458.

def observation_points(s):
    shape=(s.x_grid_size_o,s.y_grid_size_o,s.z_grid_size_o)
    ijk=np.indices(shape).reshape(3,-1).T
    return np.array([s.x_left_boundary_o,s.y_left_boundary_o,s.z_left_boundary_o])+(ijk+.5)*np.array([s.dx_o,s.dy_o,s.dz_o])

def dual_code_fields(raw,coefficient_E,coefficient_B,bridge):
    # Native raw B has r x K and a negative coefficient_B, exactly as solver.py.
    E=[raw[i]*coefficient_E for i in range(3)]
    B=[raw[i+3]*coefficient_B for i in range(3)]
    return tuple([-C_SI*bridge.E/bridge.B*x for x in B]+[bridge.B/(C_SI*bridge.E)*x for x in E])

def add_feedback_fields(electric_plus_plasma,magnetic):
    if len(electric_plus_plasma)!=6 or len(magnetic)!=6:raise ValueError('Six E/B components required')
    return tuple(a+b for a,b in zip(electric_plus_plasma,magnetic))

class MagneticFeedback:
    def __init__(self,s,points_si,weights_si,bridge,history_nodes=8,*,xp=None,kernel=None,synchronize=None):
        if getattr(s,'is_electrostatic',False) or getattr(s,'boundary_mode','open')!='open':
            raise ValueError('This feedback adapter is qualified for full-EM open-boundary observations')
        native=xp is None
        if native:
            import cupy as xp
        if kernel is None:
            from EMsolver.cuda_functions import Jefimenko_point_dipole_kernel_hermite_parallel
            kernel=Jefimenko_point_dipole_kernel_hermite_parallel
        if synchronize is None:
            if native:
                from numba import cuda
                def synchronize():
                    xp.cuda.get_current_stream().synchronize();cuda.synchronize()
            else:synchronize=lambda:None
        self.xp=xp;self.kernel=kernel;self.synchronize=synchronize;self.bridge=bridge
        self.history_nodes=int(history_nodes);self.last_node=-1;self.count=len(points_si)
        if self.history_nodes<2 or self.count<1:raise ValueError('Invalid magnetic history or source count')
        points_code=np.asarray(bridge.length_si_to_code(points_si),float)
        self.points=[xp.asarray(points_code[:,i]) for i in range(3)]
        self.weights=xp.asarray(bridge.area_si_to_code(weights_si),dtype=xp.float64)
        obs=observation_points(s)
        # For a rectangular grid the largest distance to any source occurs at a corner.
        corners=np.array(np.meshgrid(*[(obs[:,i].min(),obs[:,i].max()) for i in range(3)],indexing='ij')).reshape(3,-1).T
        delay=float(np.max(np.linalg.norm(corners[:,None]-points_code[None],axis=-1))/(s.c*s.dt))
        if math.ceil(delay)+2>self.history_nodes:raise RuntimeError('Magnetic history too short for this observation grid')
        self.maximum_delay_over_dt=delay
        self.P=[xp.zeros((self.history_nodes,self.count),dtype=xp.float64) for _ in range(3)]
        self.K=[xp.zeros_like(x) for x in self.P]
        self.raw=[xp.zeros(s.total_grid_o,dtype=xp.float64) for _ in range(6)]

    def compute(self,s,polarization_si,current_si):
        """Return SIX magnetic-only code-unit arrays; leave solver/history time intact."""
        node=int(math.floor(s.t/s.dt+1e-9))
        if node not in (self.last_node,self.last_node+1) or node<0:
            raise RuntimeError('Magnetic histories must be written from node zero, without skipped nodes')
        p=np.asarray(polarization_si);k=np.asarray(current_si)
        if p.shape!=(self.count,3) or k.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(k).all():
            raise ValueError('Invalid magnetic source state')
        p=self.bridge.surface_polarization_density_si_to_code(p);k=self.bridge.surface_current_density_si_to_code(k)
        slot=node%self.history_nodes
        for i in range(3):self.P[i][slot]=self.xp.asarray(p[:,i]);self.K[i][slot]=self.xp.asarray(k[:,i])
        for array in self.raw:array.fill(0.)
        self.synchronize()
        self.kernel[(s.total_grid_o+127)//128,128](
            *self.points,self.weights,*self.P,*self.K,self.count,0.,0.,0.,0,
            self.history_nodes,s.dt,s.t,s.dx_s,s.dy_s,s.dz_s,s.total_grid_o,
            s.x_grid_size_o,s.y_grid_size_o,s.z_grid_size_o,s.dx_o,s.dy_o,s.dz_o,
            s.x_left_boundary_o,s.y_left_boundary_o,s.z_left_boundary_o,*self.raw,s.c,False)
        self.synchronize();self.last_node=node
        return dual_code_fields(self.raw,s.coeff_E,s.coeff_B,self.bridge)
