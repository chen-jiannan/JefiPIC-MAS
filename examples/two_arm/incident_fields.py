'Driven two-arm example: incident fields. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
import math
import numpy as np

def _history_python(hist,s,nt,dt,t,r,c):
    tr=t-r/c
    if tr<0:return False,0.,0.
    current=int(math.floor(t/dt+1e-9));oldest=max(current-nt+1,0)
    tau=tr/dt;low=int(math.floor(tau));a=tau-low
    if a<1e-12:
        if low>0:high=low;low-=1;a=1.
        else:high=0;a=0.
    else:high=low+1
    if low<oldest or high<oldest:return False,0.,0.
    if high>current:low=current;high=current;a=0.
    v0=hist[low%nt,s];v1=hist[high%nt,s]
    return True,(1-a)*v0+a*v1,0. if low==high else (v1-v0)/dt

def _point_factory(sample):
    def at_point(px,py,pz,rho,jx,jy,jz,nx,ny,nz,dx,dy,dz,x0,y0,z0,nt,dt,t,c,coeff_E):
        ex=0.;ey=0.;ez=0.;bx=0.;by=0.;bz=0.
        for s in range(nx*ny*nz):
            iz=s%nz;rest=s//nz;iy=rest%ny;ix=rest//ny
            rx=px-(x0+(ix+.5)*dx);ry=py-(y0+(iy+.5)*dy);rz=pz-(z0+(iz+.5)*dz)
            r2=rx*rx+ry*ry+rz*rz
            if r2<=1e-38:continue
            r=math.sqrt(r2);valid,v,d=sample(rho,s,nt,dt,t,r,c)
            if valid:
                f=v/(r2*r)+d/(c*r2);ex+=rx*f;ey+=ry*f;ez+=rz*f
        for axis in range(3):
            if axis==0:hist=jx;ns=(nx+1)*ny*nz
            elif axis==1:hist=jy;ns=nx*(ny+1)*nz
            else:hist=jz;ns=nx*ny*(nz+1)
            for s in range(ns):
                if axis==0:
                    iz=s%nz;rest=s//nz;iy=rest%ny;ix=rest//ny
                    sx=x0+ix*dx;sy=y0+(iy+.5)*dy;sz=z0+(iz+.5)*dz
                elif axis==1:
                    iz=s%nz;rest=s//nz;iy=rest%(ny+1);ix=rest//(ny+1)
                    sx=x0+(ix+.5)*dx;sy=y0+iy*dy;sz=z0+(iz+.5)*dz
                else:
                    iz=s%(nz+1);rest=s//(nz+1);iy=rest%ny;ix=rest//ny
                    sx=x0+(ix+.5)*dx;sy=y0+(iy+.5)*dy;sz=z0+iz*dz
                rx=px-sx;ry=py-sy;rz=pz-sz;r2=rx*rx+ry*ry+rz*rz
                if r2<=1e-38:continue
                r=math.sqrt(r2);valid,v,d=sample(hist,s,nt,dt,t,r,c)
                if not valid:continue
                fe=-d/(c*c*r);fb=v/(r2*r)+d/(c*r2)
                if axis==0:ex+=fe;by-=rz*fb;bz+=ry*fb
                elif axis==1:ey+=fe;bx+=rz*fb;bz-=rx*fb
                else:ez+=fe;bx-=ry*fb;by+=rx*fb
        return ex*coeff_E,ey*coeff_E,ez*coeff_E,bx*coeff_E/(c*c),by*coeff_E/(c*c),bz*coeff_E/(c*c)
    return at_point

point_cpu=_point_factory(_history_python)
try:
    from numba import cuda
except ImportError:
    cuda=None

if cuda is not None:
    _history_gpu=cuda.jit(device=True)(_history_python)
    _point_gpu=cuda.jit(device=True)(_point_factory(_history_gpu))
    @cuda.jit
    def _kernel(points,rho,jx,jy,jz,nx,ny,nz,dx,dy,dz,x0,y0,z0,nt,dt,t,c,coeff_E,out):
        i=cuda.grid(1)
        if i>=len(points):return
        v=_point_gpu(points[i,0],points[i,1],points[i,2],rho,jx,jy,jz,nx,ny,nz,dx,dy,dz,x0,y0,z0,nt,dt,t,c,coeff_E)
        for j in range(6):out[i,j]=v[j]

def sample_solver(s,points):
    """Return small N x 3 E/B arrays in the solver's code units."""
    if cuda is None:raise RuntimeError('Native NumBa CUDA required for direct incident sampling')
    import cupy as cp
    if s.boundary_mode!='open' or s.is_electrostatic or s.grid_truncation_radius!=0:
        raise RuntimeError('Direct incident adapter requires full EM, open EM boundary, no spatial cutoff')
    points=cp.asarray(points,dtype=cp.float64);out=cp.empty((len(points),6),dtype=cp.float64)
    cp.cuda.get_current_stream().synchronize()
    _kernel[(len(points)+127)//128,128](points,s.rho_GPU,s.Jx_face_GPU,s.Jy_face_GPU,s.Jz_face_GPU,
        s.x_grid_size_s,s.y_grid_size_s,s.z_grid_size_s,s.dx_s,s.dy_s,s.dz_s,
        s.x_left_boundary_s,s.y_left_boundary_s,s.z_left_boundary_s,s.len_time_snapshots,s.dt,s.t,s.c,s.coeff_E,out)
    cuda.synchronize()
    v=cp.asnumpy(out)
    return v[:,:3],v[:,3:]
