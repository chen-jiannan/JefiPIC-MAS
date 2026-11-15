'Driven two-arm example: reflecting sources. Physical discretization and numerical expressions are retained from the executed campaign; only imports, paths and release verification are reorganized.'
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(*args,**kwargs):
        def decorate(f):return f
        return decorate

@njit(cache=True)
def _coordinate(x,left,spacing,n):
    return min(max((x-left)/spacing-.5,0.),float(n-1))

@njit(cache=True)
def _weight(s,i):
    b=int(np.floor(s));f=s-b
    if i==b:return 1-f
    if i==b+1:return f
    return 0.

@njit(cache=True)
def deposit_rho(rho,points,charge,left,spacing):
    nx,ny,nz=rho.shape
    for p in points:
        if not (left[0]<p[0]<left[0]+nx*spacing[0] and left[1]<p[1]<left[1]+ny*spacing[1] and left[2]<p[2]<left[2]+nz*spacing[2]):
            raise ValueError('Reflective source particle lies outside physical box')
        sx=_coordinate(p[0],left[0],spacing[0],nx);sy=_coordinate(p[1],left[1],spacing[1],ny);sz=_coordinate(p[2],left[2],spacing[2],nz)
        ix=int(np.floor(sx));iy=int(np.floor(sy));iz=int(np.floor(sz))
        for i in range(ix,min(ix+2,nx)):
            for j in range(iy,min(iy+2,ny)):
                for k in range(iz,min(iz+2,nz)):
                    rho[i,j,k]+=charge*_weight(sx,i)*_weight(sy,j)*_weight(sz,k)

@njit(cache=True)
def _axis(a,b,left,dx,n,indices,mean,delta,flux):
    sa=_coordinate(a,left,dx,n);sb=_coordinate(b,left,dx,n)
    lo=min(int(np.floor(sa)),int(np.floor(sb)));hi=min(max(int(np.floor(sa)),int(np.floor(sb)))+1,n-1)
    count=hi-lo+1
    if count>4:raise ValueError('Unsplit CIC trajectory')
    running=0.;flux[0]=0.
    for j in range(count):
        i=lo+j;wa=_weight(sa,i);wb=_weight(sb,i)
        indices[j]=i;mean[j]=.5*(wa+wb);delta[j]=wb-wa
        running-=delta[j];flux[j+1]=running
    return count

@njit(cache=True)
def deposit_current(jx,jy,jz,previous,current,active,charge,dt,left,spacing):
    nx=jx.shape[0]-1;ny=jy.shape[1]-1;nz=jz.shape[2]-1
    ns=np.array([nx,ny,nz]);idx=np.empty((3,4),np.int64);m=np.empty((3,4));d=np.empty((3,4));flux=np.empty((3,5))
    cuts=np.empty(nx+ny+nz+8);cnt=np.empty(3,np.int64)
    for particle in range(len(current)):
        if not active[particle]:continue
        p0=previous[particle];p1=current[particle];dp=p1-p0;cuts[0]=0.;cuts[1]=1.;nc=2
        for dim in range(3):
            if dp[dim]==0:continue
            for j in range(ns[dim]):
                u=(left[dim]+(j+.5)*spacing[dim]-p0[dim])/dp[dim]
                if u>1e-14 and u<1-1e-14:cuts[nc]=u;nc+=1
        # Tiny per-particle insertion sort; repeated simultaneous cuts are safe.
        for i in range(1,nc):
            value=cuts[i];j=i-1
            while j>=0 and cuts[j]>value:cuts[j+1]=cuts[j];j-=1
            cuts[j+1]=value
        for segment in range(nc-1):
            if cuts[segment+1]-cuts[segment]<=1e-14:continue
            pa=p0+dp*cuts[segment];pb=p0+dp*cuts[segment+1]
            for dim in range(3):cnt[dim]=_axis(pa[dim],pb[dim],left[dim],spacing[dim],ns[dim],idx[dim],m[dim],d[dim],flux[dim])
            for fi in range(cnt[0]+1):
                face=idx[0,0]+fi
                if face==0 or face==nx:continue
                factor=charge*spacing[0]/dt*flux[0,fi]
                if factor==0:continue
                for j in range(cnt[1]):
                    for k in range(cnt[2]):jx[face,idx[1,j],idx[2,k]]+=factor*(m[1,j]*m[2,k]+d[1,j]*d[2,k]/12.)
            for fi in range(cnt[1]+1):
                face=idx[1,0]+fi
                if face==0 or face==ny:continue
                factor=charge*spacing[1]/dt*flux[1,fi]
                if factor==0:continue
                for i in range(cnt[0]):
                    for k in range(cnt[2]):jy[idx[0,i],face,idx[2,k]]+=factor*(m[2,k]*m[0,i]+d[2,k]*d[0,i]/12.)
            for fi in range(cnt[2]+1):
                face=idx[2,0]+fi
                if face==0 or face==nz:continue
                factor=charge*spacing[2]/dt*flux[2,fi]
                if factor==0:continue
                for i in range(cnt[0]):
                    for j in range(cnt[1]):jz[idx[0,i],idx[1,j],face]+=factor*(m[0,i]*m[1,j]+d[0,i]*d[1,j]/12.)

def install(pic):
    from types import MethodType
    if pic.boundary_mode!='reflective':raise RuntimeError('Folded CIC adapter requires reflective particle box')
    old_rho=pic._deposit_rho_numpy;old_j=pic._deposit_current_esirkepov_style_numpy
    left=np.array([pic.left_bound_x,pic.left_bound_y,pic.left_bound_z],float);spacing=np.array([pic.dx,pic.dy,pic.dz],float)
    def rho(self,target,points,charge,order):
        if order!=1:raise RuntimeError('Folded reflective CIC is qualified only at order=1')
        deposit_rho(target,points,float(charge),left,spacing)
    def current(self,jx,jy,jz,p0,p1,active,charge,order):
        if order!=1:raise RuntimeError('Folded reflective CIC is qualified only at order=1')
        deposit_current(jx,jy,jz,p0,p1,active,float(charge),float(self.dt),left,spacing)
    pic._deposit_rho_numpy=MethodType(rho,pic);pic._deposit_current_esirkepov_style_numpy=MethodType(current,pic)
    pic._reflective_cic_source_revision='folded-CIC-zero-normal-flux-v1'
    return old_rho,old_j
