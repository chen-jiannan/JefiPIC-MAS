import cupy as cp
from numba import cuda, float32, float64
import numba.cuda.random as random
import time
import math

# =============== 设备函数 ===============

@cuda.jit(device=True)
def is_in_domain(ind_x, ind_y, ind_z, nx, ny, nz):
    """检查索引是否在计算域内"""
    return (0 <= ind_x < nx) and (0 <= ind_y < ny) and (0 <= ind_z < nz)

@cuda.jit(device=True)
def atomic_add_rho_J(rho, Jx, Jy, Jz, i, j, k, charge, vx, vy, vz, weight):
    """原子添加电荷和电流"""
    if weight != 0.0:
        cuda.atomic.add(rho, (i, j, k), charge * weight)
        cuda.atomic.add(Jx, (i, j, k), charge * vx * weight)
        cuda.atomic.add(Jy, (i, j, k), charge * vy * weight)
        cuda.atomic.add(Jz, (i, j, k), charge * vz * weight)

@cuda.jit(device=True)
def add_field_contribution(ex, ey, ez, bx, by, bz, 
                          Ex, Ey, Ez, Bx, By, Bz, 
                          i, j, k, weight):
    """添加场贡献"""
    if weight != 0.0:
        ex += Ex[i, j, k] * weight
        ey += Ey[i, j, k] * weight
        ez += Ez[i, j, k] * weight
        bx += Bx[i, j, k] * weight
        by += By[i, j, k] * weight
        bz += Bz[i, j, k] * weight
    return ex, ey, ez, bx, by, bz

@cuda.jit(device=True)
def get_tsc_weight_for_offset(offset, di):
    """获取TSC在某个方向的权重"""
    if di == -1:
        return 0.5 * (0.5 - offset) ** 2
    elif di == 0:
        return 0.75 - offset * offset
    else:  # di == 1
        return 0.5 * (0.5 + offset) ** 2

# =============== 主核函数 ===============

@cuda.jit
def find_rho_J(num_particles, pos, velocity, dx, dy, dz, nx, ny, nz, charge, 
               rho, Jx, Jy, Jz, left_bound_x, left_bound_y, left_bound_z, 
               right_bound_x, right_bound_y, right_bound_z, particle_order):
    """计算电荷密度和电流密度"""
    
    i_grid = cuda.grid(1)
    
    if i_grid >= num_particles:
        return
    
    # 检查粒子是否在计算域内
    px = pos[i_grid, 0]
    py = pos[i_grid, 1]
    pz = pos[i_grid, 2]
    
    if not (left_bound_x < px < right_bound_x and 
            left_bound_y < py < right_bound_y and 
            left_bound_z < pz < right_bound_z):
        return
    
    # 当前粒子的速度
    vx = velocity[i_grid, 0]
    vy = velocity[i_grid, 1]
    vz = velocity[i_grid, 2]
    
    # 零阶插值 (NGP - Nearest Grid Point)
    if particle_order == 0:
        ind_x = int((px - left_bound_x) // dx)
        ind_y = int((py - left_bound_y) // dy)
        ind_z = int((pz - left_bound_z) // dz)
        
        if is_in_domain(ind_x, ind_y, ind_z, nx, ny, nz):
            cuda.atomic.add(rho, (ind_x, ind_y, ind_z), charge)
            cuda.atomic.add(Jx, (ind_x, ind_y, ind_z), charge * vx)
            cuda.atomic.add(Jy, (ind_x, ind_y, ind_z), charge * vy)
            cuda.atomic.add(Jz, (ind_x, ind_y, ind_z), charge * vz)
    
    # 一阶插值 (CIC - Cloud-In-Cell)
    elif particle_order == 1:
        # 计算相对位置
        x_norm = (px - left_bound_x) / dx - 0.5
        y_norm = (py - left_bound_y) / dy - 0.5
        z_norm = (pz - left_bound_z) / dz - 0.5
        
        base_i = int(math.floor(x_norm))
        base_j = int(math.floor(y_norm))
        base_k = int(math.floor(z_norm))
        
        # 计算偏移量
        wx = x_norm - base_i
        wy = y_norm - base_j
        wz = z_norm - base_k
        
        # 遍历周围的8个网格点
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = base_i + di
                    j = base_j + dj
                    k = base_k + dk
                    
                    # 跳过边界外的网格点
                    if not is_in_domain(i, j, k, nx, ny, nz):
                        continue
                    
                    # 计算权重
                    weight_x = (1.0 - wx) if di == 0 else wx
                    weight_y = (1.0 - wy) if dj == 0 else wy
                    weight_z = (1.0 - wz) if dk == 0 else wz
                    
                    weight = weight_x * weight_y * weight_z
                    
                    if weight != 0.0:
                        cuda.atomic.add(rho, (i, j, k), charge * weight)
                        cuda.atomic.add(Jx, (i, j, k), charge * vx * weight)
                        cuda.atomic.add(Jy, (i, j, k), charge * vy * weight)
                        cuda.atomic.add(Jz, (i, j, k), charge * vz * weight)
    
    # 二阶插值 (TSC - Triangular Shaped Cloud)
    elif particle_order == 2:
        # 计算相对位置
        x_norm = (px - left_bound_x) / dx
        y_norm = (py - left_bound_y) / dy
        z_norm = (pz - left_bound_z) / dz
        
        base_i = int(math.floor(x_norm))
        base_j = int(math.floor(y_norm))
        base_k = int(math.floor(z_norm))
        
        # 计算相对偏移
        dx_local = x_norm - base_i - 0.5
        dy_local = y_norm - base_j - 0.5
        dz_local = z_norm - base_k - 0.5
        
        # 预计算权重
        wx0 = 0.5 * (0.5 - dx_local) ** 2
        wx1 = 0.75 - dx_local * dx_local
        wx2 = 0.5 * (0.5 + dx_local) ** 2
        
        wy0 = 0.5 * (0.5 - dy_local) ** 2
        wy1 = 0.75 - dy_local * dy_local
        wy2 = 0.5 * (0.5 + dy_local) ** 2
        
        wz0 = 0.5 * (0.5 - dz_local) ** 2
        wz1 = 0.75 - dz_local * dz_local
        wz2 = 0.5 * (0.5 + dz_local) ** 2
        
        # 遍历周围的27个网格点
        for di in range(-1, 2):  # -1, 0, 1
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    i = base_i + di
                    j = base_j + dj
                    k = base_k + dk
                    
                    # 跳过边界外的网格点
                    if not is_in_domain(i, j, k, nx, ny, nz):
                        continue
                    
                    # 根据di,dj,dk选择对应的权重
                    if di == -1:
                        weight_x = wx0
                    elif di == 0:
                        weight_x = wx1
                    else:  # di == 1
                        weight_x = wx2
                        
                    if dj == -1:
                        weight_y = wy0
                    elif dj == 0:
                        weight_y = wy1
                    else:  # dj == 1
                        weight_y = wy2
                        
                    if dk == -1:
                        weight_z = wz0
                    elif dk == 0:
                        weight_z = wz1
                    else:  # dk == 1
                        weight_z = wz2
                    
                    weight = weight_x * weight_y * weight_z
                    
                    if weight != 0.0:
                        cuda.atomic.add(rho, (i, j, k), charge * weight)
                        cuda.atomic.add(Jx, (i, j, k), charge * vx * weight)
                        cuda.atomic.add(Jy, (i, j, k), charge * vy * weight)
                        cuda.atomic.add(Jz, (i, j, k), charge * vz * weight)

@cuda.jit
def find_n_nu(num_particles, pos, dx, dy, dz, number_charge, rho_n, 
              left_bound_x, left_bound_y, left_bound_z, 
              right_bound_x, right_bound_y, right_bound_z):
    """计算粒子数密度（优化版本）"""
    
    i_grid = cuda.grid(1)
    
    if i_grid >= num_particles:
        return
    
    # 检查粒子是否在计算域内
    px, py, pz = pos[i_grid, 0], pos[i_grid, 1], pos[i_grid, 2]
    
    if not (left_bound_x < px < right_bound_x and 
            left_bound_y < py < right_bound_y and 
            left_bound_z < pz < right_bound_z):
        return
    
    # 计算网格索引
    ind_x = int((px - left_bound_x) // dx)
    ind_y = int((py - left_bound_y) // dy)
    ind_z = int((pz - left_bound_z) // dz)
    
    # 原子添加
    cuda.atomic.add(rho_n, (ind_x, ind_y, ind_z), number_charge)

@cuda.jit
def find_EM(num_particles, pos, velocity, nx, ny, nz, 
            Ex_Jefi, Ey_Jefi, Ez_Jefi, Bx_Jefi, By_Jefi, Bz_Jefi, 
            dx, dy, dz, mass, charge, dt, 
            left_bound_x, left_bound_y, left_bound_z, 
            right_bound_x, right_bound_y, right_bound_z, 
            particle_EM, field_order):
    """插值粒子处的电磁场（优化版本）"""
    
    i_grid = cuda.grid(1)
    
    if i_grid >= num_particles:
        return
    
    # 检查粒子是否在计算域内
    px = pos[i_grid, 0]
    py = pos[i_grid, 1]
    pz = pos[i_grid, 2]
    
    if not (left_bound_x < px < right_bound_x and 
            left_bound_y < py < right_bound_y and 
            left_bound_z < pz < right_bound_z):
        return
    
    # 初始化场变量
    ex = 0.0
    ey = 0.0
    ez = 0.0
    bx = 0.0
    by = 0.0
    bz = 0.0
    
    # 零阶插值 (NGP - Nearest Grid Point)
    if field_order == 0:
        ind_x = int((px - left_bound_x) // dx)
        ind_y = int((py - left_bound_y) // dy)
        ind_z = int((pz - left_bound_z) // dz)
        
        if is_in_domain(ind_x, ind_y, ind_z, nx, ny, nz):
            ex = Ex_Jefi[ind_x, ind_y, ind_z]
            ey = Ey_Jefi[ind_x, ind_y, ind_z]
            ez = Ez_Jefi[ind_x, ind_y, ind_z]
            bx = Bx_Jefi[ind_x, ind_y, ind_z]
            by = By_Jefi[ind_x, ind_y, ind_z]
            bz = Bz_Jefi[ind_x, ind_y, ind_z]
    
    # 一阶插值 (CIC - Cloud-In-Cell)
    elif field_order == 1:
        # 计算相对位置
        x_norm = (px - left_bound_x) / dx - 0.5
        y_norm = (py - left_bound_y) / dy - 0.5
        z_norm = (pz - left_bound_z) / dz - 0.5
        
        base_i = int(math.floor(x_norm))
        base_j = int(math.floor(y_norm))
        base_k = int(math.floor(z_norm))
        
        # 计算偏移量
        wx = x_norm - base_i
        wy = y_norm - base_j
        wz = z_norm - base_k
        
        # 遍历周围的8个网格点
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = base_i + di
                    j = base_j + dj
                    k = base_k + dk
                    
                    if is_in_domain(i, j, k, nx, ny, nz):
                        # 计算权重
                        weight_x = (1.0 - wx) if di == 0 else wx
                        weight_y = (1.0 - wy) if dj == 0 else wy
                        weight_z = (1.0 - wz) if dk == 0 else wz
                        
                        weight = weight_x * weight_y * weight_z
                        
                        if weight != 0.0:
                            ex += Ex_Jefi[i, j, k] * weight
                            ey += Ey_Jefi[i, j, k] * weight
                            ez += Ez_Jefi[i, j, k] * weight
                            bx += Bx_Jefi[i, j, k] * weight
                            by += By_Jefi[i, j, k] * weight
                            bz += Bz_Jefi[i, j, k] * weight
    
    # 二阶插值 (TSC - Triangular Shaped Cloud)
    elif field_order == 2:
        # 计算相对位置
        x_norm = (px - left_bound_x) / dx
        y_norm = (py - left_bound_y) / dy
        z_norm = (pz - left_bound_z) / dz
        
        base_i = int(math.floor(x_norm))
        base_j = int(math.floor(y_norm))
        base_k = int(math.floor(z_norm))
        
        # 计算相对偏移
        dx_local = x_norm - base_i - 0.5
        dy_local = y_norm - base_j - 0.5
        dz_local = z_norm - base_k - 0.5
        
        # 预计算权重
        wx0 = 0.5 * (0.5 - dx_local) ** 2
        wx1 = 0.75 - dx_local * dx_local
        wx2 = 0.5 * (0.5 + dx_local) ** 2
        
        wy0 = 0.5 * (0.5 - dy_local) ** 2
        wy1 = 0.75 - dy_local * dy_local
        wy2 = 0.5 * (0.5 + dy_local) ** 2
        
        wz0 = 0.5 * (0.5 - dz_local) ** 2
        wz1 = 0.75 - dz_local * dz_local
        wz2 = 0.5 * (0.5 + dz_local) ** 2
        
        # 遍历周围的27个网格点
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    i = base_i + di
                    j = base_j + dj
                    k = base_k + dk
                    
                    if is_in_domain(i, j, k, nx, ny, nz):
                        # 根据di,dj,dk选择对应的权重
                        if di == -1:
                            weight_x = wx0
                        elif di == 0:
                            weight_x = wx1
                        else:  # di == 1
                            weight_x = wx2
                            
                        if dj == -1:
                            weight_y = wy0
                        elif dj == 0:
                            weight_y = wy1
                        else:  # dj == 1
                            weight_y = wy2
                            
                        if dk == -1:
                            weight_z = wz0
                        elif dk == 0:
                            weight_z = wz1
                        else:  # dk == 1
                            weight_z = wz2
                        
                        weight = weight_x * weight_y * weight_z
                        
                        if weight != 0.0:
                            ex += Ex_Jefi[i, j, k] * weight
                            ey += Ey_Jefi[i, j, k] * weight
                            ez += Ez_Jefi[i, j, k] * weight
                            bx += Bx_Jefi[i, j, k] * weight
                            by += By_Jefi[i, j, k] * weight
                            bz += Bz_Jefi[i, j, k] * weight
    
    # 存储插值后的电磁场
    particle_EM[i_grid, 0] = ex
    particle_EM[i_grid, 1] = ey
    particle_EM[i_grid, 2] = ez
    particle_EM[i_grid, 3] = bx
    particle_EM[i_grid, 4] = by
    particle_EM[i_grid, 5] = bz

@cuda.jit(device=True)
def boris_push(ux, uy, uz, ex, ey, ez, bx, by, bz, charge, mass, dt, c):
    """Boris算法推进粒子"""
    # 预计算常量
    q_dt_over_2m = 0.5 * charge * dt / mass
    
    # 计算当前gamma
    U_squared = ux * ux + uy * uy + uz * uz
    gamma = math.sqrt(1.0 + U_squared / (c * c))
    
    # 计算t向量
    t_factor = q_dt_over_2m / gamma
    tx = t_factor * bx
    ty = t_factor * by
    tz = t_factor * bz
    
    # 计算s向量
    t2 = tx * tx + ty * ty + tz * tz
    inv_denom = 2.0 / (1.0 + t2)
    sx = tx * inv_denom
    sy = ty * inv_denom
    sz = tz * inv_denom
    
    # 半步电场加速 (v-)
    u_minus_x = ux + q_dt_over_2m * ex
    u_minus_y = uy + q_dt_over_2m * ey
    u_minus_z = uz + q_dt_over_2m * ez
    
    # 旋转 (v')
    u_prime_x = u_minus_x + u_minus_y * tz - u_minus_z * ty
    u_prime_y = u_minus_y + u_minus_z * tx - u_minus_x * tz
    u_prime_z = u_minus_z + u_minus_x * ty - u_minus_y * tx
    
    # 旋转 (v+)
    u_plus_x = u_minus_x + u_prime_y * sz - u_prime_z * sy
    u_plus_y = u_minus_y + u_prime_z * sx - u_prime_x * sz
    u_plus_z = u_minus_z + u_prime_x * sy - u_prime_y * sx
    
    # 另外半步电场加速
    ux_new = u_plus_x + q_dt_over_2m * ex
    uy_new = u_plus_y + q_dt_over_2m * ey
    uz_new = u_plus_z + q_dt_over_2m * ez
    
    return ux_new, uy_new, uz_new

@cuda.jit
def update_particle_info(num_particles, pos, velocity, u, nx, ny, nz, 
                         dx, dy, dz, mass, charge, dt, 
                         left_bound_x, left_bound_y, left_bound_z, 
                         right_bound_x, right_bound_y, right_bound_z,
                         particle_EM, field_order, initial_v, initial_u, c,
                         use_reflective=False, use_periodic=False):
    """粒子推进更新函数（优化版本）"""
    
    i_grid = cuda.grid(1)
    if i_grid >= num_particles:
        return
    
    # 检查粒子是否在计算域内
    px = pos[i_grid, 0]
    py = pos[i_grid, 1]
    pz = pos[i_grid, 2]
    
    if not (left_bound_x < px < right_bound_x and
            left_bound_y < py < right_bound_y and
            left_bound_z < pz < right_bound_z):
        return
    
    # 获取电磁场
    ex = particle_EM[i_grid, 0]
    ey = particle_EM[i_grid, 1]
    ez = particle_EM[i_grid, 2]
    bx = particle_EM[i_grid, 3]
    by = particle_EM[i_grid, 4]
    bz = particle_EM[i_grid, 5]
    
    # 当前u
    ux = u[i_grid, 0]
    uy = u[i_grid, 1]
    uz = u[i_grid, 2]
    
    # Boris算法推进
    ux_new, uy_new, uz_new = boris_push(ux, uy, uz, ex, ey, ez, bx, by, bz, 
                                        charge, mass, dt, c)
    
    # 更新u
    u[i_grid, 0] = ux_new
    u[i_grid, 1] = uy_new
    u[i_grid, 2] = uz_new
    
    # 计算新速度和位置
    U_new_squared = ux_new * ux_new + uy_new * uy_new + uz_new * uz_new
    gamma_new = math.sqrt(1.0 + U_new_squared / (c * c))
    
    inv_gamma_new = 1.0 / gamma_new
    vx_new = ux_new * inv_gamma_new
    vy_new = uy_new * inv_gamma_new
    vz_new = uz_new * inv_gamma_new
    
    # 更新速度
    velocity[i_grid, 0] = vx_new
    velocity[i_grid, 1] = vy_new
    velocity[i_grid, 2] = vz_new
    
    # 更新位置
    pos[i_grid, 0] += vx_new * dt
    pos[i_grid, 1] += vy_new * dt
    pos[i_grid, 2] += vz_new * dt
    
    # 边界处理
    if use_reflective:
        # x方向
        if pos[i_grid, 0] < left_bound_x:
            pos[i_grid, 0] = 2.0 * left_bound_x - pos[i_grid, 0]
            u[i_grid, 0] = -u[i_grid, 0]
            velocity[i_grid, 0] = -velocity[i_grid, 0]
        elif pos[i_grid, 0] > right_bound_x:
            pos[i_grid, 0] = 2.0 * right_bound_x - pos[i_grid, 0]
            u[i_grid, 0] = -u[i_grid, 0]
            velocity[i_grid, 0] = -velocity[i_grid, 0]
        
        # y方向
        if pos[i_grid, 1] < left_bound_y:
            pos[i_grid, 1] = 2.0 * left_bound_y - pos[i_grid, 1]
            u[i_grid, 1] = -u[i_grid, 1]
            velocity[i_grid, 1] = -velocity[i_grid, 1]
        elif pos[i_grid, 1] > right_bound_y:
            pos[i_grid, 1] = 2.0 * right_bound_y - pos[i_grid, 1]
            u[i_grid, 1] = -u[i_grid, 1]
            velocity[i_grid, 1] = -velocity[i_grid, 1]
        
        # z方向
        if pos[i_grid, 2] < left_bound_z:
            pos[i_grid, 2] = 2.0 * left_bound_z - pos[i_grid, 2]
            u[i_grid, 2] = -u[i_grid, 2]
            velocity[i_grid, 2] = -velocity[i_grid, 2]
        elif pos[i_grid, 2] > right_bound_z:
            pos[i_grid, 2] = 2.0 * right_bound_z - pos[i_grid, 2]
            u[i_grid, 2] = -u[i_grid, 2]
            velocity[i_grid, 2] = -velocity[i_grid, 2]
    
    elif use_periodic:
        domain_size_x = right_bound_x - left_bound_x
        domain_size_y = right_bound_y - left_bound_y
        domain_size_z = right_bound_z - left_bound_z
        
        # x方向
        if pos[i_grid, 0] < left_bound_x:
            pos[i_grid, 0] += domain_size_x
        elif pos[i_grid, 0] > right_bound_x:
            pos[i_grid, 0] -= domain_size_x
        
        # y方向
        if pos[i_grid, 1] < left_bound_y:
            pos[i_grid, 1] += domain_size_y
        elif pos[i_grid, 1] > right_bound_y:
            pos[i_grid, 1] -= domain_size_y
        
        # z方向
        if pos[i_grid, 2] < left_bound_z:
            pos[i_grid, 2] += domain_size_z
        elif pos[i_grid, 2] > right_bound_z:
            pos[i_grid, 2] -= domain_size_z