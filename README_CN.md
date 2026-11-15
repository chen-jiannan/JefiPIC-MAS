# JefiPIC-MAS 当前源码

本目录只有当前实现，不包含旧仓库或 legacy 目录，也不使用 v1.2.0 的对外发布号。
两套算例共用根目录 EMsolver、Particle_transport 和 RBG_Maxwell 基础模块。
程序物理参数、公式、粒子库、几何和已验证结果保持不变。

CPU：
    python -m pip install -r requirements-cpu.txt
    python run.py verify
    python run.py cpu
    python run.py sphere-cpu
    python run.py figures

GPU：
    python run.py rf --gpu 0 --busy share --threads 4
    python run.py sphere --gpu 0 --busy share

目录重组后已重新进行 CPU 检查；任何新 GPU 计算仍会先执行原生资格检查。
本轮不要求作者重跑已经完成的 GPU 物理实验。
论文中的完整历史数据仍由 Mendeley 已发布版本保存，无需复制到 GitHub。
