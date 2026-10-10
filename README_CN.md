# JefiPIC-MAS v1.1.0

这是与最新 JefiPIC-MAS 手稿数值链对应的公开科研代码包。它不是开发历史备份，而是整理后的可复现版本。

## 本版核心变化

- 保留原有一阶 charge-conserving CIC 与 native staggered-face current；
- plasma grid-source 主路径升级为 full electromagnetic Jefimenko；
- 加入 native staggered-current 的 pair-parallel atomic-add（Atomic）实现；
- coincident source/observation grid 的同单元自项使用 integer index 排除；
- observation-parallel 保留为独立 reference/parity 实现；
- 增加最终 full-EM broadband、15 MHz temporal characterization、
  900/arm overfit、full-EM Step4D-A 和 12-cycle Step4D-B 脚本；
- reference_data 明确区分 principal full-EM evidence 和 quasi-electrostatic auxiliary evidence。

这里的 `Atomic` 指 CUDA 原子加法归约，不是原子物理模型。

## CPU 快速检查

```bash
pip install -r requirements-cpu.txt
python scripts/run_cpu_repro.py
pytest -q
```

最终 Step1C 电荷守恒回归会导入 JefiPIC 粒子模块，因此需要正常的
CuPy/JefiPIC 环境；它不放在纯 CPU wrapper 中。安装 CuPy 后单独运行：

```bash
python article_cases/step1c_charge_conserving_cic_regression.py   --out article_cases_output/step1c_charge_conserving_cic_final_atomic
```

## GPU 快速检查

```bash
pip install cupy-cuda12x   # 根据本机 CUDA 版本调整
python scripts/check_environment.py
python scripts/run_gpu_quick.py
```

正式 article-scale 计算不会放进 CI 自动运行。

## 数据

GitHub 版本只保留 compact reference data。完整数值输出、raw/processed
broadband spectrum、Step4C3 forcing、full-EM two-way diagnostics 和
Mendeley Data metadata 模板放在单独的 CPC/Mendeley Data 数据包中。

## 许可

Apache-2.0。
