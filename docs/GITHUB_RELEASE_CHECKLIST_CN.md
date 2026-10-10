# GitHub 发布前最后检查

这个工程包已经可以直接作为仓库根目录使用。

建议公开前再检查：

1. 仓库名称可直接使用 `JefiPIC-MAS`。
2. Description 可写：
   `Grid-decoupled Jefimenko-PIC / MAS framework for causal plasma-conductor coupling.`
3. Topics 可添加：
   `particle-in-cell`, `plasma-physics`, `jefimenko`, `method-of-auxiliary-sources`,
   `computational-physics`, `cuda`, `electromagnetics`.
4. 正式论文 DOI 出来后，更新：
   - `README.md`
   - `CITATION.cff`
5. 如果后续在 Zenodo 对 GitHub release 做归档，可把 Zenodo DOI badge 再加到 README。
6. 建议第一次 release 标记为 `v0.1.0`。

本包没有把完整开发历史和大量淘汰的 sweep 脚本全部公开；它保留的是核心源码、
最终论文链所需脚本、代表性可复现算例和关键机器可读证据。
