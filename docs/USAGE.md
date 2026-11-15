# Current examples and archived results

Use root run.py. examples/two_arm and examples/sphere contain input,
calibration and qualification fixtures for different physical problems.
The shared kernels are at repository root; there is no nested code/ copy.
support/ holds internal helper modules, not a menu of prior experiments.

Current source fields, source-preservation contracts and scalar/vector
definitions match the executed campaign. Only module/import names, resource
paths, documentation and integrity-check scaffolding are reorganized.
Original executed hash manifests and a relocation map under provenance/
allow comparison to the public archive. They are metadata, not an old code tree.

For static GetDP refinement, run benchmarks/getdp/run_sphere_comparison.py
with --mas-source support/static_sphere.py and installed GetDP/Gmsh paths.
For external SCUFF runs use benchmarks/scuff/run_scuff.py --help.
Comparison outputs and geometry stay within their separately documented scopes.
