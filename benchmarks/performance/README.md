# Plasma field-call measurement

With a compatible CUDA/CuPy environment, run from the repository root:

    CUDA_VISIBLE_DEVICES=0 python benchmarks/performance/field_backends.py

This repeats the same discrete charge/face-current histories for Atomic and
observation-parallel full-EM reductions, with synchronization, warmup and
three trials. It measures the plasma field call, not the coupled PIC-MAS run.
The complete augmented case is profiled by run.py rf --profile.
Reference measurements are archived under data/ and the Mendeley record.
