*Parallel Programming 2015 HW4 shared materials*

1. Makefile:
- modify this file to meet your requirements

# Make everything:
`> make -j

# Clean executables
`> make clean

# Compile an executable (say `HW4_cuda.exe`):
`> make HW4_cuda.exe


2. Sample code provided:
- seq_FW.cpp
    *  Single-threaded implementation of original FW algorithm

- block_FW.cpp
    *  Single-threaded CPU implementation of blocked FW algorithm
    *  Code has been partition into functions to ease your implementation in CUDA

3. hostfile:
- The -hostfile option to mpirun takes a filename that lists hosts on which to launch MPI processes.
- For example:
> mpirun -np 4 -hostfile hostfile ./myProgram

- Example of a hostfile: (this will assign rank 0-1 on gpucluster1, rank 2-3 on gpucluster2)
> gpucluster1:2
> gpucluster2:2
`

if you have any question, please ask on [iLMS](http://lms.nthu.edu.tw) or email [TA](jacoblee@lsalab.cs.nthu.edu.tw)
Thanks!

