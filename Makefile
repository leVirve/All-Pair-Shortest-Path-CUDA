CC       := g++
CCFLAGS  := -O3
NVCC     := nvcc
NVFLAGS  := -arch=sm_20 -O3
OMPFLAGS := -fopenmp
MPILIBS  := -I/usr/include/mpich-x86_64 -L/usr/lib64/mpich/lib -lmpich
EXES     := block_FW.exe seq_FW.exe HW4_cuda.exe HW4_openmp.exe HW4_mpi.exe

all: $(EXES)

clean:
	rm -f $(EXES)

block_FW.exe: ./sample/block_FW.cpp
	$(CC) $(CCFLAGS) -o block_FW.exe ./sample/block_FW.cpp

seq_FW.exe: ./sample/seq_FW.cpp
	$(CC) $(CCFLAGS) -o seq_FW.exe ./sample/seq_FW.cpp

HW4_cuda.exe: ./src/apsp_cuda.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(OMPFLAGS)" -o HW4_cuda.exe ./src/apsp_cuda.cu

HW4_openmp.exe: ./src/apsp_cuda_openmp.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(OMPFLAGS)" -o HW4_openmp.exe ./src/apsp_cuda_openmp.cu

HW4_mpi.exe: ./src/apsp_cuda_mpi.cu
	$(NVCC) $(NVFLAGS) $(MPILIBS) -o HW4_mpi.exe ./src/apsp_cuda_mpi.cu

test:
	mpirun -np 2 ./HW4_mpi.exe testcase/in$(c) t1 $(s)
	cmp testcase/ans$(c) t1

cuda: ./src/cuda_imp.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(OMPFLAGS)" -o cuda.exe ./src/cuda_imp.cu
