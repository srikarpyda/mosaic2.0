############################################################
#	     CUDA KERNELS MAKEFILE
############################################################
#
# Compile the stand-alone CUDA kernels into .ptx files
#
# AUTHOR: Saumya Jain, Srikar Pyda
#
############################################################
############################################################
### PARAMETERS
# GPU options (use empty REGNUBLOCK for default value)
GPU_ARCH    = 37
GPU_SM      = 37
REGNUMBLOCK = 
GPU_OPTIM   = -O3 --use_fast_math
# compilers
NVCC = /usr/local/cuda/bin/nvcc
CC   = g++
# misc
RM = rm -f
############################################################
### FLAGS/OPTIONS
# NVCC flags
NCFLAGS = -ccbin=$(CC) \
          -gencode \
	  arch=compute_$(GPU_ARCH),code=\"compute_$(GPU_ARCH),sm_$(GPU_SM)\" \
	  -m64 $(GPU_OPTIM)
ifdef REGNUMBLOCK
  NCFLAGS += --ptxas-options=-v --maxrregcount=$(REGNUMBLOCK)
endif
############################################################
### TARGETS
# all generic kernel names
KERNELS_ALL = mosaic_cuda
# PTX targets
T_KERNELS_ALL := $(KERNELS_ALL:%=%.ptx)
############################################################
### COMPILATION
all: $(T_KERNELS_ALL)
$(T_KERNELS_ALL): %.ptx: %.cu
	$(NVCC) $< -ptx $(NCFLAGS)
clean:
	$(RM) $(T_KERNELS_ALL)