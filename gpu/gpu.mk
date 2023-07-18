GPU				?= 		AMD
CONFIG 			= $(if $(GPU_CONFIG),-DGPU_CONFIG='"$(GPU_CONFIG)"')
CONFIG			+= $(if $(LONG_BLOCK_SIZE),-D__LONG_BLOCK_SIZE__=\($(LONG_BLOCK_SIZE)\))
CONFIG			+= $(if $(MID_BLOCK_SIZE),-D__MID_BLOCK_SIZE__=\($(MID_BLOCK_SIZE)\))
CONFIG			+= $(if $(SHORT_BLOCK_SIZE),-D__SHORT_BLOCK_SIZE__=\($(SHORT_BLOCK_SIZE)\))
CONFIG			+= $(if $(MID_CUT),-DMM_MID_SEG_CUTOFF=\($(MID_CUT)\))
CONFIG			+= $(if $(LONG_CUT),-DMM_LONG_SEG_CUTOFF=\($(LONG_CUT)\))

###################################################
############  	CPU Compile 	###################
###################################################
CU_SRC			= $(wildcard gpu/*.cu)
CC_SRC			= $(wildcard gpu/*.c)
CU_OBJS			= $(CU_SRC:%.cu=%.o)
CU_OBJS			+= $(CC_SRC:%.c=%.o)
INCLUDES		+= -I gpu

###################################################
############  	CUDA Compile 	###################
###################################################
NVCC 			= nvcc
CUDAFLAGS		= -rdc=true -lineinfo ## turn off assert
CUDATESTFLAG	= -G

###################################################
############	HIP Compile		###################
###################################################
HIPCC			= hipcc
HIPFLAGS		= -DUSEHIP
HIPTESTFLAGS	= -g   

ifneq ($(PRINT_RESOURCE_USAGE), 0)
	HIPTESTFLAGS += -Rpass-analysis=kernel-resource-usage
endif

###################################################
############	DEBUG Options	###################
###################################################
ifeq ($(GPU), AMD)
	GPU_CC 		= $(HIPCC)
	GPU_FLAGS	= $(HIPFLAGS)
	GPU_TESTFL	= $(HIPTESTFLAGS)
else
	GPU_CC 		= $(NVCC)
	GPU_FLAGS	= $(CUDAFLAGS)
	GPU_TESTFL	= $(CUDATESTFLAG)
endif

ifeq ($(NDEBUG), 1)
	CFLAGS 	+= -DNDEBUG
endif

ifneq ($(DEBUG), 0)
	GPU_FLAGS += $(GPU_TESTFL)
	CFLAGS 		+= -DDEBUG_CHECK -DDEBUG_VERBOSE
endif

ifneq ($(CPU_LONG_SEG), 0)
	CFLAGS += -D__CPU_LONG_SEG__=$(CPU_LONG_SEG)
endif



%.o: %.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $(CONFIG) $< -o $@

%.o: %.cu
	$(GPU_CC) -c $(GPU_FLAGS) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $(CONFIG) $< -o $@

cleangpu: 
	rm -f $(CU_OBJS)

# profile:CFLAGS += -pg -g3
# profile:all
# 	perf record --call-graph=dwarf -e cycles:u time ./minimap2 -a test/MT-human.fa test/MT-orang.fa > test.sam

cudep: gpu/.depend

gpu/.depend: $(CU_SRC) $(CC_SRC)
	rm -f gpu/.depend
	$(GPU_CC) -c $(GPU_FLAGS) $(CFLAGS)  $(CPPFLAGS) $(INCLUDES) -MM $^ > $@

include gpu/.depend