CCFLAGS	  := -Wall
NVCCFLAGS := -arch=sm_35 -lrt -Wno-deprecated-gpu-targets -dc
LASTFLAG  := -Wno-deprecated-gpu-targets
LDFLAGS   := -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -lcuda -lcudadevrt -lcudart -lcublas -lpthread -lcusparse
NVCC	  := /usr/local/cuda/bin/nvcc
DISABLEW  := -Xnvlink -w

CXX := g++

ODIR = bin
exe_name = mascot
release_bin := $(ODIR)/release/$(exe_name)
debug_bin := $(ODIR)/debug/$(exe_name)
$(shell mkdir -p $(ODIR)/release)
$(shell mkdir -p $(ODIR)/debug)

OBJ = cacheLAT.o cacheLRU.o cacheMLRU.o cacheMRU.o DataIO.o BaseLibsvmReader.o ReadHelper.o\
	  fileOps.o hostStorageManager.o\
	  commandLineParser.o gpu_global_utility.o initCuda.o\
	  baseHessian.o accessHessian.o parAccessor.o seqAccessor.o svmProblem.o deviceHessian.o\
	  deviceHessianOnFly.o kernelFunction.o rbfKernelFunction.o\
	  LinearCalculater.o LinearCalGPUHelper.o PolynomialCalGPUHelper.o PolynomialCalculater.o\
	  RBFCalculater.o RBFCalGPUHelper.o SigmoidCalculater.o SigmoidCalGPUHelper.o\
	  devUtility.o storageManager.o classificationKernel.o\
	  smoGPUHelper.o baseSMO.o smoSharedSolver.o smoSolver.o svmPredictor.o\
	  svmSharedTrainer.o svmTrainer.o modelSelector.o trainingFunction.o svmModel.o\
	  cvFunction.o svmMain.o multiSmoSolver.o gpuCache.o predictionGPUHelper.o\
	  multiPredictor.o

$(release_bin): $(OBJ)
	$(NVCC) $(LASTFLAG) $(LDFLAGS) $(DISABLEW) -o $@ $^
$(debug_bin): $(OBJ)
	$(NVCC) $(LASTFLAG) $(LDFLAGS) $(DISABLEW) -o $@ $^

.PHONY: release
.PHONY: debug

release: CCFLAGS += -O2
release: NVCCFLAGS += -O2
release: LASTFLAG += -O2
release: $(release_bin)

debug: CCFLAGS += -g
debug: NVCCFLAGS += -G -g
debug: LASTFLAG += -G -g
debug: $(debug_bin)

#compile files of mascot
%.o: mascot/%.c* mascot/*.h svm-shared/*.h svm-shared/HessianIO/*.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c $<

#compile files of svm-shared 
%.o: svm-shared/%.c* svm-shared/*.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c $<
	
#compile caching strategies
%.o: svm-shared/Cache/%.c* svm-shared/Cache/*.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c $<

#compile data reader
%.o: mascot/DataIOOps/%.cpp mascot/DataIOOps/*.h
	$(CXX) $(CCFLAGS) -o $@ -c $<

#compile hessian operators
%.o: svm-shared/HessianIO/%.c* svm-shared/HessianIO/*.h svm-shared/host_constant.h	
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c $<

#compile kernel value calculaters
%.o: svm-shared/kernelCalculater/%.cu svm-shared/kernelCalculater/kernelCalculater.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c $<
%.o: svm-shared/HessianIO/hostKernelCalculater/%.cpp svm-shared/HessianIO/hostKernelCalculater/*.h
	$(CXX) $(CCFLAGS) -o $@ -c $<

.PHONY:clean

clean:
	rm -f *.o *.txt bin/*.bin bin/result.txt bin/release/* bin/debug/*
