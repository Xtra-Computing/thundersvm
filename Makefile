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

OBJ = cacheLAT.o cacheLRU.o cacheMLRU.o cacheMRU.o DataIO.o baseLibsvmReader.o ReadHelper.o\
	  fileOps.o hostStorageManager.o\
	  commandLineParser.o gpu_global_utility.o initCuda_cu.o\
	  baseHessian_cu.o accessHessian.o parAccessor.o seqAccessor.o svmProblem.o deviceHessian_cu.o\
	  deviceHessianOnFly_cu.o kernelFunction.o rbfKernelFunction.o\
	  LinearCalculater_cu.o LinearCalGPUHelper_cu.o PolynomialCalGPUHelper_cu.o PolynomialCalculater_cu.o\
	  RBFCalculater_cu.o RBFCalGPUHelper_cu.o SigmoidCalculater_cu.o SigmoidCalGPUHelper_cu.o\
	  devUtility_cu.o storageManager_cu.o classificationKernel_cu.o\
	  smoGPUHelper_cu.o baseSMO_cu.o smoSharedSolver_cu.o smoSolver_cu.o svmPredictor_cu.o\
	  svmSharedTrainer_cu.o svmTrainer_cu.o modelSelector_cu.o trainingFunction_cu.o svmModel_cu.o\
	  cvFunction.o svmMain.o MultiSmoSolver_cu.o gpuCache.o predictionGPUHelper_cu.o

$(release_bin): $(OBJ)
	$(NVCC) $(LASTFLAG) $(LDFLAGS) $(DISABLEW) -o $@ $(OBJ)
$(debug_bin): $(OBJ)
	$(NVCC) $(LASTFLAG) $(LDFLAGS) $(DISABLEW) -o $@ $(OBJ)

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


cvFunction.o: mascot/cvFunction.cpp
	g++ $(CCFLAGS) $(LDFLAGS) -o $@ -c mascot/cvFunction.cpp

hostStorageManager.o: svm-shared/hostStorageManager.h svm-shared/hostStorageManager.cpp
	$(CXX) $(CCFLAGS) -o $@ -c svm-shared/hostStorageManager.cpp

fileOps.o: svm-shared/fileOps.cpp
	$(CXX) $(CCFLAGS) -o $@ -c svm-shared/fileOps.cpp

baseLibsvmReader.o: mascot/DataIOOps/BaseLibsvmReader.cpp mascot/DataIOOps/BaseLibsvmReader.h
	g++ $(CCFLAGS) -o $@ -c mascot/DataIOOps/BaseLibsvmReader.cpp

classificationKernel_cu.o: mascot/classificationKernel.h mascot/classificationKernel.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c mascot/classificationKernel.cu

commandLineParser.o: mascot/commandLineParser.h mascot/commandLineParser.cpp
	g++ $(CCFLAGS) -o $@ -c mascot/commandLineParser.cpp

gpu_global_utility.o: svm-shared/gpu_global_utility.h svm-shared/gpu_global_utility.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/gpu_global_utility.cu

initCuda_cu.o: svm-shared/initCuda.h svm-shared/initCuda.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/initCuda.cu

storageManager_cu.o: svm-shared/storageManager.h svm-shared/storageManager.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/storageManager.cu

modelSelector_cu.o: mascot/modelSelector.h mascot/modelSelector.cu svm-shared/HessianIO/deviceHessian.h\
					svm-shared/HessianIO/baseHessian.h svm-shared/HessianIO/parAccessor.h svm-shared/HessianIO/seqAccessor.h\
					svm-shared/storageManager.h svm-shared/svmTrainer.h svm-shared/host_constant.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c mascot/modelSelector.cu

devUtility_cu.o: svm-shared/devUtility.h svm-shared/devUtility.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/devUtility.cu

smoGPUHelper_cu.o: svm-shared/smoGPUHelper.h svm-shared/devUtility.h svm-shared/smoGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/smoGPUHelper.cu

smoSharedSolver_cu.o: svm-shared/smoSolver.h svm-shared/smoSharedSolver.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/smoSharedSolver.cu

smoSolver_cu.o: svm-shared/smoSolver.h mascot/smoSolver.cu svm-shared/baseSMO.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c mascot/smoSolver.cu

svmMain.o: mascot/svmMain.cu mascot/commandLineParser.h svm-shared/initCuda.h mascot/cvFunction.h mascot/trainingFunction.h mascot/svmModel.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c mascot/svmMain.cu

svmPredictor_cu.o: mascot/svmPredictor.h mascot/svmPredictor.cu svm-shared/host_constant.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c mascot/svmPredictor.cu

svmSharedTrainer_cu.o: svm-shared/svmTrainer.h svm-shared/svmSharedTrainer.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/svmSharedTrainer.cu

svmTrainer_cu.o: svm-shared/svmTrainer.h mascot/svmTrainer.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c mascot/svmTrainer.cu

trainingFunction_cu.o: mascot/trainingFunction.h mascot/trainingFunction.cu svm-shared/host_constant.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c mascot/trainingFunction.cu

cacheLAT.o: svm-shared/Cache/cache.h svm-shared/Cache/cacheLAT.cpp
	g++ $(CCFLAGS) -o $@ -c svm-shared/Cache/cacheLAT.cpp

cacheLRU.o: svm-shared/Cache/cache.h svm-shared/Cache/cacheLRU.cpp
	g++ $(CCFLAGS) -o $@ -c svm-shared/Cache/cacheLRU.cpp

cacheMLRU.o: svm-shared/Cache/cache.h svm-shared/Cache/cacheMLRU.cpp
	g++ $(CCFLAGS) -o $@ -c svm-shared/Cache/cacheMLRU.cpp

cacheMRU.o: svm-shared/Cache/cache.h svm-shared/Cache/cacheMRU.cpp
	g++ $(CCFLAGS) -o $@ -c svm-shared/Cache/cacheMRU.cpp

gpuCache.o: svm-shared/Cache/gpuCache.h svm-shared/Cache/gpuCache.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/Cache/gpuCache.cu

DataIO.o: mascot/DataIOOps/DataIO.h mascot/DataIOOps/DataIO.cpp
	g++ $(CCFLAGS) -o $@ -c mascot/DataIOOps/DataIO.cpp

ReadHelper.o: mascot/DataIOOps/ReadHelper.cpp
	g++ $(CCFLAGS) -o $@ -c mascot/DataIOOps/ReadHelper.cpp

svmProblem.o: mascot/svmProblem.cu mascot/svmProblem.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c mascot/svmProblem.cu

kernelFunction.o: svm-shared/HessianIO/hostKernelCalculater/kernelFunction.cpp svm-shared/HessianIO/hostKernelCalculater/kernelFunction.h
	g++ $(CCFLAGS) -o $@ -c svm-shared/HessianIO/hostKernelCalculater/kernelFunction.cpp

rbfKernelFunction.o: svm-shared/HessianIO/hostKernelCalculater/rbfKernelFunction.cpp svm-shared/HessianIO/hostKernelCalculater/rbfKernelFunction.h
	g++ $(CCFLAGS) -o $@ -c svm-shared/HessianIO/hostKernelCalculater/rbfKernelFunction.cpp

svmModel_cu.o: mascot/svmModel.cu mascot/svmModel.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c mascot/svmModel.cu

baseHessian_cu.o: svm-shared/HessianIO/baseHessian.h svm-shared/HessianIO/baseHessian.cu svm-shared/host_constant.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/HessianIO/baseHessian.cu

accessHessian.o: svm-shared/HessianIO/accessHessian.h svm-shared/HessianIO/accessHessian.cpp
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/HessianIO/accessHessian.cpp

parAccessor.o: svm-shared/HessianIO/parAccessor.h svm-shared/HessianIO/parAccessor.cpp svm-shared/HessianIO/accessHessian.h svm-shared/host_constant.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/HessianIO/parAccessor.cpp

seqAccessor.o: svm-shared/HessianIO/seqAccessor.h svm-shared/HessianIO/seqAccessor.cpp svm-shared/HessianIO/accessHessian.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/HessianIO/seqAccessor.cpp

deviceHessian_cu.o: svm-shared/HessianIO/baseHessian.h svm-shared/HessianIO/deviceHessian.h svm-shared/HessianIO/deviceHessian.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/HessianIO/deviceHessian.cu

deviceHessianOnFly_cu.o: svm-shared/HessianIO/deviceHessianOnFly.h svm-shared/HessianIO/deviceHessianOnFly.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/HessianIO/deviceHessianOnFly.cu

LinearCalculater_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/LinearCalculater.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/kernelCalculater/LinearCalculater.cu

LinearCalGPUHelper_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/LinearCalGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/kernelCalculater/LinearCalGPUHelper.cu

PolynomialCalGPUHelper_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/PolynomialCalGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/kernelCalculater/PolynomialCalGPUHelper.cu

PolynomialCalculater_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/PolynomialCalculater.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/kernelCalculater/PolynomialCalculater.cu

RBFCalculater_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/RBFCalculater.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/kernelCalculater/RBFCalculater.cu

RBFCalGPUHelper_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/RBFCalGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/kernelCalculater/RBFCalGPUHelper.cu

SigmoidCalculater_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/SigmoidCalculater.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/kernelCalculater/SigmoidCalculater.cu

SigmoidCalGPUHelper_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/SigmoidCalGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/kernelCalculater/SigmoidCalGPUHelper.cu

baseSMO_cu.o: svm-shared/baseSMO.h svm-shared/baseSMO.cu svm-shared/smoGPUHelper.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c svm-shared/baseSMO.cu

MultiSmoSolver_cu.o: mascot/multiSmoSolver.h mascot/multiSmoSolver.cu svm-shared/baseSMO.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c mascot/multiSmoSolver.cu
	
predictionGPUHelper_cu.o: mascot/predictionGPUHelper.*
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ -c mascot/predictionGPUHelper.cu

.PHONY:clean

clean:
	rm -f *.o *.txt bin/*.bin bin/result.txt bin/release/* bin/debug/*

