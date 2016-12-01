CCFLAGS	  := -Wall
NVCCFLAGS := -arch=sm_35 -lrt -Wno-deprecated-gpu-targets -dc
LASTFLAG  := -Wno-deprecated-gpu-targets
LDFLAGS   := -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -lcuda -lcudadevrt -lcudart -lcublas -lpthread -lcusparse
NVCC	  := /usr/local/cuda/bin/nvcc
DISABLEW  := -Xnvlink -w

ODIR = bin
dummy_build_folder := $(shell mkdir -p $(ODIR))

OBJ = cacheGS.o cacheLRU.o cacheMLRU.o cacheMRU.o DataIO.o baseLibsvmReader.o ReadHelper.o\
	  commandLineParser.o fileOps.o gpu_global_utility.o initCuda_cu.o\
	  baseHessian_cu.o accessHessian.o parAccessor.o seqAccessor.o svmProblem.o deviceHessian_cu.o\
	  deviceHessianOnFly_cu.o kernelFunction.o rbfKernelFunction.o\
	  LinearCalculater_cu.o LinearCalGPUHelper_cu.o PolynomialCalGPUHelper_cu.o PolynomialCalculater_cu.o\
	  RBFCalculater_cu.o RBFCalGPUHelper_cu.o SigmoidCalculater_cu.o SigmoidCalGPUHelper_cu.o\
	  devUtility_cu.o storageManager_cu.o hostStorageManager.o classificationKernel_cu.o\
	  smoGPUHelper_cu.o smoSharedSolver_cu.o smoSolver_cu.o svmPredictor_cu.o\
	  svmSharedTrainer_cu.o svmTrainer_cu.o modelSelector_cu.o trainingFunction_cu.o svmModel_cu.o\
	  cvFunction.o svmMain.o

.PHONY: release
.PHONY: debug

release: CCFLAGS += -O2
release: NVCCFLAGS += -O2
release: LASTFLAG += -O2
release: bin/mascot

debug: CCFLAGS += -g
debug: NVCCFLAGS += -G -g
debug: LASTFLAG += -G -g
debug: bin/mascot

bin/mascot: $(OBJ)
	$(NVCC) $(LASTFLAG) $(LDFLAGS) $(DISABLEW) -o bin/mascot $(OBJ)

cvFunction.o: mascot/cvFunction.cpp
	g++ $(CCFLAGS) $(LDFLAGS) -o cvFunction.o -c mascot/cvFunction.cpp

fileOps.o: svm-shared/fileOps.cpp
	g++ $(CCFLAGS) -o fileOps.o -c svm-shared/fileOps.cpp

baseLibsvmReader.o: mascot/DataIOOps/BaseLibsvmReader.cpp mascot/DataIOOps/BaseLibsvmReader.h
	g++ $(CCFLAGS) -o baseLibsvmReader.o -c mascot/DataIOOps/BaseLibsvmReader.cpp

classificationKernel_cu.o: mascot/classificationKernel.h mascot/classificationKernel.cu
	$(NVCC) $(NVCCFLAGS) -o classificationKernel_cu.o -c mascot/classificationKernel.cu

commandLineParser.o: mascot/commandLineParser.h mascot/commandLineParser.cpp
	g++ $(CCFLAGS) -o commandLineParser.o -c mascot/commandLineParser.cpp

gpu_global_utility.o: svm-shared/gpu_global_utility.h svm-shared/gpu_global_utility.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o gpu_global_utility.o -c svm-shared/gpu_global_utility.cu

initCuda_cu.o: svm-shared/initCuda.h svm-shared/initCuda.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o initCuda_cu.o -c svm-shared/initCuda.cu

storageManager_cu.o: svm-shared/storageManager.h svm-shared/storageManager.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o storageManager_cu.o -c svm-shared/storageManager.cu

hostStorageManager.o: svm-shared/hostStorageManager.h svm-shared/hostStorageManager.cpp
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o hostStorageManager.o -c svm-shared/hostStorageManager.cpp

modelSelector_cu.o: mascot/modelSelector.h mascot/modelSelector.cu svm-shared/HessianIO/deviceHessian.h\
					svm-shared/HessianIO/baseHessian.h svm-shared/HessianIO/parAccessor.h svm-shared/HessianIO/seqAccessor.h\
					svm-shared/storageManager.h svm-shared/svmTrainer.h svm-shared/host_constant.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o modelSelector_cu.o -c mascot/modelSelector.cu

devUtility_cu.o: svm-shared/devUtility.h svm-shared/devUtility.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o devUtility_cu.o -c svm-shared/devUtility.cu

smoGPUHelper_cu.o: svm-shared/smoGPUHelper.h svm-shared/devUtility.h svm-shared/smoGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o smoGPUHelper_cu.o -c svm-shared/smoGPUHelper.cu

smoSharedSolver_cu.o: svm-shared/smoSolver.h svm-shared/smoSharedSolver.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o smoSharedSolver_cu.o -c svm-shared/smoSharedSolver.cu

smoSolver_cu.o: svm-shared/smoSolver.h mascot/smoSolver.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o smoSolver_cu.o -c mascot/smoSolver.cu

svmMain.o: mascot/svmMain.cu mascot/commandLineParser.h svm-shared/initCuda.h mascot/cvFunction.h mascot/trainingFunction.h mascot/svmModel.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o svmMain.o -c mascot/svmMain.cu

svmPredictor_cu.o: mascot/svmPredictor.h mascot/svmPredictor.cu svm-shared/host_constant.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o svmPredictor_cu.o -c mascot/svmPredictor.cu

svmSharedTrainer_cu.o: svm-shared/svmTrainer.h svm-shared/svmSharedTrainer.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o svmSharedTrainer_cu.o -c svm-shared/svmSharedTrainer.cu

svmTrainer_cu.o: svm-shared/svmTrainer.h mascot/svmTrainer.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o svmTrainer_cu.o -c mascot/svmTrainer.cu

trainingFunction_cu.o: mascot/trainingFunction.h mascot/trainingFunction.cu svm-shared/host_constant.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o trainingFunction_cu.o -c mascot/trainingFunction.cu

cacheGS.o: svm-shared/Cache/cache.h svm-shared/Cache/cacheGS.cpp
	g++ $(CCFLAGS) -o cacheGS.o -c svm-shared/Cache/cacheGS.cpp

cacheLRU.o: svm-shared/Cache/cache.h svm-shared/Cache/cacheLRU.cpp
	g++ $(CCFLAGS) -o cacheLRU.o -c svm-shared/Cache/cacheLRU.cpp

cacheMLRU.o: svm-shared/Cache/cache.h svm-shared/Cache/cacheMLRU.cpp
	g++ $(CCFLAGS) -o cacheMLRU.o -c svm-shared/Cache/cacheMLRU.cpp

cacheMRU.o: svm-shared/Cache/cache.h svm-shared/Cache/cacheMRU.cpp
	g++ $(CCFLAGS) -o cacheMRU.o -c svm-shared/Cache/cacheMRU.cpp

DataIO.o: mascot/DataIOOps/DataIO.h mascot/DataIOOps/DataIO.cpp
	g++ $(CCFLAGS) -o DataIO.o -c mascot/DataIOOps/DataIO.cpp

ReadHelper.o: mascot/DataIOOps/ReadHelper.cpp
	g++ $(CCFLAGS) -o ReadHelper.o -c mascot/DataIOOps/ReadHelper.cpp

svmProblem.o: mascot/svmProblem.cu mascot/svmProblem.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o svmProblem.o -c mascot/svmProblem.cu

kernelFunction.o: svm-shared/HessianIO/hostKernelCalculater/kernelFunction.cpp svm-shared/HessianIO/hostKernelCalculater/kernelFunction.h
	g++ $(CCFLAGS) -o kernelFunction.o -c svm-shared/HessianIO/hostKernelCalculater/kernelFunction.cpp

rbfKernelFunction.o: svm-shared/HessianIO/hostKernelCalculater/rbfKernelFunction.cpp svm-shared/HessianIO/hostKernelCalculater/rbfKernelFunction.h
	g++ $(CCFLAGS) -o rbfKernelFunction.o -c svm-shared/HessianIO/hostKernelCalculater/rbfKernelFunction.cpp

svmModel_cu.o: mascot/svmModel.cu mascot/svmModel.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o svmModel_cu.o -c mascot/svmModel.cu

baseHessian_cu.o: svm-shared/HessianIO/baseHessian.h svm-shared/HessianIO/baseHessian.cu svm-shared/host_constant.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o baseHessian_cu.o -c svm-shared/HessianIO/baseHessian.cu

accessHessian.o: svm-shared/HessianIO/accessHessian.h svm-shared/HessianIO/accessHessian.cpp
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o accessHessian.o -c svm-shared/HessianIO/accessHessian.cpp

parAccessor.o: svm-shared/HessianIO/parAccessor.h svm-shared/HessianIO/parAccessor.cpp svm-shared/HessianIO/accessHessian.h svm-shared/host_constant.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o parAccessor.o -c svm-shared/HessianIO/parAccessor.cpp

seqAccessor.o: svm-shared/HessianIO/seqAccessor.h svm-shared/HessianIO/seqAccessor.cpp svm-shared/HessianIO/accessHessian.h
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o seqAccessor.o -c svm-shared/HessianIO/seqAccessor.cpp

deviceHessian_cu.o: svm-shared/HessianIO/baseHessian.h svm-shared/HessianIO/deviceHessian.h svm-shared/HessianIO/deviceHessian.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o deviceHessian_cu.o -c svm-shared/HessianIO/deviceHessian.cu

deviceHessianOnFly_cu.o: svm-shared/HessianIO/deviceHessianOnFly.h svm-shared/HessianIO/deviceHessianOnFly.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o deviceHessianOnFly_cu.o -c svm-shared/HessianIO/deviceHessianOnFly.cu

LinearCalculater_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/LinearCalculater.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o LinearCalculater_cu.o -c svm-shared/kernelCalculater/LinearCalculater.cu

LinearCalGPUHelper_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/LinearCalGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o LinearCalGPUHelper_cu.o -c svm-shared/kernelCalculater/LinearCalGPUHelper.cu

PolynomialCalGPUHelper_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/PolynomialCalGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o PolynomialCalGPUHelper_cu.o -c svm-shared/kernelCalculater/PolynomialCalGPUHelper.cu

PolynomialCalculater_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/PolynomialCalculater.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o PolynomialCalculater_cu.o -c svm-shared/kernelCalculater/PolynomialCalculater.cu

RBFCalculater_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/RBFCalculater.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o RBFCalculater_cu.o -c svm-shared/kernelCalculater/RBFCalculater.cu

RBFCalGPUHelper_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/RBFCalGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o RBFCalGPUHelper_cu.o -c svm-shared/kernelCalculater/RBFCalGPUHelper.cu

SigmoidCalculater_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/SigmoidCalculater.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o SigmoidCalculater_cu.o -c svm-shared/kernelCalculater/SigmoidCalculater.cu

SigmoidCalGPUHelper_cu.o: svm-shared/kernelCalculater/kernelCalculater.h svm-shared/kernelCalculater/SigmoidCalGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o SigmoidCalGPUHelper_cu.o -c svm-shared/kernelCalculater/SigmoidCalGPUHelper.cu

.PHONY:clean

clean:
	rm -f *.o bin/hessian2.bin bin/result.txt

