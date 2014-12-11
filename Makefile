CCFLAGS := -O2 -Wall
NVCCFLAGS := -O2 -arch=sm_20 -lrt -lcuda -lcudadevrt -lcudart -lcublas
LDFLAGS   := -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc
NVCC	:= /usr/local/cuda/bin/nvcc

ODIR = bin
dummy_build_folder := $(shell mkdir -p $(ODIR))

bin/mascot: classificationKernel_cu.o commandLineParser.o cvFunction.o fileOps.o gpu_global_utility.o initCuda_cu.o modelSelector_cu.o smoGPUHelper_cu.o smoSolver_cu.o svmMain.o svmPredictor_cu.o svmTrainer_cu.o trainingFunction_cu.o cacheGS.o cacheLRU.o cacheMLRU.o cacheMRU.o DataIO.o ReadHelper.o hessianIO_cu.o parHessianIO.o seqHessianIO.o LinearCalculater_cu.o LinearCalGPUHelper_cu.o PolynomialCalGPUHelper_cu.o PolynomialCalulater_cu.o RBFCalculater_cu.o RBFCalGPUHelper_cu.o SigmoidCalculater_cu.o SigmoidCalGPUHelper_cu.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o bin/mascot classificationKernel_cu.o commandLineParser.o cvFunction.o fileOps.o gpu_global_utility.o initCuda_cu.o modelSelector_cu.o smoGPUHelper_cu.o smoSolver_cu.o svmMain.o svmPredictor_cu.o svmTrainer_cu.o trainingFunction_cu.o cacheGS.o cacheLRU.o cacheMLRU.o cacheMRU.o DataIO.o ReadHelper.o hessianIO_cu.o parHessianIO.o seqHessianIO.o LinearCalculater_cu.o LinearCalGPUHelper_cu.o PolynomialCalGPUHelper_cu.o PolynomialCalulater_cu.o RBFCalculater_cu.o RBFCalGPUHelper_cu.o SigmoidCalculater_cu.o SigmoidCalGPUHelper_cu.o

cvFunction.o: src/cvFunction.cpp
	g++ $(CCFLAGS) $(LDFLAGS) -o cvFunction.o -c src/cvFunction.cpp

fileOps.o: src/fileOps.cpp
	g++ $(CCFLAGS) -o fileOps.o -c src/fileOps.cpp

classificationKernel_cu.o: src/classificationKernel.h src/classificationKernel.cu
	$(NVCC) $(NVCCFLAGS) -o classificationKernel_cu.o -c src/classificationKernel.cu

commandLineParser.o: src/commandLineParser.h src/commandLineParser.cpp
	g++ $(CCFLAGS) -o commandLineParser.o -c src/commandLineParser.cpp

gpu_global_utility.o: src/gpu_global_utility.h src/gpu_global_utility.cpp
	g++ $(CCFLAGS) $(LDFLAGS) -o gpu_global_utility.o -c src/gpu_global_utility.cpp

initCuda_cu.o: src/initCuda.h src/initCuda.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o initCuda_cu.o -c src/initCuda.cu

modelSelector_cu.o: src/modelSelector.h src/modelSelector.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o modelSelector_cu.o -c src/modelSelector.cu

smoGPUHelper_cu.o: src/smoGPUHelper.h src/smoGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o smoGPUHelper_cu.o -c src/smoGPUHelper.cu

smoSolver_cu.o: src/smoSolver.h src/smoSolver.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o smoSolver_cu.o -c src/smoSolver.cu

svmMain.o: src/svmMain.cpp
	g++ $(CCFLAGS) -o svmMain.o -c src/svmMain.cpp

svmPredictor_cu.o: src/svmPredictor.h src/svmPredictor.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o svmPredictor_cu.o -c src/svmPredictor.cu

svmTrainer_cu.o: src/svmTrainer.h src/svmTrainer.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o svmTrainer_cu.o -c src/svmTrainer.cu

trainingFunction_cu.o: src/trainingFunction.h src/trainingFunction.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o trainingFunction_cu.o -c src/trainingFunction.cu

cacheGS.o: src/Cache/cache.h src/Cache/cacheGS.cpp
	g++ $(CCFLAGS) -o cacheGS.o -c src/Cache/cacheGS.cpp

cacheLRU.o: src/Cache/cache.h src/Cache/cacheLRU.cpp
	g++ $(CCFLAGS) -o cacheLRU.o -c src/Cache/cacheLRU.cpp

cacheMLRU.o: src/Cache/cache.h src/Cache/cacheMLRU.cpp
	g++ $(CCFLAGS) -o cacheMLRU.o -c src/Cache/cacheMLRU.cpp

cacheMRU.o: src/Cache/cache.h src/Cache/cacheMRU.cpp
	g++ $(CCFLAGS) -o cacheMRU.o -c src/Cache/cacheMRU.cpp

DataIO.o: src/DataIOOps/DataIO.h src/DataIOOps/DataIO.cpp
	g++ $(CCFLAGS) -o DataIO.o -c src/DataIOOps/DataIO.cpp

ReadHelper.o: src/DataIOOps/ReadHelper.cpp
	g++ $(CCFLAGS) -o ReadHelper.o -c src/DataIOOps/ReadHelper.cpp

hessianIO_cu.o: src/HessianIO/hessianIO.h src/HessianIO/hessianIO.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o hessianIO_cu.o -c src/HessianIO/hessianIO.cu

parHessianIO.o: src/HessianIO/parHessianIO.h src/HessianIO/parHessianIO.cpp
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o parHessianIO.o -c src/HessianIO/parHessianIO.cpp

seqHessianIO.o: src/HessianIO/seqHessianIO.h src/HessianIO/seqHessianIO.cpp
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o seqHessianIO.o -c src/HessianIO/seqHessianIO.cpp

LinearCalculater_cu.o: src/kernelCalculater/kernelCalculater.h src/kernelCalculater/LinearCalculater.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o LinearCalculater_cu.o -c src/kernelCalculater/LinearCalculater.cu

LinearCalGPUHelper_cu.o: src/kernelCalculater/kernelCalculater.h src/kernelCalculater/LinearCalGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o LinearCalGPUHelper_cu.o -c src/kernelCalculater/LinearCalGPUHelper.cu

PolynomialCalGPUHelper_cu.o: src/kernelCalculater/kernelCalculater.h src/kernelCalculater/PolynomialCalGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o PolynomialCalGPUHelper_cu.o -c src/kernelCalculater/PolynomialCalGPUHelper.cu

PolynomialCalulater_cu.o: src/kernelCalculater/kernelCalculater.h src/kernelCalculater/PolynomialCalulater.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o PolynomialCalulater_cu.o -c src/kernelCalculater/PolynomialCalulater.cu

RBFCalculater_cu.o: src/kernelCalculater/kernelCalculater.h src/kernelCalculater/RBFCalculater.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o RBFCalculater_cu.o -c src/kernelCalculater/RBFCalculater.cu

RBFCalGPUHelper_cu.o: src/kernelCalculater/kernelCalculater.h src/kernelCalculater/RBFCalGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o RBFCalGPUHelper_cu.o -c src/kernelCalculater/RBFCalGPUHelper.cu

SigmoidCalculater_cu.o: src/kernelCalculater/kernelCalculater.h src/kernelCalculater/SigmoidCalculater.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o SigmoidCalculater_cu.o -c src/kernelCalculater/SigmoidCalculater.cu

SigmoidCalGPUHelper_cu.o: src/kernelCalculater/kernelCalculater.h src/kernelCalculater/SigmoidCalGPUHelper.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o SigmoidCalGPUHelper_cu.o -c src/kernelCalculater/SigmoidCalGPUHelper.cu

clean:
	rm -f *.o bin/hessian2.bin bin/result.txt
