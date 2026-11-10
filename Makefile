# =============================================================================
#  Bitcrack-Minikeys-BR  —  Makefile unificado
#
#  Alvos:
#    make minikey           referencia CPU (corretude, vetores conhecidos)
#    make minikey_gpu       triagem GPU (SHA256 + filtro 0x00)
#    make minikey_gpu_full  pipeline completo INGENUO (double-and-add)
#    make minikey_gpu_opt   pipeline OTIMIZADO (comb de base fixa + inversao em lote)
#    make all               todos
#    make clean
#
#  Reusa, de forma enxuta, os modulos criptograficos do BitCrack (MIT, Ben Richard)
#  ja adaptados no BitCrack-BR: secp256k1, CryptoUtil, AddressUtil, util, Logger e
#  os headers de curva da cudaMath. Veja bitcrack/ e LICENSE.MIT.
#
#  Requisitos: CUDA >= 12.8 (sm_120/Blackwell exige 12.8+); g++ compativel (<=14).
#  Ajuste a arquitetura da SUA GPU em SM para um build muito mais rapido, ex.:
#    make minikey_gpu_opt SM="-gencode=arch=compute_86,code=sm_86"   # RTX 30xx
# =============================================================================
BC=bitcrack
INC=-I$(BC)/secp256k1lib -I$(BC)/CryptoUtil -I$(BC)/AddressUtil -I$(BC)/util -I$(BC)/Logger

CXX?=g++
CXXFLAGS=-O2 -std=c++14

CUDA_HOME?=/usr/local/cuda-12.9
NVCC?=$(CUDA_HOME)/bin/nvcc
CCBIN?=g++-14
NVML=-L$(CUDA_HOME)/lib64/stubs -lnvidia-ml

# Arquiteturas CUDA (SASS nativo). Default cobre Maxwell(sm_52) -> Blackwell(sm_120).
# Para build rapido, sobrescreva SM com a arquitetura da sua GPU (veja acima).
SM?= -gencode=arch=compute_52,code=sm_52 \
     -gencode=arch=compute_61,code=sm_61 \
     -gencode=arch=compute_75,code=sm_75 \
     -gencode=arch=compute_86,code=sm_86 \
     -gencode=arch=compute_89,code=sm_89 \
     -gencode=arch=compute_90,code=sm_90 \
     -gencode=arch=compute_120,code=sm_120
NVCCFLAGS=-O3 -ccbin $(CCBIN) -std=c++14 $(SM) -Wno-deprecated-gpu-targets

# BATCHM = chaves por lote na inversao de Montgomery (minikey_gpu_opt).
BATCHM?=8

# Fontes CPU reusadas do BitCrack (compiladas direto; sem .a, p/ reprodutibilidade).
CPU_SRC=$(BC)/secp256k1lib/secp256k1.cpp \
        $(BC)/CryptoUtil/Rng.cpp $(BC)/CryptoUtil/checksum.cpp \
        $(BC)/CryptoUtil/ripemd160.cpp $(BC)/CryptoUtil/sha256.cpp \
        $(BC)/AddressUtil/Base58.cpp $(BC)/AddressUtil/hash.cpp \
        $(BC)/util/util.cpp $(BC)/Logger/Logger.cpp

all: minikey minikey_gpu minikey_gpu_full minikey_gpu_opt BitcrackM

minikey: src/minikey.cpp $(CPU_SRC)
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^

minikey_gpu: src/minikey_gpu.cu
	$(NVCC) $(NVCCFLAGS) -o $@ src/minikey_gpu.cu

minikey_gpu_full: src/minikey_gpu_full.cu
	$(NVCC) $(NVCCFLAGS) -I$(BC)/cudaMath -o $@ src/minikey_gpu_full.cu $(NVML)

minikey_gpu_opt: src/minikey_gpu_opt.cu
	$(NVCC) $(NVCCFLAGS) -DBATCHM=$(BATCHM) -I$(BC)/cudaMath -o $@ src/minikey_gpu_opt.cu $(NVML)

# BitcrackM: front-end CLI estilo BitCrack (busca contra lista de enderecos-alvo)
bitcrackm: BitcrackM
BitcrackM: src/BitcrackM.cu
	$(NVCC) $(NVCCFLAGS) -DBATCHM=$(BATCHM) -I$(BC)/cudaMath -o $@ src/BitcrackM.cu $(NVML)

clean:
	rm -f minikey minikey_gpu minikey_gpu_full minikey_gpu_opt BitcrackM

.PHONY: all clean
