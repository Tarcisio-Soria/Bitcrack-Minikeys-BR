// =============================================================================
//  minikey_gpu.cu  —  Fase 2a: kernel de TRIAGEM na GPU
//  Mede a taxa de "geração" de minikeys Casascius S1: para cada candidato,
//  calcula SHA256(minikey + '?') e testa o filtro 0x00 (o "atalho").
//
//  NÃO deriva chave pública (isso é a Fase 2b). Aqui medimos só a triagem,
//  que é a parte RÁPIDA — para contrastar depois com o pipeline completo.
//
//  Compila: make gpu     (nvcc, sm_86)   Roda: ./minikey_gpu [-b N -t N -p N]
// =============================================================================
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// Alfabeto base58 (sem 0 O I l)
__device__ __constant__ char d_B58[58] = {
    '1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','c','d','e','f','g','h','i','j','k','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
};

__device__ __constant__ unsigned int d_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ __forceinline__ unsigned int rotr(unsigned int x, int n) { return (x >> n) | (x << (32 - n)); }

// SHA256 de um único bloco (mensagem <= 55 bytes), bloco já com padding em w[16].
__device__ void sha256_block(const unsigned int w16[16], unsigned int h[8])
{
    unsigned int w[64];
#pragma unroll
    for (int i = 0; i < 16; i++) w[i] = w16[i];
#pragma unroll
    for (int i = 16; i < 64; i++) {
        unsigned int s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
        unsigned int s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    unsigned int a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,
                 e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,hh=0x5be0cd19;
#pragma unroll
    for (int i = 0; i < 64; i++) {
        unsigned int S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        unsigned int ch = (e & f) ^ (~e & g);
        unsigned int t1 = hh + S1 + ch + d_K[i] + w[i];
        unsigned int S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        unsigned int maj = (a & b) ^ (a & c) ^ (b & c);
        unsigned int t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]=0x6a09e667+a; h[1]=0xbb67ae85+b; h[2]=0x3c6ef372+c; h[3]=0xa54ff53a+d;
    h[4]=0x510e527f+e; h[5]=0x9b05688c+f; h[6]=0x1f83d9ab+g; h[7]=0x5be0cd19+hh;
}

// xorshift64 para espalhar os índices iniciais por thread
__device__ __forceinline__ uint64_t xs64(uint64_t &s) { s^=s<<13; s^=s>>7; s^=s<<17; return s; }

// Kernel de triagem: cada thread testa 'ppt' candidatos de 22 chars (S + 21).
__global__ void triage(uint64_t seed, int ppt, unsigned long long *validCount)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char idx[21];
    uint64_t s = seed ^ ((uint64_t)gid * 0x9E3779B97F4A7C15ULL);
    for (int i = 0; i < 21; i++) idx[i] = (unsigned char)(xs64(s) % 58);

    unsigned long long localValid = 0;

    for (int p = 0; p < ppt; p++) {
        // monta a mensagem "S" + 21 chars + "?"  (23 bytes) no bloco w[16]
        unsigned int w[16];
#pragma unroll
        for (int i = 0; i < 16; i++) w[i] = 0;
        unsigned char m[23];
        m[0] = 'S';
        for (int i = 0; i < 21; i++) m[i+1] = (unsigned char)d_B58[idx[i]];
        m[22] = '?';
#pragma unroll
        for (int i = 0; i < 23; i++) w[i >> 2] |= (unsigned int)m[i] << (24 - 8*(i & 3));
        w[23 >> 2] |= 0x80u << (24 - 8*(23 & 3));   // padding
        w[15] = 23 * 8;                              // comprimento em bits

        unsigned int h[8];
        sha256_block(w, h);
        if ((h[0] >> 24) == 0) localValid++;        // filtro 0x00

        // incrementa o "odômetro" base58 de 21 dígitos
        int cpos = 0;
        while (cpos < 21) { if (++idx[cpos] >= 58) { idx[cpos] = 0; cpos++; } else break; }
    }
    atomicAdd(validCount, localValid);
}

int main(int argc, char **argv)
{
    int blocks = 240, threads = 256, ppt = 2048;
    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "-b")) blocks  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t")) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-p")) ppt     = atoi(argv[++i]);
    }

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s  (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("Config: %d blocos x %d threads x %d pontos = %.1f M candidatos\n",
           blocks, threads, ppt, (double)blocks*threads*ppt/1e6);

    unsigned long long *d_valid, h_valid = 0;
    cudaMalloc(&d_valid, sizeof(unsigned long long));
    cudaMemset(d_valid, 0, sizeof(unsigned long long));

    uint64_t seed = (uint64_t)time(NULL) * 0x100000001B3ULL ^ (uint64_t)clock();

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    triage<<<blocks, threads>>>(seed, ppt, d_valid);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("ERRO CUDA: %s\n", cudaGetErrorString(err)); return 1; }

    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaMemcpy(&h_valid, d_valid, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    unsigned long long total = (unsigned long long)blocks * threads * ppt;
    double sec = ms / 1000.0;
    printf("\n--- TRIAGEM (SHA256 + filtro 0x00) ---\n");
    printf("  tempo:       %.3f s\n", sec);
    printf("  candidatos:  %llu\n", total);
    printf("  válidas:     %llu   (1 em %.1f = %.4f%%)\n",
           h_valid, h_valid ? (double)total/h_valid : 0.0, 100.0*h_valid/total);
    printf("  TAXA:        %.1f M cand/s\n", total/sec/1e6);
    printf("\n  (apenas a triagem; a derivação EC completa vem na Fase 2b)\n");

    cudaFree(d_valid);
    return 0;
}
