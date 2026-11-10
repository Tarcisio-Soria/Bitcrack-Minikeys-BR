// =============================================================================
//  BitcrackM  —  front-end CLI (estilo BitCrack) do Bitcrack-Minikeys-BR
//
//  Carrega uma lista de ENDERECOS-alvo, busca minikeys Casascius S1 na GPU e,
//  quando o endereco derivado bate com um da lista, grava a chave encontrada.
//
//  Uso:
//    BitcrackM -list <enderecos.txt> [-s RESULTS.txt] [-b N -t N -p N]
//              [-reduce <minikey22> <K>] [--naive]
//
//    -list <arq>        enderecos-alvo Base58 (1 por linha; aceita arquivos com
//                       varias colunas - qualquer token que seja um P2PKH valido)
//    -s / -o <arq>      salva "minikey  privkey  endereco" ao achar (append)
//    -b -t -p           blocos / threads / candidatos-por-thread
//    -reduce <mk> <K>   DEMONSTRACAO: fixa (21-K) chars da minikey conhecida e
//                       bruteforca os K ultimos -> espaco pequeno, ACHA e grava
//    --naive            usa a mult. de ponto ingenua (sem comb/lote), p/ comparar
//    (sem -reduce)      busca COMPLETA no espaco 2^115 -> prova a inviabilidade
//
//  Pipeline otimizado: comb de base fixa + inversao em lote de Montgomery.
//  Compila: make bitcrackm   (BATCHM = chaves/lote, default 8)
// =============================================================================
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <utility>
#include <algorithm>
#include <csignal>
#include <ctime>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include "secp256k1.cuh"

#ifndef BATCHM
#define BATCHM 8
#endif

// =============================== SHA256 ======================================
__device__ __constant__ unsigned int K256[64] = {
 0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
 0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
 0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
 0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
 0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
 0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
 0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
 0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ __forceinline__ unsigned int ror(unsigned int x,int n){return (x>>n)|(x<<(32-n));}

__device__ void sha256_compress(const unsigned int blk[16], unsigned int h[8]){
    unsigned int w[64];
#pragma unroll
    for(int i=0;i<16;i++) w[i]=blk[i];
#pragma unroll
    for(int i=16;i<64;i++){
        unsigned int s0=ror(w[i-15],7)^ror(w[i-15],18)^(w[i-15]>>3);
        unsigned int s1=ror(w[i-2],17)^ror(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    unsigned int a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
#pragma unroll
    for(int i=0;i<64;i++){
        unsigned int S1=ror(e,6)^ror(e,11)^ror(e,25);
        unsigned int ch=(e&f)^(~e&g);
        unsigned int t1=hh+S1+ch+K256[i]+w[i];
        unsigned int S0=ror(a,2)^ror(a,13)^ror(a,22);
        unsigned int maj=(a&b)^(a&c)^(b&c);
        unsigned int t2=S0+maj;
        hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
}
__device__ void sha256(const unsigned char *d,int len,unsigned int h[8]){
    h[0]=0x6a09e667;h[1]=0xbb67ae85;h[2]=0x3c6ef372;h[3]=0xa54ff53a;
    h[4]=0x510e527f;h[5]=0x9b05688c;h[6]=0x1f83d9ab;h[7]=0x5be0cd19;
    unsigned char buf[128]; int total=len;
    for(int i=0;i<len;i++) buf[i]=d[i];
    int padlen=(len%64<56)?((len/64)*64+64):((len/64)*64+128);
    buf[len]=0x80; for(int i=len+1;i<padlen;i++) buf[i]=0;
    uint64_t bits=(uint64_t)total*8;
    for(int i=0;i<8;i++) buf[padlen-1-i]=(unsigned char)(bits>>(8*i));
    for(int off=0;off<padlen;off+=64){
        unsigned int blk[16];
#pragma unroll
        for(int i=0;i<16;i++)
            blk[i]=(buf[off+4*i]<<24)|(buf[off+4*i+1]<<16)|(buf[off+4*i+2]<<8)|buf[off+4*i+3];
        sha256_compress(blk,h);
    }
}

// ============================= RIPEMD160 =====================================
__device__ __constant__ int RL[80]={
 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
 3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
 4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
__device__ __constant__ int RR[80]={
 5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
 15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
 12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
__device__ __constant__ int SL[80]={
 11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
 11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
 9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
__device__ __constant__ int SR[80]={
 8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
 9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
 8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
__device__ __forceinline__ unsigned int rol(unsigned int x,int n){return (x<<n)|(x>>(32-n));}
__device__ void ripemd160(const unsigned char *d, int len, unsigned int out[5]){
    unsigned int X[16]; for(int i=0;i<16;i++) X[i]=0;
    for(int i=0;i<len;i++) X[i>>2] |= (unsigned int)d[i] << (8*(i&3));
    X[len>>2] |= 0x80u << (8*(len&3));
    X[14] = (unsigned int)(len*8);
    unsigned int al=0x67452301,bl=0xefcdab89,cl=0x98badcfe,dl=0x10325476,el=0xc3d2e1f0;
    unsigned int ar=al,br=bl,cr=cl,dr=dl,er=el;
    for(int j=0;j<80;j++){
        int rnd=j/16; unsigned int fl,fr,kl,kr;
        if(rnd==0){fl=bl^cl^dl;kl=0x00000000;} else if(rnd==1){fl=(bl&cl)|(~bl&dl);kl=0x5a827999;}
        else if(rnd==2){fl=(bl|~cl)^dl;kl=0x6ed9eba1;} else if(rnd==3){fl=(bl&dl)|(cl&~dl);kl=0x8f1bbcdc;}
        else {fl=bl^(cl|~dl);kl=0xa953fd4e;}
        if(rnd==0){fr=br^(cr|~dr);kr=0x50a28be6;} else if(rnd==1){fr=(br&dr)|(cr&~dr);kr=0x5c4dd124;}
        else if(rnd==2){fr=(br|~cr)^dr;kr=0x6d703ef3;} else if(rnd==3){fr=(br&cr)|(~br&dr);kr=0x7a6d76e9;}
        else {fr=br^cr^dr;kr=0x00000000;}
        unsigned int t=rol(al+fl+X[RL[j]]+kl,SL[j])+el;
        al=el;el=dl;dl=rol(cl,10);cl=bl;bl=t;
        t=rol(ar+fr+X[RR[j]]+kr,SR[j])+er;
        ar=er;er=dr;dr=rol(cr,10);cr=br;br=t;
    }
    unsigned int h0=0x67452301,h1=0xefcdab89,h2=0x98badcfe,h3=0x10325476,h4=0xc3d2e1f0;
    out[0]=h1+cl+dr; out[1]=h2+dl+er; out[2]=h3+el+ar; out[3]=h4+al+br; out[4]=h0+bl+cr;
}
__device__ void hash160(const unsigned int x[8], const unsigned int y[8], unsigned int out[5]){
    unsigned char pub[65]; pub[0]=0x04;
    for(int i=0;i<8;i++){ pub[1+4*i]=x[i]>>24; pub[2+4*i]=x[i]>>16; pub[3+4*i]=x[i]>>8; pub[4+4*i]=x[i]; }
    for(int i=0;i<8;i++){ pub[33+4*i]=y[i]>>24; pub[34+4*i]=y[i]>>16; pub[35+4*i]=y[i]>>8; pub[36+4*i]=y[i]; }
    unsigned int sh[8]; sha256(pub,65,sh);
    unsigned char shb[32];
    for(int i=0;i<8;i++){ shb[4*i]=sh[i]>>24; shb[4*i+1]=sh[i]>>16; shb[4*i+2]=sh[i]>>8; shb[4*i+3]=sh[i]; }
    ripemd160(shb,32,out);
}

// ============================= EC (Jacobiano) ================================
__device__ __forceinline__ bool isZeroF(const unsigned int a[8]){ unsigned int z=0; for(int i=0;i<8;i++) z|=a[i]; return z==0; }
__device__ __forceinline__ void setField(unsigned int a[8], const unsigned int *s){ for(int i=0;i<8;i++) a[i]=s[i]; }

__device__ void jacDouble(const unsigned int X[8],const unsigned int Y[8],const unsigned int Z[8],
                          unsigned int X3[8],unsigned int Y3[8],unsigned int Z3[8]){
    if(isZeroF(Z)){ for(int i=0;i<8;i++){X3[i]=0;Y3[i]=0;Z3[i]=0;} return; }
    unsigned int A[8],B[8],C[8],D[8],E[8],F[8],t1[8],t2[8],t3[8];
    squareModP(X,A); squareModP(Y,B); squareModP(B,C);
    addModP(X,B,t1); squareModP(t1,t2); subModP(t2,A,t3); subModP(t3,C,t1); addModP(t1,t1,D);
    addModP(A,A,t1); addModP(t1,A,E); squareModP(E,F);
    addModP(D,D,t1); subModP(F,t1,X3);
    subModP(D,X3,t1); mulModP(E,t1,t2);
    addModP(C,C,t1); addModP(t1,t1,t3); addModP(t3,t3,t1); subModP(t2,t1,Y3);
    mulModP(Y,Z,t1); addModP(t1,t1,Z3);
}
__device__ void jacAddAffine(const unsigned int X1[8],const unsigned int Y1[8],const unsigned int Z1[8],
                             const unsigned int x2[8],const unsigned int y2[8],
                             unsigned int X3[8],unsigned int Y3[8],unsigned int Z3[8]){
    if(isZeroF(Z1)){ setField(X3,x2); setField(Y3,y2); for(int i=0;i<8;i++) Z3[i]=0; Z3[7]=1; return; }
    unsigned int Z1Z1[8],U2[8],S2[8],H[8],HH[8],I[8],J[8],r[8],V[8],t1[8],t2[8],t3[8];
    squareModP(Z1,Z1Z1); mulModP(x2,Z1Z1,U2); mulModP(Z1,Z1Z1,t1); mulModP(y2,t1,S2);
    subModP(U2,X1,H); squareModP(H,HH); addModP(HH,HH,t1); addModP(t1,t1,I); mulModP(H,I,J);
    subModP(S2,Y1,t1); addModP(t1,t1,r); mulModP(X1,I,V);
    squareModP(r,t1); subModP(t1,J,t2); addModP(V,V,t3); subModP(t2,t3,X3);
    subModP(V,X3,t1); mulModP(r,t1,t2); mulModP(Y1,J,t1); addModP(t1,t1,t3); subModP(t2,t3,Y3);
    mulModP(Z1,H,t1); addModP(t1,t1,Z3);
}
__device__ void scalarMulG_naive(const unsigned int k[8], unsigned int outX[8], unsigned int outY[8]){
    unsigned int X[8],Y[8],Z[8]; for(int i=0;i<8;i++){X[i]=0;Y[i]=0;Z[i]=0;}
    unsigned int nX[8],nY[8],nZ[8];
    for(int wi=0; wi<8; wi++){ unsigned int wv=k[wi];
        for(int bp=31; bp>=0; bp--){
            jacDouble(X,Y,Z,nX,nY,nZ); setField(X,nX);setField(Y,nY);setField(Z,nZ);
            if((wv>>bp)&1){ jacAddAffine(X,Y,Z,_GX,_GY,nX,nY,nZ); setField(X,nX);setField(Y,nY);setField(Z,nZ); } } }
    unsigned int Zi[8],Zi2[8],Zi3[8]; invModP(Z,Zi); squareModP(Zi,Zi2); mulModP(Zi2,Zi,Zi3);
    mulModP(X,Zi2,outX); mulModP(Y,Zi3,outY);
}

// comb de base fixa parametrizado por WBITS (largura da janela). Tabela dinamica
// g_comb[(i*NDIG+j)*16], gerada na GPU pela rotina ingenua (bit-exata). Default
// WBITS=10 = ponto otimo medido na RTX 3060 (ver estudo no minikey_gpu_opt).
#ifndef WBITS
#define WBITS 10
#endif
#define NWIN  ((256 + WBITS - 1) / WBITS)
#define NDIG  (1u << WBITS)
#define WMASK (NDIG - 1u)
__device__ unsigned int *g_comb;
__device__ __forceinline__ unsigned int combDigit(const unsigned int k[8], int p){
    int lw=p>>5, bit=p&31;
    unsigned long long lo=k[7-lw];
    unsigned long long hi=(lw+1<8)?(unsigned long long)k[7-lw-1]:0ull;
    return (unsigned int)(((lo|(hi<<32))>>bit)&WMASK);
}
__global__ void kBuildComb(){
    int i=blockIdx.x; unsigned int j=blockIdx.y*blockDim.x+threadIdx.x;
    if(j>=NDIG) return;
    unsigned int *T=g_comb+(((size_t)i*NDIG+j)*16);
    if(j==0){ for(int t=0;t<16;t++) T[t]=0; return; }
    int base=WBITS*i, lw=base>>5, bit=base&31;
    unsigned long long val=((unsigned long long)j)<<bit;
    unsigned int w0=(unsigned int)val, w1=(unsigned int)(val>>32);
    unsigned int k[8]; for(int t=0;t<8;t++) k[t]=0; bool ovf=false;
    if(lw<8) k[7-lw]=w0; else if(w0) ovf=true;
    if(w1){ if(lw+1<8) k[7-lw-1]=w1; else ovf=true; }
    if(ovf){ for(int t=0;t<16;t++) T[t]=0; return; }
    unsigned int x[8],y[8]; scalarMulG_naive(k,x,y); for(int t=0;t<8;t++){ T[t]=x[t]; T[8+t]=y[t]; }
}
__device__ void combJac(const unsigned int k[8], unsigned int X[8],unsigned int Y[8],unsigned int Z[8]){
    for(int i=0;i<8;i++){X[i]=0;Y[i]=0;Z[i]=0;} unsigned int nX[8],nY[8],nZ[8];
    for(int i=0;i<NWIN;i++){ unsigned int digit=combDigit(k,WBITS*i);
        if(digit){ const unsigned int *T=g_comb+(((size_t)i*NDIG+digit)*16);
            jacAddAffine(X,Y,Z,T,T+8,nX,nY,nZ); setField(X,nX);setField(Y,nY);setField(Z,nZ); } }
}
__device__ void scalarMulG_comb(const unsigned int k[8], unsigned int outX[8], unsigned int outY[8]){
    unsigned int X[8],Y[8],Z[8]; combJac(k,X,Y,Z);
    unsigned int Zi[8],Zi2[8],Zi3[8]; invModP(Z,Zi); squareModP(Zi,Zi2); mulModP(Zi2,Zi,Zi3);
    mulModP(X,Zi2,outX); mulModP(Y,Zi3,outY);
}
// hash160 de um lote (1 inversao de Montgomery). priv[m] -> outh[m].
__device__ void deriveBatchHash(const unsigned int priv[BATCHM][8], int cnt, unsigned int outh[BATCHM][5]){
    unsigned int X[BATCHM][8],Y[BATCHM][8],Z[BATCHM][8],pre[BATCHM][8];
    for(int m=0;m<cnt;m++) combJac(priv[m],X[m],Y[m],Z[m]);
    setField(pre[0],Z[0]); for(int m=1;m<cnt;m++) mulModP(pre[m-1],Z[m],pre[m]);
    unsigned int acc[8]; invModP(pre[cnt-1],acc);
    for(int m=cnt-1;m>=1;m--){ unsigned int zi[8],zi2[8],zi3[8],x[8],y[8];
        mulModP(acc,pre[m-1],zi); squareModP(zi,zi2); mulModP(zi2,zi,zi3);
        mulModP(X[m],zi2,x); mulModP(Y[m],zi3,y); hash160(x,y,outh[m]);
        unsigned int tacc[8]; mulModP(acc,Z[m],tacc); setField(acc,tacc); }   // sem aliasar saida=entrada
    { unsigned int zi2[8],zi3[8],x[8],y[8]; squareModP(acc,zi2); mulModP(zi2,acc,zi3);
      mulModP(X[0],zi2,x); mulModP(Y[0],zi3,y); hash160(x,y,outh[0]); }
}

// ===================== busca binaria no conjunto de alvos ====================
__device__ int cmp5(const unsigned int*a,const unsigned int*b){
    for(int i=0;i<5;i++){ if(a[i]<b[i])return -1; if(a[i]>b[i])return 1; } return 0; }
__device__ int targetFind(const unsigned int*h,const unsigned int*tg,int n){
    int lo=0,hi=n-1; while(lo<=hi){ int mid=(lo+hi)>>1; int c=cmp5(h,tg+(size_t)mid*5);
        if(c==0)return mid; if(c<0)hi=mid-1; else lo=mid+1; } return -1; }

__device__ __constant__ char dB58[58]={
 '1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
 'a','b','c','d','e','f','g','h','i','j','k','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
__device__ __forceinline__ uint64_t xs(uint64_t &s){s^=s<<13;s^=s>>7;s^=s<<17;return s;}

// resultado de um achado
struct Hit { int flag; unsigned char idx[21]; unsigned int priv[8]; int tgt; };

// FASE A: triagem -> buffer de validas
__global__ void kTriageCollect(uint64_t seed,int ppt, unsigned char *buf, unsigned int *cnt, unsigned int cap){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned char idx[21]; uint64_t s=seed^((uint64_t)gid*0x9E3779B97F4A7C15ULL);
    for(int i=0;i<21;i++) idx[i]=(unsigned char)(xs(s)%58);
    for(int p=0;p<ppt;p++){
        unsigned char mkq[23]; mkq[0]='S'; for(int i=0;i<21;i++) mkq[i+1]=dB58[idx[i]]; mkq[22]='?';
        unsigned int h[8]; sha256(mkq,23,h);
        if((h[0]>>24)==0){ unsigned int pos=atomicAdd(cnt,1u);
            if(pos<cap){ unsigned char *d=buf+(size_t)pos*21; for(int i=0;i<21;i++) d[i]=idx[i]; } }
        int c=0; while(c<21){ if(++idx[c]>=58){idx[c]=0;c++;} else break; }
    }
}
// FASE B: deriva as validas e compara com os alvos (comb + lote)
__global__ void kDeriveList(const unsigned char *buf, unsigned int n, const unsigned int *tg, int ntgt, Hit *hit){
    unsigned int stride=(unsigned int)gridDim.x*blockDim.x*BATCHM;
    for(unsigned int base=(blockIdx.x*blockDim.x+threadIdx.x)*BATCHM; base<n; base+=stride){
        if(hit->flag) return;
        int cnt=BATCHM; if(base+cnt>n) cnt=n-base;
        unsigned int priv[BATCHM][8];
        for(int m=0;m<cnt;m++){ const unsigned char *d=buf+(size_t)(base+m)*21;
            unsigned char mk[22]; mk[0]='S'; for(int j=0;j<21;j++) mk[j+1]=dB58[d[j]]; sha256(mk,22,priv[m]); }
        unsigned int oh[BATCHM][5]; deriveBatchHash(priv,cnt,oh);
        for(int m=0;m<cnt;m++){ int t=targetFind(oh[m],tg,ntgt);
            if(t>=0 && atomicCAS(&hit->flag,0,1)==0){ const unsigned char *d=buf+(size_t)(base+m)*21;
                for(int j=0;j<21;j++) hit->idx[j]=d[j]; for(int j=0;j<8;j++) hit->priv[j]=priv[m][j]; hit->tgt=t; } }
    }
}
// FASE B ingenua (--naive): 1 chave por vez, mult. de ponto double-and-add
__global__ void kDeriveListNaive(const unsigned char *buf, unsigned int n, const unsigned int *tg, int ntgt, Hit *hit){
    unsigned int stride=(unsigned int)gridDim.x*blockDim.x;
    for(unsigned int i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=stride){
        if(hit->flag) return; const unsigned char *d=buf+(size_t)i*21;
        unsigned char mk[22]; mk[0]='S'; for(int j=0;j<21;j++) mk[j+1]=dB58[d[j]];
        unsigned int priv[8]; sha256(mk,22,priv);
        unsigned int x[8],y[8],oh[5]; scalarMulG_naive(priv,x,y); hash160(x,y,oh);
        int t=targetFind(oh,tg,ntgt);
        if(t>=0 && atomicCAS(&hit->flag,0,1)==0){ for(int j=0;j<21;j++) hit->idx[j]=d[j];
            for(int j=0;j<8;j++) hit->priv[j]=priv[j]; hit->tgt=t; }
    }
}
// modo -reduce: fixa (21-K) chars conhecidos, bruteforca os K ultimos, compara c/ alvos
__device__ __constant__ unsigned char cKnownIdx[21];
// Varre TODO o espaco 58^K (sem early-exit, p/ tempo deterministico e resumo honesto);
// conta quantas passaram no filtro 0x00 (= chaves efetivamente TESTADAS na curva).
// -reduce em DUAS FASES (como a busca completa): FASE A triagem do bloco
// [startC, startC+count) com odometro base58 -> empurra as validas pro buffer;
// a derivacao densa fica com kDeriveList (lote). Evita a divergencia de warp.
__global__ void kReduceTriage(int K, unsigned long long startC, unsigned long long count,
                              unsigned char *buf, unsigned int *cnt, unsigned int cap){
    unsigned long long nth=(unsigned long long)gridDim.x*blockDim.x;
    unsigned long long tid=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    unsigned long long per=(count+nth-1)/nth, s=tid*per, e=s+per; if(e>count) e=count;
    if(s>=e) return;
    unsigned char idx[21]; for(int i=0;i<21;i++) idx[i]=cKnownIdx[i];
    unsigned long long t=startC+s; for(int j=0;j<K;j++){ idx[21-K+j]=(unsigned char)(t%58); t/=58; }
    for(unsigned long long c=s;c<e;c++){
        unsigned char mkq[23]; mkq[0]='S'; for(int i=0;i<21;i++) mkq[i+1]=dB58[idx[i]]; mkq[22]='?';
        unsigned int h[8]; sha256(mkq,23,h);
        if((h[0]>>24)==0){ unsigned int pos=atomicAdd(cnt,1u);
            if(pos<cap){ unsigned char *d=buf+(size_t)pos*21; for(int i=0;i<21;i++) d[i]=idx[i]; } }
        int p=21-K; while(p<21){ if(++idx[p]>=58){ idx[p]=0; p++; } else break; }  // odometro +1
    }
}

// ============================== host: utilidades =============================
static const char *HB58="123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static int b58idx(char c){ for(int i=0;i<58;i++) if(HB58[i]==c) return i; return -1; }

// decodifica um endereco Base58Check P2PKH (versao 0x00) -> hash160 em 5 words LE.
static bool addr2hash160(const std::string &s, unsigned int out[5]){
    if(s.size()<26||s.size()>35) return false;
    std::vector<unsigned char> b; b.push_back(0);
    for(char c: s){ int v=b58idx(c); if(v<0) return false; int carry=v;
        for(size_t i=0;i<b.size();i++){ carry+=(int)b[i]*58; b[i]=carry&0xff; carry>>=8; }
        while(carry){ b.push_back(carry&0xff); carry>>=8; } }
    for(char c: s){ if(c=='1') b.push_back(0); else break; }
    std::reverse(b.begin(),b.end());
    if(b.size()!=25 || b[0]!=0x00) return false;
    // hash160 = b[1..20]; word i (LE) = b[1+4i] | b[1+4i+1]<<8 | ...
    for(int i=0;i<5;i++) out[i]=(unsigned int)b[1+4*i]|((unsigned int)b[1+4*i+1]<<8)|((unsigned int)b[1+4*i+2]<<16)|((unsigned int)b[1+4*i+3]<<24);
    return true;
}

static volatile sig_atomic_t g_stop=0; static void onint(int){ g_stop=1; }
static nvmlDevice_t g_nd; static bool g_nv=false;
static void nvInit(){ if(nvmlInit()==NVML_SUCCESS && nvmlDeviceGetHandleByIndex(0,&g_nd)==NVML_SUCCESS) g_nv=true; }
static void nvRead(unsigned int&t,unsigned int&c){ t=0;c=0; if(g_nv){ nvmlDeviceGetTemperature(g_nd,NVML_TEMPERATURE_GPU,&t); nvmlDeviceGetClockInfo(g_nd,NVML_CLOCK_SM,&c);} }
// cores ANSI (viram "" com --plain ou saida nao-TTY)
static const char *CR="\033[0m",*CB="\033[1m",*CD="\033[2m",*Cg="\033[32m",*Cy="\033[33m",
                  *Cb="\033[38;5;39m",*Cc="\033[36m",*Co="\033[38;5;208m",*Ce="\033[38;5;214m",*Cr="\033[31m";
static bool g_plain=false;
static void clearColors(){ CR=CB=CD=Cg=Cy=Cb=Cc=Co=Ce=Cr=""; }
static const char* tcol(unsigned int t){ if(g_plain) return ""; return t>=85?Cr:(t>=75?Cy:Cg); }
static std::string rateStr(double r){ char b[48]; if(r>=1e9) sprintf(b,"%.2f G",r/1e9); else if(r>=1e6) sprintf(b,"%.2f M",r/1e6); else if(r>=1e3) sprintf(b,"%.1f k",r/1e3); else sprintf(b,"%.0f ",r); return b; }
static std::string thou(unsigned long long v){ std::string s=std::to_string(v),o; int c=0; for(int i=(int)s.size()-1;i>=0;i--){ o=std::string(1,s[i])+o; if(++c%3==0&&i>0) o=","+o; } return o; }

static void usage(){
    printf("BitcrackM — busca de minikeys Casascius S1 (Bitcrack-Minikeys-BR)\n");
    printf("uso: BitcrackM -list <enderecos.txt> [-s RESULTS.txt] [-b N -t N -p N]\n");
    printf("               [-reduce <minikey22> <K>] [--naive] [--plain]\n");
    printf("  --plain: sem cores/painel (auto quando a saida nao e um terminal)\n");
}

int main(int argc,char**argv){
    const char *listPath=nullptr,*savePath=nullptr,*reduceMk=nullptr; int reduceK=0;
    int blk=120,thr=256,ppt=2048; bool naive=false;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-list")&&i+1<argc) listPath=argv[++i];
        else if((!strcmp(argv[i],"-s")||!strcmp(argv[i],"-o"))&&i+1<argc) savePath=argv[++i];
        else if(!strcmp(argv[i],"-b")&&i+1<argc) blk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-t")&&i+1<argc) thr=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-p")&&i+1<argc) ppt=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--naive")) naive=true;
        else if(!strcmp(argv[i],"--plain")) g_plain=true;
        else if(!strcmp(argv[i],"-reduce")&&i+2<argc){ reduceMk=argv[++i]; reduceK=atoi(argv[++i]); }
        else if(!strcmp(argv[i],"-h")||!strcmp(argv[i],"--help")){ usage(); return 0; }
    }
    if(!listPath){ usage(); return 1; }
    if(!isatty(1)) g_plain=true;          // saida redirecionada (tee/arquivo) -> sem ANSI
    if(g_plain) clearColors();

    // carrega os alvos
    FILE *f=fopen(listPath,"r"); if(!f){ printf("nao abri a lista: %s\n",listPath); return 1; }
    std::vector<std::pair<std::array<unsigned int,5>,std::string>> tgts; char line[512];
    while(fgets(line,sizeof(line),f)){
        char *tok=strtok(line," \t\r\n");
        while(tok){ std::array<unsigned int,5> h; if(addr2hash160(tok,h.data())) tgts.push_back({h,std::string(tok)}); tok=strtok(nullptr," \t\r\n"); }
    }
    fclose(f);
    if(tgts.empty()){ printf("nenhum endereco P2PKH valido em %s\n",listPath); return 1; }
    std::sort(tgts.begin(),tgts.end(),[](const auto&a,const auto&b){
        for(int i=0;i<5;i++){ if(a.first[i]<b.first[i])return true; if(a.first[i]>b.first[i])return false; } return false; });
    int ntgt=(int)tgts.size();
    std::vector<unsigned int> flat(ntgt*5); std::vector<std::string> addrs(ntgt);
    for(int i=0;i<ntgt;i++){ for(int j=0;j<5;j++) flat[i*5+j]=tgts[i].first[j]; addrs[i]=tgts[i].second; }

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
    printf("=== BitcrackM · Bitcrack-Minikeys-BR ===\n");
    printf("GPU: %s   alvos carregados: %d   pipeline: %s (WBITS=%d)%s\n",prop.name,ntgt,
           naive?"INGENUO (double-and-add)":"OTIMIZADO (comb + lote)", WBITS, savePath?"":"   (sem -s: nao grava)");
    { size_t bytes=(size_t)NWIN*NDIG*16*sizeof(unsigned int);
      unsigned int *d_ptr; cudaError_t ce=cudaMalloc(&d_ptr,bytes);
      if(ce){ printf("ERRO cudaMalloc tabela comb (%zu MB): %s\n",bytes>>20,cudaGetErrorString(ce)); return 1; }
      cudaMemcpyToSymbol(g_comb,&d_ptr,sizeof(d_ptr));
      dim3 grid(NWIN,(NDIG+255)/256); kBuildComb<<<grid,256>>>(); cudaDeviceSynchronize(); }
    cudaError_t e=cudaGetLastError(); if(e){ printf("ERRO build comb: %s\n",cudaGetErrorString(e)); return 1; }

    unsigned int *d_tg; cudaMalloc(&d_tg,flat.size()*sizeof(unsigned int));
    cudaMemcpy(d_tg,flat.data(),flat.size()*sizeof(unsigned int),cudaMemcpyHostToDevice);
    Hit *d_hit; cudaMalloc(&d_hit,sizeof(Hit)); cudaMemset(d_hit,0,sizeof(Hit));
    Hit h_hit; memset(&h_hit,0,sizeof(Hit));
    signal(SIGINT,onint); nvInit();

    auto report=[&](bool found){
        printf("\n");
        if(found){
            char mk[23]; mk[0]='S'; for(int i=0;i<21;i++) mk[i+1]=HB58[h_hit.idx[i]]; mk[22]=0;
            char priv[65]; for(int i=0;i<8;i++) sprintf(priv+8*i,"%08x",h_hit.priv[i]); priv[64]=0;
            const std::string &addr=addrs[h_hit.tgt];
            printf("%s▸ ACHOU!%s\n",Cg,CR);
            printf("   minikey : %s%s%s\n",Ce,mk,CR);
            printf("   privkey : %s%s%s\n",Co,priv,CR);
            printf("   endereco: %s%s%s\n",Cb,addr.c_str(),CR);
            if(savePath){ FILE *o=fopen(savePath,"a");
                if(o){ fprintf(o,"%s  %s  %s\n",mk,priv,addr.c_str()); fclose(o);
                       printf("   %s✓ gravado em %s%s\n",Cg,savePath,CR); }
                else printf("   %s(nao consegui gravar em %s)%s\n",Cr,savePath,CR); }
        } else {
            printf("%s[parado]%s nenhuma chave encontrada — exatamente o esperado para 2¹¹⁵.\n",Cy,CR);
        }
    };

    if(reduceMk){
        if(strlen(reduceMk)!=22||reduceMk[0]!='S'){ printf("minikey do -reduce deve ter 22 chars e comecar com S\n"); return 1; }
        unsigned char hIdx[21]; for(int i=0;i<21;i++){ int v=b58idx(reduceMk[1+i]); if(v<0){printf("char invalido na minikey\n");return 1;} hIdx[i]=(unsigned char)v; }
        cudaMemcpyToSymbol(cKnownIdx,hIdx,21);
        unsigned long long total=1; for(int i=0;i<reduceK;i++) total*=58; double bits=reduceK*5.8579;
        printf("modo -reduce (duas fases, varredura completa): fixando %d chars de %s, buscando os %d ultimos\n",22-reduceK,reduceMk,reduceK);
        printf("espaco: 58^%d = %s candidatos (~%.1f bits)\n",reduceK,thou(total).c_str(),bits);
        signal(SIGINT,onint); nvInit();
        const unsigned int CAP=2000000; const unsigned long long CHUNK=300000000ULL;  // ~1,17M validas/chunk < CAP
        unsigned char *d_buf; unsigned int *d_cnt; cudaMalloc(&d_buf,(size_t)CAP*21); cudaMalloc(&d_cnt,4);
        unsigned long long swept=0, tested=0; double cRate=0,tRate=0;
        struct timespec ts0; clock_gettime(CLOCK_MONOTONIC,&ts0);
        while(swept<total && !g_stop){
            unsigned long long cnt=(total-swept<CHUNK)?(total-swept):CHUNK;
            cudaMemset(d_cnt,0,4);
            cudaEvent_t a0,a1; cudaEventCreate(&a0); cudaEventCreate(&a1);
            cudaEventRecord(a0); kReduceTriage<<<blk,thr>>>(reduceK,swept,cnt,d_buf,d_cnt,CAP); cudaEventRecord(a1); cudaEventSynchronize(a1);
            float msA=0; cudaEventElapsedTime(&msA,a0,a1); cudaEventDestroy(a0); cudaEventDestroy(a1);
            unsigned int nval=0; cudaMemcpy(&nval,d_cnt,4,cudaMemcpyDeviceToHost); if(nval>CAP) nval=CAP;
            cudaEvent_t b0,b1; cudaEventCreate(&b0); cudaEventCreate(&b1);
            cudaEventRecord(b0); kDeriveList<<<blk,thr>>>(d_buf,nval,d_tg,ntgt,d_hit); cudaEventRecord(b1); cudaEventSynchronize(b1);
            float msB=0; cudaEventElapsedTime(&msB,b0,b1); cudaEventDestroy(b0); cudaEventDestroy(b1);
            swept+=cnt; tested+=nval; (void)msA; (void)msB;
            cudaMemcpy(&h_hit,d_hit,sizeof(Hit),cudaMemcpyDeviceToHost);
            struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
            double elapsed=(ts.tv_sec-ts0.tv_sec)+(ts.tv_nsec-ts0.tv_nsec)/1e9; long el=(long)elapsed;
            cRate=elapsed>0?swept/elapsed:0; tRate=elapsed>0?tested/elapsed:0;   // taxas cumulativas (estaveis)
            double pct=100.0*swept/total; unsigned int temp=0,clk=0; nvRead(temp,clk);
            if(g_plain) printf("[%02ld:%02ld] %.1f%%  triadas=%s  testadas=%s  triagem=%sH/s  teste=%s keys/s\n",
                               el/60,el%60,pct,thou(swept).c_str(),thou(tested).c_str(),rateStr(cRate).c_str(),rateStr(tRate).c_str());
            else { printf("\r\033[K%s[%02ld:%02ld]%s %5.1f%%  %s│%s triagem %s%sH/s%s  teste %s%s keys/s%s  %s│%s testadas %s%s%s",
                          CD,el/60,el%60,CR,pct,CD,CR,Cg,rateStr(cRate).c_str(),CR,Cg,rateStr(tRate).c_str(),CR,CD,CR,CB,thou(tested).c_str(),CR);
                   if(g_nv) printf("  %s│%s %s%u°C%s %s%u MHz%s",CD,CR,tcol(temp),temp,CR,CB,clk,CR);
                   fflush(stdout); }
        }
        cudaMemcpy(&h_hit,d_hit,sizeof(Hit),cudaMemcpyDeviceToHost);
        struct timespec tsf; clock_gettime(CLOCK_MONOTONIC,&tsf);
        double sec=(tsf.tv_sec-ts0.tv_sec)+(tsf.tv_nsec-ts0.tv_nsec)/1e9;
        report(h_hit.flag!=0);
        printf("\n%s--- resumo ---%s\n",CB,CR);
        printf("  tempo:     %s%.2f s%s%s\n",CB,sec,CR,g_stop?"  (interrompido)":"");
        printf("  triadas:   %s%s%s candidatos   (%s%sH/s%s)\n",CB,thou(swept).c_str(),CR,Cg,rateStr(sec>0?swept/sec:0).c_str(),CR);
        printf("  testadas:  %s%s%s chaves (passaram 0x00, ~1/256)   (%s%s keys/s%s)\n",CB,thou(tested).c_str(),CR,Cg,rateStr(sec>0?tested/sec:0).c_str(),CR);
        cudaFree(d_buf); cudaFree(d_cnt); cudaFree(d_tg); cudaFree(d_hit); return 0;
    }

    // busca completa (prova 2^115)
    const unsigned int CAP=2000000; unsigned char *d_buf; unsigned int *d_cnt;
    cudaMalloc(&d_buf,(size_t)CAP*21); cudaMalloc(&d_cnt,4);
    unsigned long long totC=0,totT=0; double candRate=0,testRate=0;
    uint64_t seed=(uint64_t)time(NULL)*0x9E3779B97F4A7C15ULL;
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC,&ts0);
    const int PL=12;
    if(g_plain) printf("busca COMPLETA no espaco 2^115 — Ctrl+C para parar\n");
    else { printf("%sbusca COMPLETA no espaco 2^115 — Ctrl+C para parar%s\n",CD,CR); printf("\033[?25l"); for(int i=0;i<PL;i++) printf("\n"); }
    while(!g_stop && !h_hit.flag){
        seed=seed*6364136223846793005ULL+1442695040888963407ULL;
        cudaMemset(d_cnt,0,4);
        cudaEvent_t a0,a1; cudaEventCreate(&a0); cudaEventCreate(&a1);
        cudaEventRecord(a0); kTriageCollect<<<blk,thr>>>(seed,ppt,d_buf,d_cnt,CAP); cudaEventRecord(a1); cudaEventSynchronize(a1);
        float msA=0; cudaEventElapsedTime(&msA,a0,a1); cudaEventDestroy(a0); cudaEventDestroy(a1);
        unsigned int nval=0; cudaMemcpy(&nval,d_cnt,4,cudaMemcpyDeviceToHost); if(nval>CAP) nval=CAP;
        unsigned long long candA=(unsigned long long)blk*thr*ppt;
        cudaEvent_t b0,b1; cudaEventCreate(&b0); cudaEventCreate(&b1); cudaEventRecord(b0);
        if(naive) kDeriveListNaive<<<blk,thr>>>(d_buf,nval,d_tg,ntgt,d_hit);
        else      kDeriveList<<<blk,thr>>>(d_buf,nval,d_tg,ntgt,d_hit);
        cudaEventRecord(b1); cudaEventSynchronize(b1);
        float msB=0; cudaEventElapsedTime(&msB,b0,b1); cudaEventDestroy(b0); cudaEventDestroy(b1);
        totC+=candA; totT+=nval; if(msA>0) candRate=candA/(msA/1000.0); if(msB>0) testRate=nval/(msB/1000.0);
        cudaMemcpy(&h_hit,d_hit,sizeof(Hit),cudaMemcpyDeviceToHost);
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); long el=(long)(ts.tv_sec-ts0.tv_sec);
        unsigned int temp=0,clk=0; nvRead(temp,clk);
        if(g_plain){
            printf("\r[%02ld:%02ld:%02ld] triagem %sH/s · teste %s keys/s · %s chaves",
                   el/3600,(el%3600)/60,el%60,rateStr(candRate).c_str(),rateStr(testRate).c_str(),thou(totT).c_str());
            fflush(stdout); continue;
        }
        char tele[160]; if(g_nv) snprintf(tele,sizeof tele,"%s%u°C%s %s%u MHz%s",tcol(temp),temp,CR,CB,clk,CR); else snprintf(tele,sizeof tele,"%s-- °C  -- MHz%s",CD,CR);
        printf("\033[%dA",PL);
        printf("\r\033[K%s══════════ %s%sBitcrackM · Bitcrack-Minikeys-BR%s %s══════════%s\n",Cc,CR,CB,CR,Cc,CR);
        printf("\r\033[K  %s%s%s  %s│%s  %s  %s│%s  alvos: %s%d%s  %s│%s  modo: %s%s%s\n",CB,prop.name,CR,CD,CR,tele,CD,CR,CB,ntgt,CR,CD,CR,naive?Cy:Cg,naive?"ingenuo":"otimizado",CR);
        printf("\r\033[K  %s──────────────────────────────────────────────────%s\n",CD,CR);
        printf("\r\033[K  %sTRIAGEM%s %s(SHA256 + filtro 0x00)%s\n",Cb,CR,CD,CR);
        printf("\r\033[K     velocidade:  %s%sH/s%s\n",Cg,rateStr(candRate).c_str(),CR);
        printf("\r\033[K     percorridas: %s%s%s candidatos\n",CB,thou(totC).c_str(),CR);
        printf("\r\033[K  %s──────────────────────────────────────────────────%s\n",CD,CR);
        printf("\r\033[K  %sTESTE%s   %s(comb de base fixa + inversao em lote + hash160)%s\n",Co,CR,CD,CR);
        printf("\r\033[K     velocidade:  %s%s keys/s%s\n",Cg,rateStr(testRate).c_str(),CR);
        printf("\r\033[K     percorridas: %s%s%s chaves\n",CB,thou(totT).c_str(),CR);
        printf("\r\033[K  %s──────────────────────────────────────────────────%s\n",CD,CR);
        printf("\r\033[K  %sdecorrido %02ld:%02ld:%02ld%s  %s·%s  espaco S1 = %s2¹¹⁵%s  %s·%s  %s→ %s%s  %s·%s  %sCtrl+C%s\n",
               CD,el/3600,(el%3600)/60,el%60,CR,CD,CR,CB,CR,CD,CR,Cb,savePath?savePath:"(sem -s)",CR,CD,CR,CD,CR);
        fflush(stdout);
    }
    if(!g_plain) printf("\033[?25h");
    report(h_hit.flag!=0);
    cudaFree(d_buf); cudaFree(d_cnt); cudaFree(d_tg); cudaFree(d_hit); return 0;
}
