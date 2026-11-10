// =============================================================================
//  minikey_gpu_opt.cu  —  Pipeline OTIMIZADO (TCC: "mesmo otimizado, inviavel")
//
//  Versao turbinada do minikey_gpu_full.cu. Mantem SHA256/RIPEMD160/hash160 e as
//  ops de campo mod P da cudaMath, mas troca a multiplicacao de ponto ingenua
//  (double-and-add, 256 duplicacoes/chave) por duas otimizacoes REAIS, as unicas
//  que se aplicam a uma base FIXA (G) com chaves aleatorias (priv = SHA256):
//
//   (1) COMB DE BASE FIXA  — como G e sempre o mesmo, pre-computamos uma tabela
//       T[i][j] = (j << 8i)*G (32 janelas x 256 digitos). Cada k*G vira no maximo
//       31 somas de ponto e ZERO duplicacoes. (gerada na GPU pela rotina ingenua
//       ja validada => bit-exata por construcao)
//
//   (2) INVERSAO EM LOTE (truque de Montgomery) — invModP e Fermat (~385 mult/chave!).
//       Processamos M chaves por thread e fazemos 1 inversao + 3(M-1) mult no lugar
//       de M inversoes. A conversao Jacobiano->afim deixa de ser o gargalo.
//
//  wNAF: subsumido pelo comb (sem duplicacoes a otimizar).
//  GLV (endomorfismo lambda/beta): serve p/ base VARIAVEL; no comb de base fixa ele
//     so troca "1 comb de 256 bits" por "2 combs de 128 bits" = MESMO nº de somas,
//     entao nao acelera e so adiciona risco. Documentado, nao implementado de proposito.
//
//  Modos:  validate | bench [-b -t -p] | search <mk22> <K> | run [-b -t -p]
//  Compila: make gpuopt     (BATCHM = chaves por lote, default 8; -DBATCHM=N p/ mudar)
// =============================================================================
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <csignal>
#include <ctime>
#include <cuda_runtime.h>
#include <nvml.h>
#include "secp256k1.cuh"

#ifndef BATCHM
#define BATCHM 8
#endif

// ------------------------------- SHA256 --------------------------------------
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

// ------------------------------- RIPEMD160 -----------------------------------
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
        int rnd=j/16;
        unsigned int fl,fr,kl,kr;
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
    unsigned int o0=h1+cl+dr, o1=h2+dl+er, o2=h3+el+ar, o3=h4+al+br, o4=h0+bl+cr;
    out[0]=o0;out[1]=o1;out[2]=o2;out[3]=o3;out[4]=o4;
}

__device__ void hash160(const unsigned int x[8], const unsigned int y[8], unsigned int out[5]){
    unsigned char pub[65];
    pub[0]=0x04;
    for(int i=0;i<8;i++){ pub[1+4*i]=x[i]>>24; pub[2+4*i]=x[i]>>16; pub[3+4*i]=x[i]>>8; pub[4+4*i]=x[i]; }
    for(int i=0;i<8;i++){ pub[33+4*i]=y[i]>>24; pub[34+4*i]=y[i]>>16; pub[35+4*i]=y[i]>>8; pub[36+4*i]=y[i]; }
    unsigned int sh[8]; sha256(pub,65,sh);
    unsigned char shb[32];
    for(int i=0;i<8;i++){ shb[4*i]=sh[i]>>24; shb[4*i+1]=sh[i]>>16; shb[4*i+2]=sh[i]>>8; shb[4*i+3]=sh[i]; }
    ripemd160(shb,32,out);
}

// ------------------------------- EC (Jacobiano) ------------------------------
__device__ __forceinline__ bool isZeroF(const unsigned int a[8]){
    unsigned int z=0; for(int i=0;i<8;i++) z|=a[i]; return z==0;
}
__device__ __forceinline__ void setField(unsigned int a[8], const unsigned int *s){ for(int i=0;i<8;i++) a[i]=s[i]; }

__device__ void jacDouble(const unsigned int X[8],const unsigned int Y[8],const unsigned int Z[8],
                           unsigned int X3[8],unsigned int Y3[8],unsigned int Z3[8]){
    if(isZeroF(Z)){ for(int i=0;i<8;i++){X3[i]=0;Y3[i]=0;Z3[i]=0;} return; }
    unsigned int A[8],B[8],C[8],D[8],E[8],F[8],t1[8],t2[8],t3[8];
    squareModP(X,A);
    squareModP(Y,B);
    squareModP(B,C);
    addModP(X,B,t1); squareModP(t1,t2);
    subModP(t2,A,t3); subModP(t3,C,t1);
    addModP(t1,t1,D);
    addModP(A,A,t1); addModP(t1,A,E);
    squareModP(E,F);
    addModP(D,D,t1); subModP(F,t1,X3);
    subModP(D,X3,t1); mulModP(E,t1,t2);
    addModP(C,C,t1); addModP(t1,t1,t3); addModP(t3,t3,t1);
    subModP(t2,t1,Y3);
    mulModP(Y,Z,t1); addModP(t1,t1,Z3);
}

__device__ void jacAddAffine(const unsigned int X1[8],const unsigned int Y1[8],const unsigned int Z1[8],
                             const unsigned int x2[8],const unsigned int y2[8],
                             unsigned int X3[8],unsigned int Y3[8],unsigned int Z3[8]){
    if(isZeroF(Z1)){ setField(X3,x2); setField(Y3,y2); for(int i=0;i<8;i++) Z3[i]=0; Z3[7]=1; return; }
    unsigned int Z1Z1[8],U2[8],S2[8],H[8],HH[8],I[8],J[8],r[8],V[8],t1[8],t2[8],t3[8];
    squareModP(Z1,Z1Z1);
    mulModP(x2,Z1Z1,U2);
    mulModP(Z1,Z1Z1,t1); mulModP(y2,t1,S2);
    subModP(U2,X1,H);
    squareModP(H,HH);
    addModP(HH,HH,t1); addModP(t1,t1,I);
    mulModP(H,I,J);
    subModP(S2,Y1,t1); addModP(t1,t1,r);
    mulModP(X1,I,V);
    squareModP(r,t1); subModP(t1,J,t2); addModP(V,V,t3); subModP(t2,t3,X3);
    subModP(V,X3,t1); mulModP(r,t1,t2);
    mulModP(Y1,J,t1); addModP(t1,t1,t3); subModP(t2,t3,Y3);
    mulModP(Z1,H,t1); addModP(t1,t1,Z3);
}

// --- k*G ingenua (double-and-add): usada SO p/ gerar a tabela e validar ------
__device__ void scalarMulG_naive(const unsigned int k[8], unsigned int outX[8], unsigned int outY[8]){
    unsigned int X[8],Y[8],Z[8]; for(int i=0;i<8;i++){X[i]=0;Y[i]=0;Z[i]=0;}
    unsigned int nX[8],nY[8],nZ[8];
    for(int wi=0; wi<8; wi++){
        unsigned int wv=k[wi];
        for(int bp=31; bp>=0; bp--){
            jacDouble(X,Y,Z,nX,nY,nZ); setField(X,nX);setField(Y,nY);setField(Z,nZ);
            if((wv>>bp)&1){ jacAddAffine(X,Y,Z,_GX,_GY,nX,nY,nZ); setField(X,nX);setField(Y,nY);setField(Z,nZ); }
        }
    }
    unsigned int Zi[8],Zi2[8],Zi3[8];
    invModP(Z,Zi); squareModP(Zi,Zi2); mulModP(Zi2,Zi,Zi3);
    mulModP(X,Zi2,outX); mulModP(Y,Zi3,outY);
}

// ============================ OTIMIZACAO 1: COMB =============================
// Tabela de base fixa: d_comb[(i*256 + j)*16 + 0..7] = X, +8..15 = Y  do ponto
// (j << 8i)*G, em coordenadas afim. 32 janelas (8 bits) x 256 digitos = 512 KB.
#define NWIN 32
__device__ unsigned int d_comb[NWIN*256*16];

__global__ void kBuildComb(){
    int i=blockIdx.x;      // janela 0..31 (byte i, do menos significativo)
    int j=threadIdx.x;     // digito 0..255
    unsigned int *T=d_comb+((i*256+j)*16);
    if(j==0){ for(int t=0;t<16;t++) T[t]=0; return; }   // 0*G = infinito (nao usado)
    unsigned int k[8]; for(int t=0;t<8;t++) k[t]=0;
    int wi=7-(i>>2), sh=8*(i&3);                          // k big-endian: k[0]=MSW
    k[wi]=((unsigned int)j)<<sh;
    unsigned int x[8],y[8]; scalarMulG_naive(k,x,y);
    for(int t=0;t<8;t++){ T[t]=x[t]; T[8+t]=y[t]; }
}

// k*G via comb -> Jacobiano (sem inversao final; p/ lote). <=31 somas, 0 duplicacoes.
__device__ void combJac(const unsigned int k[8], unsigned int X[8],unsigned int Y[8],unsigned int Z[8]){
    for(int i=0;i<8;i++){X[i]=0;Y[i]=0;Z[i]=0;}          // infinito
    unsigned int nX[8],nY[8],nZ[8];
    for(int i=0;i<NWIN;i++){
        unsigned int digit=(k[7-(i>>2)]>>(8*(i&3)))&0xFF;
        if(digit){
            const unsigned int *T=d_comb+((i*256+digit)*16);
            jacAddAffine(X,Y,Z,T,T+8,nX,nY,nZ);
            setField(X,nX);setField(Y,nY);setField(Z,nZ);
        }
    }
}

// k*G via comb -> afim (1 inversao). Usada em caminhos de 1 chave (search/alvos).
__device__ void scalarMulG_comb(const unsigned int k[8], unsigned int outX[8], unsigned int outY[8]){
    unsigned int X[8],Y[8],Z[8]; combJac(k,X,Y,Z);
    unsigned int Zi[8],Zi2[8],Zi3[8];
    invModP(Z,Zi); squareModP(Zi,Zi2); mulModP(Zi2,Zi,Zi3);
    mulModP(X,Zi2,outX); mulModP(Y,Zi3,outY);
}

// ===================== OTIMIZACAO 2: hash160 de um LOTE ======================
// Deriva hash160 de 'cnt' chaves (cnt<=BATCHM) com UMA inversao (Montgomery).
__device__ void deriveBatchHash(const unsigned int priv[BATCHM][8], int cnt, unsigned int outh[BATCHM][5]){
    unsigned int X[BATCHM][8],Y[BATCHM][8],Z[BATCHM][8],pre[BATCHM][8];
    for(int m=0;m<cnt;m++) combJac(priv[m],X[m],Y[m],Z[m]);
    // prefixos: pre[m] = Z[0]*...*Z[m]
    setField(pre[0],Z[0]);
    for(int m=1;m<cnt;m++) mulModP(pre[m-1],Z[m],pre[m]);
    unsigned int acc[8]; invModP(pre[cnt-1],acc);         // acc = 1/(Z0..Z_{cnt-1})
    for(int m=cnt-1;m>=1;m--){
        unsigned int zi[8],zi2[8],zi3[8],x[8],y[8];
        mulModP(acc,pre[m-1],zi);                         // zi = 1/Z[m]
        squareModP(zi,zi2); mulModP(zi2,zi,zi3);
        mulModP(X[m],zi2,x); mulModP(Y[m],zi3,y);
        hash160(x,y,outh[m]);
        mulModP(acc,Z[m],acc);                            // acc = 1/(Z0..Z_{m-1})
    }
    { unsigned int zi2[8],zi3[8],x[8],y[8];               // m=0: zi = acc
      squareModP(acc,zi2); mulModP(zi2,acc,zi3);
      mulModP(X[0],zi2,x); mulModP(Y[0],zi3,y);
      hash160(x,y,outh[0]); }
}

// ------------------------------- Validacao -----------------------------------
// Prova que o COMB e o lote dao bit-exatamente o mesmo que a rotina ingenua.
__global__ void kValidate(unsigned int *g_out){
    unsigned int k[8],x[8],y[8],h[5];
    for(int i=0;i<8;i++) k[i]=0; k[7]=1; scalarMulG_comb(k,x,y);     // comb G
    for(int i=0;i<8;i++){ g_out[i]=x[i]; g_out[8+i]=y[i]; }
    hash160(x,y,h); for(int i=0;i<5;i++) g_out[16+i]=h[i];
    for(int i=0;i<8;i++) k[i]=0; k[7]=2; scalarMulG_comb(k,x,y);     // comb 2G
    for(int i=0;i<8;i++){ g_out[21+i]=x[i]; g_out[29+i]=y[i]; }
    unsigned char abc[3]={'a','b','c'}; unsigned int sh[8]; sha256(abc,3,sh);
    for(int i=0;i<8;i++) g_out[37+i]=sh[i];
    const char *s56="abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    unsigned char b56[56]; for(int i=0;i<56;i++) b56[i]=s56[i];
    unsigned int sh2[8]; sha256(b56,56,sh2);
    for(int i=0;i<8;i++) g_out[45+i]=sh2[i];
    unsigned int rc[5]; unsigned char dummy[1];
    ripemd160(dummy,0,rc); for(int i=0;i<5;i++) g_out[53+i]=rc[i];
    unsigned char ab[3]={'a','b','c'}; ripemd160(ab,3,rc); for(int i=0;i<5;i++) g_out[58+i]=rc[i];
    // CRUZADO: k "aleatorio" (= SHA256("tcc")) derivado por NAIVE e por COMB+LOTE.
    unsigned char seed3[3]={'t','c','c'}; unsigned int kk[8]; sha256(seed3,3,kk);
    unsigned int nx[8],ny[8],nh[5]; scalarMulG_naive(kk,nx,ny); hash160(nx,ny,nh);
    for(int i=0;i<5;i++) g_out[63+i]=nh[i];                          // hash160 via NAIVE
    unsigned int priv[BATCHM][8]; unsigned int oh[BATCHM][5];
    for(int i=0;i<8;i++) priv[0][i]=kk[i];
    deriveBatchHash(priv,1,oh);                                     // hash160 via COMB+LOTE
    for(int i=0;i<5;i++) g_out[68+i]=oh[0][i];
}

// ------------------------------- Benchmark -----------------------------------
__device__ __constant__ char dB58[58]={
 '1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
 'a','b','c','d','e','f','g','h','i','j','k','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
__device__ __forceinline__ uint64_t xs(uint64_t &s){s^=s<<13;s^=s>>7;s^=s<<17;return s;}

// Pipeline COMPLETO otimizado: lotes de BATCHM chaves (comb + inversao unica).
__global__ void kBench(uint64_t seed,int groups, unsigned long long *done){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned char idx[21]; uint64_t s=seed^((uint64_t)gid*0x9E3779B97F4A7C15ULL);
    for(int i=0;i<21;i++) idx[i]=(unsigned char)(xs(s)%58);
    unsigned int acc=0;
    for(int g=0;g<groups;g++){
        unsigned int priv[BATCHM][8];
        for(int m=0;m<BATCHM;m++){
            unsigned char mk[22]; mk[0]='S'; for(int i=0;i<21;i++) mk[i+1]=dB58[idx[i]];
            sha256(mk,22,priv[m]);
            int c=0; while(c<21){ if(++idx[c]>=58){idx[c]=0;c++;} else break; }
        }
        unsigned int oh[BATCHM][5]; deriveBatchHash(priv,BATCHM,oh);
        for(int m=0;m<BATCHM;m++) acc^=oh[m][0];
    }
    if(acc==0xFFFFFFFF) done[gid&0]+=0;
    atomicAdd(done,(unsigned long long)groups*BATCHM);
}

// ------------------------------- Busca (bits reduzidos) ----------------------
__device__ __constant__ unsigned char cKnownIdx[21];

__global__ void kDeriveKnown(unsigned int *target){
    unsigned char mk[22]; mk[0]='S'; for(int i=0;i<21;i++) mk[i+1]=dB58[cKnownIdx[i]];
    unsigned int priv[8]; sha256(mk,22,priv);
    unsigned int x[8],y[8]; scalarMulG_comb(priv,x,y); hash160(x,y,target);
}

__global__ void kSearch(const unsigned int *target,int K,unsigned long long total,
                        int *foundFlag, unsigned char *foundIdx){
    unsigned long long tid=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
    unsigned char idx[21];
    for(unsigned long long c=tid;c<total;c+=stride){
        if(*foundFlag) return;
        for(int i=0;i<21;i++) idx[i]=cKnownIdx[i];
        unsigned long long t=c; for(int j=0;j<K;j++){ idx[21-K+j]=(unsigned char)(t%58); t/=58; }
        unsigned char mkq[23]; mkq[0]='S'; for(int i=0;i<21;i++) mkq[i+1]=dB58[idx[i]]; mkq[22]='?';
        unsigned int h[8]; sha256(mkq,23,h); if((h[0]>>24)!=0) continue;
        unsigned char mk[22]; for(int i=0;i<22;i++) mk[i]=mkq[i];
        unsigned int priv[8]; sha256(mk,22,priv);
        unsigned int x[8],y[8],hh[5]; scalarMulG_comb(priv,x,y); hash160(x,y,hh);
        bool eq=true; for(int i=0;i<5;i++) if(hh[i]!=target[i]) eq=false;
        if(eq){ if(atomicCAS(foundFlag,0,1)==0){ for(int i=0;i<21;i++) foundIdx[i]=idx[i]; } return; }
    }
}

// ------------------------------- Modo RUN (painel ao vivo) -------------------
__constant__ unsigned int cTgt[16][5];
__constant__ int cNtgt;

__global__ void kTriageCollect(uint64_t seed,int ppt, unsigned char *buf, unsigned int *cnt, unsigned int cap){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned char idx[21]; uint64_t s=seed^((uint64_t)gid*0x9E3779B97F4A7C15ULL);
    for(int i=0;i<21;i++) idx[i]=(unsigned char)(xs(s)%58);
    for(int p=0;p<ppt;p++){
        unsigned char mkq[23]; mkq[0]='S'; for(int i=0;i<21;i++) mkq[i+1]=dB58[idx[i]]; mkq[22]='?';
        unsigned int h[8]; sha256(mkq,23,h);
        if((h[0]>>24)==0){
            unsigned int pos=atomicAdd(cnt,1u);
            if(pos<cap){ unsigned char *d=buf+(size_t)pos*21; for(int i=0;i<21;i++) d[i]=idx[i]; }
        }
        int c=0; while(c<21){ if(++idx[c]>=58){idx[c]=0;c++;} else break; }
    }
}

// FASE B otimizada: cada thread processa BATCHM validas por rodada (1 inversao).
__global__ void kDerive(const unsigned char *buf, unsigned int n, int *foundFlag, unsigned char *foundIdx){
    unsigned int stride=(unsigned int)gridDim.x*blockDim.x*BATCHM;
    for(unsigned int base=(blockIdx.x*blockDim.x+threadIdx.x)*BATCHM; base<n; base+=stride){
        int cnt=BATCHM; if(base+cnt>n) cnt=n-base;
        unsigned int priv[BATCHM][8];
        for(int m=0;m<cnt;m++){
            const unsigned char *d=buf+(size_t)(base+m)*21;
            unsigned char mk[22]; mk[0]='S'; for(int j=0;j<21;j++) mk[j+1]=dB58[d[j]];
            sha256(mk,22,priv[m]);
        }
        unsigned int oh[BATCHM][5]; deriveBatchHash(priv,cnt,oh);
        for(int m=0;m<cnt;m++)
            for(int tI=0;tI<cNtgt;tI++){ bool eq=true; for(int k=0;k<5;k++) if(oh[m][k]!=cTgt[tI][k]) eq=false;
                if(eq){ if(atomicCAS(foundFlag,0,1)==0){ const unsigned char *d=buf+(size_t)(base+m)*21; for(int j=0;j<21;j++) foundIdx[j]=d[j]; } } }
    }
}

static const char *HB58="123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static int b58idx(char c){ for(int i=0;i<58;i++) if(HB58[i]==c) return i; return -1; }
static void hx(const unsigned int *w,int n){ for(int i=0;i<n;i++) printf("%08x",w[i]); }

#define CR "\033[0m"
#define CB "\033[1m"
#define CD "\033[2m"
#define Cg "\033[32m"
#define Cy "\033[33m"
#define Cb "\033[38;5;39m"
#define Cc "\033[36m"
#define Co "\033[38;5;208m"
#define Cr "\033[31m"
static volatile sig_atomic_t g_stop=0;
static void onint(int){ g_stop=1; }
static nvmlDevice_t g_nd; static bool g_nv=false;
static void nvInit(){ if(nvmlInit()==NVML_SUCCESS && nvmlDeviceGetHandleByIndex(0,&g_nd)==NVML_SUCCESS) g_nv=true; }
static void nvRead(unsigned int&t,unsigned int&c){ t=0;c=0; if(g_nv){ nvmlDeviceGetTemperature(g_nd,NVML_TEMPERATURE_GPU,&t); nvmlDeviceGetClockInfo(g_nd,NVML_CLOCK_SM,&c);} }
static std::string thou(unsigned long long v){ std::string s=std::to_string(v),o; int c=0; for(int i=(int)s.size()-1;i>=0;i--){ o=std::string(1,s[i])+o; if(++c%3==0&&i>0) o=","+o; } return o; }
static std::string rateStr(double r){ char b[48]; if(r>=1e9) sprintf(b,"%.3f G",r/1e9); else if(r>=1e6) sprintf(b,"%.2f M",r/1e6); else if(r>=1e3) sprintf(b,"%.1f k",r/1e3); else sprintf(b,"%.0f ",r); return b; }
static const char* tcol(unsigned int t){ return t>=85?Cr : (t>=75?Cy:Cg); }

static void buildCombHost(){
    kBuildComb<<<NWIN,256>>>(); cudaDeviceSynchronize();
    cudaError_t e=cudaGetLastError(); if(e){ printf("ERRO ao gerar a tabela comb: %s\n",cudaGetErrorString(e)); exit(1); }
}

int main(int argc,char**argv){
    const char*mode=argc>1?argv[1]:"validate";
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);

    if(!strcmp(mode,"validate")){
        buildCombHost();
        unsigned int *d_out,h_out[73]; cudaMalloc(&d_out,73*sizeof(unsigned int));
        kValidate<<<1,1>>>(d_out); cudaDeviceSynchronize();
        cudaError_t e=cudaGetLastError(); if(e){printf("ERRO CUDA: %s\n",cudaGetErrorString(e));return 1;}
        cudaMemcpy(h_out,d_out,73*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        printf("== Validacao COMB + lote na GPU (%s, BATCHM=%d) ==\n",prop.name,BATCHM);
        printf("  G.x = "); hx(h_out,8); printf("\n  esperado _GX = 79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798\n");
        printf("  G.y = "); hx(h_out+8,8); printf("\n");
        printf("  hash160(G) = ");
        for(int i=0;i<5;i++){unsigned int w=h_out[16+i];printf("%02x%02x%02x%02x",w&0xff,(w>>8)&0xff,(w>>16)&0xff,(w>>24)&0xff);}
        printf("\n  esperado                                           = 91b24bf9f5288532960ac687abb035127b1d28a5\n");
        printf("  2G.x = "); hx(h_out+21,8); printf("\n  esperado = c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5\n");
        printf("  2G.y = "); hx(h_out+29,8); printf("\n  esperado = 1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a\n");
        printf("  SHA256(\"abc\") = "); hx(h_out+37,8); printf("\n  esperado       = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n");
        printf("  SHA256(56B)  = "); hx(h_out+45,8); printf("\n  esperado       = 248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1\n");
        printf("  RIPEMD160(\"\")  = "); for(int i=0;i<5;i++){unsigned int w=h_out[53+i];printf("%02x%02x%02x%02x",w&0xff,(w>>8)&0xff,(w>>16)&0xff,(w>>24)&0xff);} printf("\n  esperado        = 9c1185a5c5e9fc54612808977ee8f548b2258d31\n");
        printf("  RIPEMD160(abc) = "); for(int i=0;i<5;i++){unsigned int w=h_out[58+i];printf("%02x%02x%02x%02x",w&0xff,(w>>8)&0xff,(w>>16)&0xff,(w>>24)&0xff);} printf("\n  esperado        = 8eb208f7e05d987a9b044a8e98c6b087f15a0bfc\n");
        // cruzado naive x comb+lote
        bool same=true; for(int i=0;i<5;i++) if(h_out[63+i]!=h_out[68+i]) same=false;
        printf("  CRUZADO k=SHA256(\"tcc\"):  hash160 NAIVE = ");
        for(int i=0;i<5;i++){unsigned int w=h_out[63+i];printf("%02x%02x%02x%02x",w&0xff,(w>>8)&0xff,(w>>16)&0xff,(w>>24)&0xff);}
        printf("\n                            hash160 COMB+ = ");
        for(int i=0;i<5;i++){unsigned int w=h_out[68+i];printf("%02x%02x%02x%02x",w&0xff,(w>>8)&0xff,(w>>16)&0xff,(w>>24)&0xff);}
        printf("\n  >> %s%s%s (otimizado %s ingenuo)\n", same?Cg:Cr, same?"IDENTICO":"DIFERENTE!", CR, same?"==":"!=");
        cudaFree(d_out); return same?0:2;
    }

    if(!strcmp(mode,"search")){
        if(argc<4){ printf("uso: search <minikey_conhecida_22> <K_chars_desconhecidos> [-b N -t N]\n"); return 1; }
        buildCombHost();
        const char *known=argv[2]; int K=atoi(argv[3]);
        int blk=240,thr=256;
        for(int i=4;i<argc-1;i++){ if(!strcmp(argv[i],"-b"))blk=atoi(argv[++i]); else if(!strcmp(argv[i],"-t"))thr=atoi(argv[++i]); }
        if((int)strlen(known)!=22 || known[0]!='S'){ printf("minikey deve ter 22 chars e comecar com S\n"); return 1; }
        unsigned char hIdx[21];
        for(int i=0;i<21;i++){ int v=b58idx(known[1+i]); if(v<0){printf("char invalido\n");return 1;} hIdx[i]=(unsigned char)v; }
        cudaMemcpyToSymbol(cKnownIdx,hIdx,21);
        unsigned int *d_target; cudaMalloc(&d_target,5*sizeof(unsigned int));
        kDeriveKnown<<<1,1>>>(d_target); cudaDeviceSynchronize();
        unsigned long long total=1; for(int i=0;i<K;i++) total*=58;
        double bits=K*5.8579;
        int *d_flag; unsigned char *d_found; cudaMalloc(&d_flag,sizeof(int)); cudaMemset(d_flag,0,sizeof(int)); cudaMalloc(&d_found,21);
        printf("GPU: %s  (OTIMIZADO: comb + lote)\n",prop.name);
        printf("Minikey conhecida: %s  (fixando %d chars, buscando os %d ultimos)\n",known,22-K,K);
        printf("Espaco de busca: 58^%d = %llu candidatos  (~%.1f bits)\n",K,total,bits);
        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0); kSearch<<<blk,thr>>>(d_target,K,total,d_flag,d_found); cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaError_t e=cudaGetLastError(); if(e){printf("ERRO CUDA: %s\n",cudaGetErrorString(e));return 1;}
        float ms=0; cudaEventElapsedTime(&ms,t0,t1);
        int flag=0; unsigned char found[21]; cudaMemcpy(&flag,d_flag,sizeof(int),cudaMemcpyDeviceToHost); cudaMemcpy(found,d_found,21,cudaMemcpyDeviceToHost);
        printf("tempo: %.3f s\n",ms/1000.0);
        if(flag){ char mk[23]; mk[0]='S'; for(int i=0;i<21;i++) mk[i+1]=HB58[found[i]]; mk[22]=0;
            printf(">> ACHOU: %s   %s\n",mk,strcmp(mk,known)==0?"(== conhecida, OK!)":"(DIFERENTE?!)"); }
        else printf(">> nao achou (inesperado para alvo nosso)\n");
        cudaFree(d_target); cudaFree(d_flag); cudaFree(d_found); return 0;
    }

    if(!strcmp(mode,"run")){
        buildCombHost();
        int blk=120,thr=256,ppt=2048;
        for(int i=2;i<argc-1;i++){ if(!strcmp(argv[i],"-b"))blk=atoi(argv[++i]); else if(!strcmp(argv[i],"-t"))thr=atoi(argv[++i]); else if(!strcmp(argv[i],"-p"))ppt=atoi(argv[++i]); }
        FILE *f=fopen("data/our-minikeys-s1.txt","r");
        if(!f){ printf("Rode dentro de minikey-poc/ (nao achei our-minikeys-s1.txt)\n"); return 1; }
        unsigned int htgt[16][5]; int nt=0; char line[256];
        unsigned int *d_t; cudaMalloc(&d_t,5*sizeof(unsigned int));
        while(fgets(line,sizeof(line),f) && nt<16){
            char mk[64]; if(sscanf(line,"%63s",mk)!=1) continue;
            if(strlen(mk)!=22 || mk[0]!='S') continue;
            unsigned char hi[21]; bool ok=true;
            for(int i=0;i<21;i++){ int v=b58idx(mk[1+i]); if(v<0){ok=false;break;} hi[i]=(unsigned char)v; }
            if(!ok) continue;
            cudaMemcpyToSymbol(cKnownIdx,hi,21); kDeriveKnown<<<1,1>>>(d_t); cudaDeviceSynchronize();
            cudaMemcpy(htgt[nt],d_t,5*sizeof(unsigned int),cudaMemcpyDeviceToHost); nt++;
        }
        fclose(f);
        if(nt==0){ printf("nenhuma minikey valida no arquivo\n"); return 1; }
        cudaMemcpyToSymbol(cTgt,htgt,sizeof(unsigned int)*5*nt);
        cudaMemcpyToSymbol(cNtgt,&nt,sizeof(int));

        signal(SIGINT,onint); nvInit();
        const unsigned int CAP=2000000;
        unsigned char *d_buf; unsigned int *d_cnt; int *d_flag; unsigned char *d_found;
        cudaMalloc(&d_buf,(size_t)CAP*21); cudaMalloc(&d_cnt,4); cudaMalloc(&d_flag,4); cudaMalloc(&d_found,21); cudaMemset(d_flag,0,4);
        unsigned long long totC=0,totT=0; double candRate=0,testRate=0;
        uint64_t seed=(uint64_t)time(NULL)*0x9E3779B97F4A7C15ULL;
        int foundFlag=0; unsigned char found[21];
        struct timespec ts0; clock_gettime(CLOCK_MONOTONIC,&ts0);
        const int PL=12; printf("\033[?25l"); for(int i=0;i<PL;i++) printf("\n");

        while(!g_stop && !foundFlag){
            seed = seed*6364136223846793005ULL + 1442695040888963407ULL;
            cudaMemset(d_cnt,0,4);
            cudaEvent_t a0,a1; cudaEventCreate(&a0); cudaEventCreate(&a1);
            cudaEventRecord(a0); kTriageCollect<<<blk,thr>>>(seed,ppt,d_buf,d_cnt,CAP); cudaEventRecord(a1); cudaEventSynchronize(a1);
            float msA=0; cudaEventElapsedTime(&msA,a0,a1); cudaEventDestroy(a0); cudaEventDestroy(a1);
            unsigned int nval=0; cudaMemcpy(&nval,d_cnt,4,cudaMemcpyDeviceToHost); if(nval>CAP) nval=CAP;
            unsigned long long candA=(unsigned long long)blk*thr*ppt;
            cudaEvent_t b0,b1; cudaEventCreate(&b0); cudaEventCreate(&b1);
            cudaEventRecord(b0); kDerive<<<blk,thr>>>(d_buf,nval,d_flag,d_found); cudaEventRecord(b1); cudaEventSynchronize(b1);
            float msB=0; cudaEventElapsedTime(&msB,b0,b1); cudaEventDestroy(b0); cudaEventDestroy(b1);
            totC+=candA; totT+=nval;
            if(msA>0) candRate=candA/(msA/1000.0);
            if(msB>0) testRate=nval/(msB/1000.0);
            cudaMemcpy(&foundFlag,d_flag,4,cudaMemcpyDeviceToHost);
            struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
            long el=(long)((ts.tv_sec-ts0.tv_sec));
            unsigned int temp,clk; nvRead(temp,clk);
            char tele[96]=""; if(g_nv) sprintf(tele," %s\xe2\x94\x82%s  %s%u\xc2\xb0""C%s %s%u MHz%s",CD,CR,tcol(temp),temp,CR,CB,clk,CR);

            printf("\033[%dA",PL);
            printf("\r\033[K%s\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90%s%s MINIKEY S1 \xc2\xb7 OTIMIZADO (comb+lote) %s%s\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90%s\n",Cc,CR,CB,CR,Cc,CR);
            printf("\r\033[K  %s%s%s%s  %s\xe2\x94\x82%s  alvos (nossos): %s%d%s\n",CB,"RTX 3060 Laptop",CR,tele,CD,CR,CB,nt,CR);
            printf("\r\033[K  %s\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80%s\n",CD,CR);
            printf("\r\033[K  %sTRIAGEM%s %s(SHA256 + filtro 0x00)%s\n",Cb,CR,CD,CR);
            printf("\r\033[K     velocidade:  %s%sH/s%s\n",Cg,rateStr(candRate).c_str(),CR);
            printf("\r\033[K     percorridas: %s%s%s candidatos\n",CB,thou(totC).c_str(),CR);
            printf("\r\033[K  %s\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80%s\n",CD,CR);
            printf("\r\033[K  %sTESTE%s   %s(comb de base fixa + inversao em lote + hash160)%s\n",Co,CR,CD,CR);
            printf("\r\033[K     velocidade:  %s%s keys/s%s\n",Cg,rateStr(testRate).c_str(),CR);
            printf("\r\033[K     percorridas: %s%s%s chaves\n",CB,thou(totT).c_str(),CR);
            printf("\r\033[K  %s\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80%s\n",CD,CR);
            printf("\r\033[K  %sdecorrido %02ld:%02ld:%02ld%s  %s\xc2\xb7%s  espaco S1 = %s2\xc2\xb9\xc2\xb9\xe2\x81\xb5%s  %s\xc2\xb7%s  %sCtrl+C para parar%s\n",
                   CD,el/3600,(el%3600)/60,el%60,CR,CD,CR,CB,CR,CD,CR,CD,CR);
            fflush(stdout);
        }
        printf("\033[?25h");
        if(foundFlag){ cudaMemcpy(found,d_found,21,cudaMemcpyDeviceToHost); char mk[23]; mk[0]='S'; for(int i=0;i<21;i++) mk[i+1]=HB58[found[i]]; mk[22]=0;
            printf("\n%s>> HIT:%s %s\n",Cg,CR,mk); }
        else printf("\n%s[parado]%s nenhuma chave encontrada \xe2\x80\x94 exatamente o esperado para 2^115.\n",Cy,CR);
        return 0;
    }

    // ----- bench -----
    buildCombHost();
    int blocks=120,threads=256,ppt=64;
    for(int i=1;i<argc-1;i++){ if(!strcmp(argv[i],"-b"))blocks=atoi(argv[++i]); else if(!strcmp(argv[i],"-t"))threads=atoi(argv[++i]); else if(!strcmp(argv[i],"-p"))ppt=atoi(argv[++i]); }
    int groups=ppt/BATCHM; if(groups<1) groups=1;
    printf("GPU: %s (%d SMs)  [OTIMIZADO: comb de base fixa + inversao em lote, BATCHM=%d]\n",prop.name,prop.multiProcessorCount,BATCHM);
    printf("Config: %d x %d x %d grupos x %d = %.2f M derivacoes COMPLETAS\n",blocks,threads,groups,BATCHM,(double)blocks*threads*groups*BATCHM/1e6);
    unsigned long long *d_done,h_done=0; cudaMalloc(&d_done,sizeof(unsigned long long)); cudaMemset(d_done,0,sizeof(unsigned long long));
    uint64_t seed=(uint64_t)time(NULL)*0x100000001B3ULL^(uint64_t)clock();
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0); kBench<<<blocks,threads>>>(seed,groups,d_done); cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaError_t e=cudaGetLastError(); if(e){printf("ERRO CUDA: %s\n",cudaGetErrorString(e));return 1;}
    float ms=0; cudaEventElapsedTime(&ms,t0,t1); cudaMemcpy(&h_done,d_done,sizeof(unsigned long long),cudaMemcpyDeviceToHost);
    double sec=ms/1000.0; unsigned long long tot=h_done;
    printf("\n--- PIPELINE OTIMIZADO (SHA256 + comb + inversao em lote + hash160) ---\n");
    printf("  tempo:       %.3f s\n",sec);
    printf("  derivacoes:  %llu\n",tot);
    printf("  TAXA:        %.3f M keys/s   (%.0f keys/s)\n",tot/sec/1e6,tot/sec);
    printf("\n  Compare com o pipeline INGENUO (minikey_gpu_full bench).\n");
    cudaFree(d_done); return 0;
}
