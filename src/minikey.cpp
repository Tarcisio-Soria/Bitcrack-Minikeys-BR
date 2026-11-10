// =============================================================================
//  minikey.cpp  —  PoC de referência (CPU) para minikeys Casascius
//  Projeto de TCC: demonstrar a INVIABILIDADE da força bruta de minikeys S1.
//
//  Reaproveita as primitivas já testadas do BitCrack-BR:
//    - crypto::sha256        (CryptoUtil)   -> check 0x00 e derivação da privkey
//    - secp256k1::multiplyPoint / G()       -> privkey -> ponto público
//    - Address::fromPublicKey               -> endereço (uncompressed = S1)
//
//  Pipeline de uma minikey:
//    string "S..."  ->  SHA256(mk + '?')[0]==0x00 ?  ->  priv = SHA256(mk)
//                   ->  P = priv·G  ->  endereço (uncompressed)
//
//  Esta é a Fase 1 (referência + corretude + baseline de CPU). O kernel CUDA
//  e o benchmark na GPU vêm na Fase 2.
// =============================================================================
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <random>
#include <chrono>

#include "secp256k1.h"
#include "CryptoUtil.h"
#include "AddressUtil.h"

// Alfabeto base58 (ordem do Bitcoin; sem 0 O I l)
static const char *B58 =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// -----------------------------------------------------------------------------
// SHA256 de uma string curta (<= 55 bytes -> cabe em 1 bloco de 512 bits).
// Monta o bloco já com padding e chama o sha256 single-block do BitCrack.
// -----------------------------------------------------------------------------
static void sha256_str(const std::string &s, unsigned int out[8])
{
    unsigned int w[16];
    memset(w, 0, sizeof(w));
    int len = (int)s.size();
    for (int i = 0; i < len; i++)
        w[i >> 2] |= (unsigned int)(unsigned char)s[i] << (24 - 8 * (i & 3));
    w[len >> 2] |= 0x80u << (24 - 8 * (len & 3));   // bit '1' de padding
    w[15] = (unsigned int)(len * 8);                 // comprimento em bits
    crypto::sha256Init(out);
    crypto::sha256(w, out);
}

// Check de digitação do Casascius: SHA256(minikey + '?') começa com 0x00
static inline bool passesTypoCheck(const std::string &mk)
{
    unsigned int h[8];
    sha256_str(mk + "?", h);
    return (h[0] >> 24) == 0;           // primeiro byte do hash == 0x00
}

static inline bool inAlphabet(char c) { return strchr(B58, c) != nullptr; }

// Validação completa de formato (22 = Série 1, 26, 30 = Série 2)
static bool isValidMinikey(const std::string &mk)
{
    size_t n = mk.size();
    if (n != 22 && n != 26 && n != 30) return false;
    if (mk[0] != 'S') return false;
    for (size_t i = 1; i < n; i++) if (!inAlphabet(mk[i])) return false;
    return passesTypoCheck(mk);
}

// privkey = SHA256(minikey)  (digest big-endian -> uint256)
static secp256k1::uint256 privFromMinikey(const std::string &mk)
{
    unsigned int h[8];
    sha256_str(mk, h);
    return secp256k1::uint256(h, secp256k1::uint256::BigEndian);
}

// Endereço Bitcoin da minikey (uncompressed por padrão = moedas Casascius)
static std::string addressFromMinikey(const std::string &mk, bool compressed = false)
{
    secp256k1::uint256 k = privFromMinikey(mk);
    secp256k1::ecpoint p = secp256k1::multiplyPoint(k, secp256k1::G());
    return Address::fromPublicKey(p, compressed);
}

// -----------------------------------------------------------------------------
// Gera uma minikey VÁLIDA aleatória de comprimento 'len' (força bruta do 0x00).
// 'tested' recebe quantos candidatos foram testados até achar (~256 em média).
// -----------------------------------------------------------------------------
static std::string genValidMinikey(std::mt19937_64 &rng, int len, uint64_t *tested)
{
    std::string mk(len, 'S');
    uint64_t t = 0;
    for (;;) {
        for (int i = 1; i < len; i++) mk[i] = B58[rng() % 58];
        t++;
        if (passesTypoCheck(mk)) { if (tested) *tested = t; return mk; }
    }
}

// =============================================================================
//  Modos
// =============================================================================
static int selftest()
{
    int fails = 0;
    printf("== Teste de corretude (KAT) ==\n");

    // KAT 1: SHA256("abc")
    unsigned int h[8];
    sha256_str("abc", h);
    char got[65];
    for (int i = 0; i < 8; i++) sprintf(got + 8 * i, "%08x", h[i]);
    const char *exp1 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    bool ok1 = (strcmp(got, exp1) == 0);
    printf("  [%s] SHA256(\"abc\")\n", ok1 ? "OK" : "FALHOU"); fails += !ok1;

    // KAT 2: vetor de minikey conhecido (Bitcoin Wiki)
    const std::string mk = "S6c56bnXQiBjk9mqSYE7ykVQ7NzrRy";  // 30 chars
    const std::string expPriv =
        "4C7A9640C72DC2099F23715D0C8A0D8A35F8906E3CAB61DD3F78B67BF887C9AB";
    const std::string expAddr = "1CciesT23BNionJeXrbxmjc7ywfiyM4oLW";

    bool okV = isValidMinikey(mk);
    printf("  [%s] minikey passa no check 0x00\n", okV ? "OK" : "FALHOU"); fails += !okV;

    std::string priv = privFromMinikey(mk).toString(16);
    bool okP = (priv == expPriv);
    printf("  [%s] privkey = SHA256(minikey)\n", okP ? "OK" : "FALHOU");
    if (!okP) printf("        obtido:   %s\n        esperado: %s\n", priv.c_str(), expPriv.c_str());
    fails += !okP;

    std::string addr = addressFromMinikey(mk);
    bool okA = (addr == expAddr);
    printf("  [%s] endereço (uncompressed)\n", okA ? "OK" : "FALHOU");
    if (!okA) printf("        obtido:   %s\n        esperado: %s\n", addr.c_str(), expAddr.c_str());
    fails += !okA;

    printf("\n%s\n", fails == 0 ? ">> TUDO OK: pipeline minikey validado." :
                                  ">> HÁ FALHAS no pipeline.");
    return fails;
}

static int gen(int count, int len)
{
    std::mt19937_64 rng((uint64_t)std::random_device{}() ^
                        (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count());
    printf("== Gerando %d minikey(s) válida(s) de %d chars ==\n", count, len);
    for (int i = 0; i < count; i++) {
        uint64_t tested = 0;
        std::string mk = genValidMinikey(rng, len, &tested);
        printf("  minikey : %s\n", mk.c_str());
        printf("  privkey : %s\n", privFromMinikey(mk).toString(16).c_str());
        printf("  endereço: %s   (%llu candidatos)\n\n",
               addressFromMinikey(mk).c_str(), (unsigned long long)tested);
    }
    return 0;
}

static int verify(const std::string &mk)
{
    bool ok = isValidMinikey(mk);
    printf("minikey : %s\n", mk.c_str());
    printf("válida  : %s\n", ok ? "SIM" : "NÃO");
    if (mk.size() == 22 || mk.size() == 26 || mk.size() == 30) {
        printf("privkey : %s\n", privFromMinikey(mk).toString(16).c_str());
        printf("endereço: %s\n", addressFromMinikey(mk).c_str());
    }
    return ok ? 0 : 1;
}

static int bench(double secs, int len)
{
    std::mt19937_64 rng(12345);
    using clk = std::chrono::steady_clock;

    // --- Taxa de TRIAGEM: gerar candidato + SHA256(+'?') + check 0x00 ---
    printf("== Benchmark CPU (1 thread) ==\n");
    {
        std::string mk(len, 'S');
        uint64_t n = 0, valid = 0;
        auto t0 = clk::now();
        while (std::chrono::duration<double>(clk::now() - t0).count() < secs) {
            for (int r = 0; r < 4096; r++) {
                for (int i = 1; i < len; i++) mk[i] = B58[rng() % 58];
                if (passesTypoCheck(mk)) valid++;
                n++;
            }
        }
        double dt = std::chrono::duration<double>(clk::now() - t0).count();
        printf("  Triagem (SHA256 + filtro 0x00): %.3f M cand/s  (%llu válidas em %llu)\n",
               n / dt / 1e6, (unsigned long long)valid, (unsigned long long)n);
    }

    // --- Taxa de DERIVAÇÃO COMPLETA: SHA256 -> EC -> hash160 -> endereço ---
    {
        uint64_t t = 0;
        std::string mk = genValidMinikey(rng, len, &t);
        uint64_t n = 0;
        auto t0 = clk::now();
        while (std::chrono::duration<double>(clk::now() - t0).count() < secs) {
            for (int r = 0; r < 256; r++) { volatile auto a = addressFromMinikey(mk); (void)a; n++; }
        }
        double dt = std::chrono::duration<double>(clk::now() - t0).count();
        printf("  Derivação completa (EC + hash160):  %.1f k keys/s\n", n / dt / 1e3);
        printf("  -> a multiplicação de ponto (EC) é o gargalo; é ela que limita a busca real.\n");
    }
    return 0;
}

static void usage(const char *p)
{
    printf("Uso:\n");
    printf("  %s selftest                 teste de corretude (vetores conhecidos)\n", p);
    printf("  %s gen [N] [len]            gera N minikeys válidas (len 22/26/30, padrão 22)\n", p);
    printf("  %s verify <minikey>         valida e deriva o endereço de uma minikey\n", p);
    printf("  %s bench [segundos] [len]   mede triagem e derivação na CPU (padrão 3s, len 22)\n", p);
}

int main(int argc, char **argv)
{
    if (argc < 2) { usage(argv[0]); return 1; }
    std::string cmd = argv[1];
    if (cmd == "selftest") return selftest();
    if (cmd == "gen")      return gen(argc > 2 ? atoi(argv[2]) : 1, argc > 3 ? atoi(argv[3]) : 22);
    if (cmd == "verify")   { if (argc < 3) { usage(argv[0]); return 1; } return verify(argv[2]); }
    if (cmd == "bench")    return bench(argc > 2 ? atof(argv[2]) : 3.0, argc > 3 ? atoi(argv[3]) : 22);
    usage(argv[0]);
    return 1;
}
