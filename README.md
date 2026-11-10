# Bitcrack-Minikeys-BR

**Prova de conceito acadêmica da INVIABILIDADE de quebrar por força bruta as *minikeys* Casascius Série 1.**

Adaptação do [BitCrack](https://github.com/brichard19/BitCrack) (via [BitCrack-BR](https://github.com/Tarcisio-Soria/BitCrack-BR)) para o esquema de *minikey* da Casascius Série 1, construída como material de **Trabalho de Conclusão de Curso (Ciência da Computação)**. O objetivo **não** é encontrar chaves de moedas reais — é **medir** o custo da busca em hardware real e demonstrar, com números, que mesmo aplicando todas as otimizações cabíveis o problema permanece intratável.

> ⚠️ **Escopo ético.** Este projeto **não** busca as chaves das moedas Casascius ainda lacradas (propriedade de colecionadores). Toda a demonstração de corretude usa **chaves de teste geradas por nós** (`data/our-minikeys-s1.txt`, sem fundos). Veja [Ética](#ética).

---

## A tese

Uma *minikey* Casascius Série 1 tem 22 caracteres: `S` + 21 caracteres base58. A chave privada é derivada por

```
chave_privada = SHA256(minikey)
```

e há um dígito verificador: a *minikey* só é válida se `SHA256(minikey + "?")` começa com o byte `0x00` (passa ~1 em 256).

- **Espaço bruto de candidatos:** 58²¹ ≈ 2¹²³.
- **Espaço de *minikeys* válidas** (após o filtro `0x00`): ≈ **2¹¹⁵**.

O ponto central: como `priv = SHA256(minikey)` é **pseudoaleatória**, candidatos consecutivos produzem chaves privadas **sem nenhuma relação entre si**. Isso **destrói o truque incremental `P → P + G`** que é a principal otimização de GPU do BitCrack para chaves sequenciais. Cada chave exige uma **multiplicação escalar completa** na curva.

O filtro `0x00` só reduz o **custo por candidato** (descarta antes da parte cara), **não o espaço** de 2¹¹⁵.

---

## Arquitetura do pipeline

| Etapa | O que faz | Observação |
|-------|-----------|------------|
| **Triagem** | `SHA256(minikey + "?")` → testa byte `0x00` | barata (só hash); descarta 255/256 |
| **Derivação** | `priv = SHA256(minikey)` → `P = priv·G` → `hash160(P)` → compara com o alvo | **o gargalo**: multiplicação de ponto EC completa |

O fluxo separa triagem (fase A) e derivação (fase B) por meio de um **buffer** de candidatos válidos, para evitar divergência de warp na GPU (threads que não passam no filtro não ficam paradas esperando as que fazem EC).

---

## As otimizações (e por que não dá para fazer melhor)

### Implementadas em `minikey_gpu_opt.cu`

1. **Comb de base fixa.** Como `G` é sempre o mesmo ponto, pré-computamos uma tabela `T[i][j] = (j·2⁸ⁱ)·G` (32 janelas × 256 dígitos, 512 KB, gerada na própria GPU pela rotina ingênua já validada ⇒ **bit-exata por construção**). Cada `k·G` passa a custar **≤ 31 somas de ponto e ZERO duplicações** (contra 256 duplicações do método ingênuo).
2. **Inversão em lote (truque de Montgomery).** A conversão Jacobiano→afim exige uma inversão modular, que aqui é via Fermat (~385 multiplicações de campo por chave!). Processando `BATCHM` chaves por *thread*, fazemos **1 inversão + 3(M−1) multiplicações** no lugar de M inversões.

### Deliberadamente **não** implementadas — e o porquê

- **Incremental `P + G`** (o truque do BitCrack): exige chaves **consecutivas**. `SHA256` as embaralha ⇒ inaplicável.
- **GLV (endomorfismo λ/β):** acelera base **variável**. Num comb de base **fixa**, ele apenas troca "1 comb de 256 bits" por "2 combs de 128 bits" = **mesmo número de somas** ⇒ sem ganho, só complexidade. (ver cabeçalho de `src/minikey_gpu_opt.cu`)
- **wNAF:** subsumido pelo comb (não há duplicações a otimizar).
- **Pollard Kangaroo / BSGS:** resolvem o log discreto num **intervalo conhecido**. As chaves de *minikey* estão espalhadas por **todo** o 2²⁵⁶ (não há intervalo pequeno), e exigiriam a chave **pública** do alvo — mas só temos o endereço, e `hash160` é de mão única.

**Conclusão metodológica:** as otimizações atacam a **taxa**, nunca o **espaço** (2¹¹⁵).

---

## Resultados medidos (NVIDIA RTX 3060 Laptop, 6 GB, CUDA 12.9)

| Pipeline | Taxa (derivação completa) | Ganho |
|----------|---------------------------|-------|
| Ingênuo (`minikey_gpu_full`, double-and-add) | ~1,20 M keys/s | 1× |
| **Otimizado** (`minikey_gpu_opt`, comb + lote) | **~11,3 M keys/s** (picos ~11,6) | **~9,4×** |
| Triagem isolada (só SHA256 + filtro) | ~1,7 G cand/s | — |

Validação **bit-exata**: o modo `validate` cruza o caminho otimizado com o ingênuo e com vetores conhecidos (`G`, `2G`, `hash160(G) = 91b24bf9…`, `SHA256`, `RIPEMD160`) — resultado `IDENTICO`.

### Extrapolação para 2¹¹⁵

| Cenário | Taxa | Tempo para varrer 2¹¹⁵ |
|---------|------|------------------------|
| Otimizado (medido) | 11,5 M/s | **~1,15 × 10²⁰ anos** (~8 bilhões de idades do universo) |
| Ingênuo | 2,1 M/s | ~6,3 × 10²⁰ anos |
| Fantasia 10¹² keys/s | 1 T/s | ~1,3 × 10¹⁵ anos |

Ganhar 9,4× — ou 9,4 milhões de vezes — **não muda a conclusão**. O gargalo não é o algoritmo; é a entropia.

---

## Compilação

**Requisitos:** CUDA ≥ 12.8 (o `sm_120`/Blackwell exige 12.8+), compilador C++ compatível (g++ ≤ 14), driver NVIDIA. NVML (vem no toolkit CUDA) para o painel ao vivo.

```bash
# Build rápido — use a arquitetura da SUA GPU (ex.: RTX 30xx = sm_86):
make all SM="-gencode=arch=compute_86,code=sm_86"

# Build multi-arquitetura (Maxwell sm_52 → Blackwell sm_120) — mais lento:
make all
```

Outras variáveis: `CUDA_HOME` (default `/usr/local/cuda-12.9`), `CCBIN` (default `g++-14`), `BATCHM` (chaves por lote na inversão, default 8).

## Uso

```bash
# Referência CPU (corretude; vetores conhecidos da Casascius)
./minikey selftest
./minikey gen 10                 # gera 10 minikeys de teste (gabarito)

# Triagem GPU (SHA256 + filtro 0x00)
./minikey_gpu -b 120 -t 256 -p 8192

# Pipeline completo — INGÊNUO vs OTIMIZADO (compare as taxas)
./minikey_gpu_full bench -b 240 -t 256 -p 64
./minikey_gpu_opt  bench -b 240 -t 256 -p 64

# Validação bit-exata do caminho otimizado
./minikey_gpu_opt validate

# Demonstração de corretude: busca em espaço REDUZIDO (acha uma chave NOSSA)
./minikey_gpu_opt search SVXbZMHek7Eetag8ssc7H8 4

# Painel ao vivo contra NOSSOS alvos (nunca acha → prova 2^115). Ctrl+C para parar.
./minikey_gpu_opt run -b 120 -t 256 -p 2048
```

Rode os modos que leem dados (`run`) a partir da **raiz do projeto** (eles abrem `data/our-minikeys-s1.txt`).

---

## Estrutura

```
Bitcrack-Minikeys-BR/
├── src/            # código do PoC (nosso)
│   ├── minikey.cpp           referência CPU
│   ├── minikey_gpu.cu        triagem GPU
│   ├── minikey_gpu_full.cu   pipeline completo ingênuo
│   └── minikey_gpu_opt.cu    pipeline otimizado (comb + inversão em lote)
├── data/           # minikeys de teste e conjuntos-alvo (ver data/README.md)
├── docs/           # figuras (painel.svg) e material de apoio
├── bitcrack/       # módulos criptográficos reutilizados do BitCrack (MIT)
└── Makefile        # build unificado
```

---

## Ética

As ~4907 moedas Casascius Série 1 ainda lacradas pertencem a colecionadores. **Encontrar a chave de uma delas seria furto — e, como este projeto demonstra, é inviável (2¹¹⁵).** Por isso:

- A demonstração de corretude usa **chaves de teste geradas por nós** (`data/our-minikeys-s1.txt`, sem fundos), cujos endereços estão em `data/targets-ours.txt`.
- `data/targets-s1.txt` contém **endereços públicos** (não chaves) de moedas S1 não resgatadas, usados **apenas** como conjunto-alvo *read-only* da demonstração de inviabilidade — **nunca** como alvo de uma busca real. Veja `data/README.md`.

Este repositório é material didático: sua tese é que a busca **não funciona**.

---

## Créditos e licença

- **Base criptográfica:** [BitCrack](https://github.com/brichard19/BitCrack) de Ben Richard (MIT), via [BitCrack-BR](https://github.com/Tarcisio-Soria/BitCrack-BR) (fork multi-arquitetura, sm_52→sm_120, CUDA 12.9).
- **Adaptação para minikeys e otimizações:** Tarcisio Soria (KKND), 2026.

Licença **MIT** (ver [LICENSE.MIT](LICENSE.MIT)). Para citar, veja [CITATION.cff](CITATION.cff).
