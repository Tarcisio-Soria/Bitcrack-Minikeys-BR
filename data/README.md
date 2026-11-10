# data/ — conjuntos de dados da demonstração

> ⚠️ **Leia antes de usar.** Este diretório **não** contém chaves de moedas reais com fundos. O propósito do projeto é demonstrar que a busca é **inviável**.

## `our-minikeys-s1.txt` — chaves de TESTE (nosso gabarito)
Minikeys Casascius S1 **geradas por nós**, com `minikey`, `chave_privada` e `endereço`. **Não possuem fundos.** Como conhecemos a minikey, usamos este conjunto para provar que o buscador funciona em espaço reduzido (modos `search` e `run`), sem jamais tocar em propriedade alheia.

## `targets-ours.txt` — endereços dos NOSSOS alvos de teste
Apenas os endereços correspondentes às minikeys acima. É o alvo legítimo da demonstração de corretude.

## `targets-s1.txt` — endereços PÚBLICOS de moedas S1 não resgatadas (read-only)
Lista de **endereços** (nunca chaves privadas) de moedas Casascius Série 1 ainda lacradas, de origem comunitária. Incluída **exclusivamente** como conjunto-alvo *read-only* para dimensionar a demonstração de inviabilidade (ex.: bloom filter de comparação).

**Estes endereços NÃO são alvo de nenhuma busca real neste projeto.** As moedas pertencem a colecionadores; encontrar a chave de uma delas seria furto — e, como todo este repositório demonstra, é inviável (espaço de ~2¹¹⁵). São dados públicos de referência, equivalentes a qualquer endereço visível na blockchain.
