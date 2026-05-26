**Relatorio Tecnico  
RAG para EPUBs e Livros Digitais**

Do iniciante ao avancado, com arquitetura, tecnicas, avaliacao e plano de implementacao em Python

Escopo calibrado para: 100 a 1.000 EPUBs, chatbot, busca semantica/lexical, resumo, respostas com citacoes e risco medio.

Data: 26/05/2026  
Versao: 1.0

# Sumario de secoes

- 1\. Sumario executivo e diagnostico objetivo

- 2\. Parte iniciante - o que e RAG e como ele funciona

- 3\. Parte intermediaria - recuperacao, chunking, embeddings, reranking e geracao

- 4\. Parte avancada - arquitetura modular, hierarquica, contextual e avaliavel

- 5\. RAG de livros/EPUBs - melhores tecnicas para conversar com livros

- 6\. Arquitetura recomendada em Python

- 7\. Plano de implementacao e checklist de qualidade

- 8\. Referencias

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>Premissa central</strong><br />
Para livros, RAG bom nao e apenas banco vetorial. E um sistema de recuperacao estrutural: precisa preservar livro, capitulo, secao, paragrafo, entidade, janela de contexto e fonte. Se o texto vira chunks planos, a resposta tende a ficar rasa, incompleta ou errada mesmo com embedding, BM25 e reranking.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

# 1. Sumario executivo e diagnostico objetivo

RAG, ou Retrieval-Augmented Generation, e uma arquitetura em que o modelo de linguagem gera respostas usando conhecimento externo recuperado no momento da pergunta. A ideia original formalizada por Lewis et al. combina memoria parametrica do modelo com memoria nao parametrica externa, normalmente um indice de passagens recuperaveis \[1\]. Na pratica moderna, RAG virou o padrao para sistemas que precisam responder com base em documentos privados, livros, manuais, contratos, bases tecnicas ou conteudo que muda com frequencia.

Para o seu caso - 100 a 1.000 EPUBs, uso em Python, respostas conversacionais, busca de trechos e resumos - a conclusao tecnica e direta: se voce ja usa busca semantica + lexical + reranking e ainda considera fraco, o gargalo provavelmente nao esta em “falta de RAG”, mas em uma ou mais destas camadas: ingestao estrutural ruim, chunking plano, ausencia de contexto hierarquico, pouco recall antes do reranker, reranking sobre o texto errado, falta de expansao de janela/pai, prompt sem politica forte de citacao e ausencia de avaliacao offline.

| **Sintoma**                                         | **Causa provavel**                                                       | **Correcao superior**                                                                            |
|-----------------------------------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Resposta generica apesar de haver trecho no livro   | Chunk perdeu titulo/capitulo/secao; embedding nao enxerga o assunto real | Contextual chunking + metadados no texto indexado + parent/section expansion                     |
| Busca acha o livro certo, mas nao o paragrafo certo | Reranking recebe candidatos ruins ou top-k inicial baixo                 | Candidate pool alto: BM25 top 100 + dense top 100 + summary top 20; fusao RRF; rerank top 50-100 |
| Resumo de capitulo fica incompleto                  | Sistema recupera passagens locais, nao a estrutura inteira               | Indice de sumarios por capitulo/secao + retrieval hierarquico + map-reduce/refine                |
| Respostas com citacoes fracas                       | Fonte nao tem localizador estavel                                        | Criar citacao canonica: livro, capitulo, secao, paragrafo/CFI/offset                             |
| Perguntas comparativas falham                       | RAG trata pergunta como lookup simples                                   | Query planner: decompor pergunta, buscar por subconsultas e sintetizar com evidencias separadas  |

## Recomendacao principal

Implemente um Book-aware RAG: um RAG consciente da estrutura do livro. A arquitetura deve usar pelo menos quatro indices: passagem/paragrafo, secao/capitulo, sumario hierarquico e entidades/termos. O fluxo recomendado e: extrair EPUB preservando spine e sumario; montar arvore livro \> parte \> capitulo \> secao \> bloco; gerar chunks filhos pequenos; indexar tambem janelas/pais; aplicar contextualizacao por chunk; combinar BM25, dense e eventualmente sparse vectors; usar RRF; rerank; expandir contexto para a secao pai; comprimir por relevancia; gerar resposta com citacoes e abstencao quando a evidencia for insuficiente.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>Decisao objetiva</strong><br />
Nao recomendo tentar resolver a fraqueza com fine-tuning primeiro. Fine-tuning pode melhorar estilo, formato e comportamento, mas nao substitui recuperacao correta. Para livros, o salto de qualidade vem antes de tudo de ingestao, retrieval, estrutura, avaliacao e citacao.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## Prioridades em ordem

1.  **Preservar estrutura do EPUB:** spine, sumario, titulos, hierarquia, notas, imagens, tabelas e localizadores.

2.  **Contextualizar chunks:** cada chunk deve carregar livro, autor, capitulo, secao e um micro-resumo contextual antes do embedding e do BM25.

3.  **Usar retrieval hierarquico:** buscar em chunks pequenos, mas responder com janela ou secao pai maior.

4.  **Aumentar recall antes do reranking:** reranking nao salva se o candidato correto nao entrou no pool.

5.  **Avaliar com golden set:** sem Recall@k, nDCG/MRR, faithfulness e auditoria de citacao, voce fica ajustando por sensacao.

6.  **Separar tipos de pergunta:** lookup, resumo, comparacao, opiniao baseada em fonte e pergunta multi-hop exigem rotas diferentes.

# 2. Parte iniciante - o que e RAG e como ele funciona

## 2.1 Definicao simples

RAG e uma estrategia para fazer um LLM responder usando documentos externos. Em vez de confiar apenas no que o modelo “aprendeu” no treinamento, o sistema primeiro busca trechos relevantes e depois entrega esses trechos ao modelo para que ele gere a resposta. A resposta ideal nao nasce apenas da memoria do modelo; ela nasce da combinacao entre pergunta, evidencias recuperadas e instrucoes de geracao.

A definicao tecnica moderna costuma dividir o sistema em tres blocos: retrieval, augmentation e generation. O survey de Gao et al. organiza a evolucao do campo em Naive RAG, Advanced RAG e Modular RAG \[2\]. Essa divisao e util porque mostra por que um RAG basico, feito apenas com chunking + embedding + top-k, costuma falhar quando a base e grande ou hierarquica.

| **Camada** | **Funcao**                                          | **Risco se for mal feita**                                          |
|------------|-----------------------------------------------------|---------------------------------------------------------------------|
| Ingestao   | Extrair e limpar conteudo dos livros                | Texto quebrado, ordem errada, perda de capitulos e notas            |
| Chunking   | Dividir conteudo em unidades recuperaveis           | Trechos sem contexto, cortes no meio da ideia, perda de referencias |
| Indexacao  | Criar representacoes para busca lexical e semantica | Busca nao encontra termos exatos ou conceitos equivalentes          |
| Retrieval  | Selecionar candidatos relevantes                    | Contexto errado entra no prompt                                     |
| Reranking  | Reordenar candidatos por aderencia a pergunta       | Trechos superficiais vencem trechos corretos                        |
| Generation | Responder com base nas evidencias                   | Alucinacao, extrapolacao ou resposta sem citacao                    |
| Evaluation | Medir qualidade e regressao                         | Sistema parece melhorar, mas piora em casos importantes             |

## 2.2 Por que nao usar apenas o LLM

Um LLM isolado e bom em linguagem, raciocinio aproximado e sintese, mas nao tem garantia de lembrar fielmente um livro especifico, nem de citar pagina, secao ou trecho. RAG reduz esse problema porque injeta informacao recuperada no momento da resposta. Isso nao elimina alucinacao por si so: se a recuperacao for fraca, o modelo pode responder bem escrito sobre evidencia ruim. Por isso, em producao, RAG deve ser tratado como um problema de engenharia de recuperacao de informacao antes de ser um problema de prompt.

## 2.3 Pipeline basico

| **Etapa**               | **Entrada**                   | **Saida**                                            |
|-------------------------|-------------------------------|------------------------------------------------------|
| 1\. Carregar livro      | Arquivo EPUB                  | HTMLs, metadados, imagens, sumario, ordem de leitura |
| 2\. Limpar e normalizar | HTML/CSS/texto bruto          | Blocos textuais estruturados                         |
| 3\. Chunking            | Capitulos, secoes, paragrafos | Chunks com metadados                                 |
| 4\. Embeddings/BM25     | Texto dos chunks              | Vetores densos e indice lexical                      |
| 5\. Pergunta do usuario | Mensagem + historico          | Consulta reformulada e classificada                  |
| 6\. Recuperar           | Consulta                      | Lista de trechos candidatos                          |
| 7\. Rerank e expandir   | Candidatos                    | Contexto final com janela/pai                        |
| 8\. Gerar               | Pergunta + contexto           | Resposta com citacoes                                |

## 2.4 Conceitos fundamentais

- **Embedding:** vetor numerico que representa significado. Ajuda a encontrar textos semanticamente proximos mesmo sem as mesmas palavras.

- **BM25/busca lexical:** busca por termos, frequencia e raridade. E forte para nomes, siglas, citacoes, termos tecnicos e frases exatas.

- **Chunk:** unidade recuperavel. Em livros, chunk bom normalmente nao e um pedaco arbitrario de N caracteres; e um bloco ligado a secao e ao capitulo.

- **Reranker:** modelo de segunda etapa que avalia par pergunta-documento e reordena candidatos com mais precisao. Cross-encoders sao comuns para isso \[13\].

- **Grounding:** obrigar a resposta a se apoiar no contexto recuperado, citando fonte e recusando quando nao houver evidencia.

- **Recall:** capacidade de trazer o trecho correto para a lista de candidatos. Sem recall, reranking e prompt nao resolvem.

# 3. Parte intermediaria - recuperacao, chunking, embeddings, reranking e geracao

## 3.1 Recuperacao lexical, semantica e hibrida

A busca lexical e excelente quando a pergunta contem termos que aparecem literalmente no livro. A busca semantica e melhor quando a pergunta usa linguagem diferente do texto-fonte. Em bases de livros, nenhuma das duas e suficiente isoladamente. O benchmark BEIR mostrou que BM25 e uma linha de base robusta em varios dominios, enquanto reranking e late-interaction tendem a ter desempenho alto, com maior custo computacional \[4\].

Por isso, a abordagem superior e busca hibrida: gerar candidatos por BM25 e por embeddings, fundir rankings e depois reranquear. Ferramentas como Qdrant e Weaviate documentam suporte a buscas hibridas, fusao e pesos entre busca lexical e vetorial \[18\]\[19\].

| **Metodo**               | **Forca**                                      | **Fraqueza**                                  | **Uso recomendado em livros**       |
|--------------------------|------------------------------------------------|-----------------------------------------------|-------------------------------------|
| BM25                     | Termo exato, nomes, siglas, frases             | Nao entende sinonimos bem                     | Sempre usar como um dos retrievers  |
| Dense embeddings         | Semantica e sinonimia                          | Pode perder termos raros e detalhes numericos | Usar para perguntas conceituais     |
| Sparse embeddings/SPLADE | Combina expansao lexical com busca esparsa     | Infra e custo podem aumentar                  | Bom para dominio tecnico            |
| Hybrid + RRF             | Melhora recall combinando listas               | Requer ajuste de pool e deduplicacao          | Padrao recomendado                  |
| Cross-encoder rerank     | Alta precisao local                            | Mais lento; custo cresce com docs x tokens    | Rerank top 50-100 candidatos        |
| ColBERT/late interaction | Mais preciso que bi-encoder em muitos cenarios | Infra mais complexa                           | Avaliar se qualidade for prioridade |

## 3.2 Reciprocal Rank Fusion

RRF combina rankings diferentes usando a posicao dos documentos, nao a escala bruta dos scores. Isso e util porque score de BM25 e score de similaridade vetorial nao sao diretamente comparaveis. Em termos praticos, voce pode recuperar top 100 por BM25, top 100 por dense, top 50 por sumario, e fundir por RRF antes do reranker.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>def reciprocal_rank_fusion(result_lists, k=60):<br />
# result_lists: lista de listas ordenadas; cada item precisa ter id unico de chunk/documento<br />
scores = {}<br />
for results in result_lists:<br />
for rank, item in enumerate(results, start=1):<br />
scores[item.id] = scores.get(item.id, 0.0) + 1.0 / (k + rank)<br />
return sorted(scores.items(), key=lambda x: x[1], reverse=True)</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 3.3 Chunking: onde a maioria dos RAGs de livros quebra

Chunking fixo por caracteres/tokens e simples, mas costuma destruir estrutura narrativa e argumentativa. Em livros, uma frase como “isso demonstra a tese anterior” perde sentido se o chunk nao sabe qual e a tese anterior. O problema central e que embeddings de chunks curtos sao precisos, mas perdem contexto; chunks longos preservam contexto, mas diluem o vetor. Tecnicas recentes como contextual retrieval e late chunking atacam exatamente essa perda de contexto \[11\]\[7\].

| **Estrategia**      | **Quando usar**                   | **Risco**                        | **Recomendacao**                       |
|---------------------|-----------------------------------|----------------------------------|----------------------------------------|
| Chunk fixo          | Prototipo rapido                  | Corta ideias e ignora hierarquia | Evitar como estrategia final           |
| Chunk por paragrafo | Busca precisa por trechos         | Pode ficar curto demais          | Usar como filho                        |
| Chunk por secao     | Respostas explicativas            | Pode ser longo                   | Usar como pai/contexto expandido       |
| Sentence window     | Perguntas pontuais                | Precisa boa segmentacao          | Recuperar frase e responder com janela |
| Parent-child        | Livros e manuais                  | Mais complexidade de storage     | Padrao recomendado                     |
| Contextual chunking | Chunks sem contexto proprio       | Custo na ingestao                | Muito recomendado                      |
| Late chunking       | Modelos de embedding long-context | Dependente de modelo/infra       | Avaliar em benchmark proprio           |

## 3.4 Embeddings

Nao existe embedding universalmente melhor para todo corpus. O MTEB mostrou que modelos de embedding variam por tarefa, idioma e tipo de avaliacao \[5\]. Logo, para livros, a decisao correta nao e escolher “o melhor embedding do ranking” de forma abstrata, mas comparar 3 a 5 opcoes no seu proprio conjunto de perguntas. Em APIs comerciais, embeddings da OpenAI continuam sendo uma escolha forte e simples; a documentacao atual informa vetores de 1536 dimensoes para text-embedding-3-small e 3072 para text-embedding-3-large por padrao \[12\].

| **Criterio**   | **Como avaliar**                                                            |
|----------------|-----------------------------------------------------------------------------|
| Idioma         | Portuguese/English/multilingue; teste perguntas no idioma real dos usuarios |
| Domino         | Tecnico, literario, juridico, religioso ou academico                        |
| Granularidade  | Paragrafo, secao, capitulo ou sumario                                       |
| Custo          | Preco por milhao de tokens + custo de reindexacao                           |
| Latencia       | Tempo de embed na ingestao e tempo de query                                 |
| Qualidade real | Recall@10/20/50 no seu golden set                                           |

## 3.5 Reranking

Reranking e uma segunda etapa de precisao. O retriever inicial deve ser barato e amplo; o reranker deve ser mais caro e seletivo. A documentacao de SentenceTransformers descreve a arquitetura de retrieve-and-rerank: primeiro um retriever eficiente gera candidatos; depois um Cross-Encoder recebe query e documento juntos e pontua relevancia \[13\]. Pinecone tambem descreve reranking como etapa de dois estagios para melhorar a qualidade dos resultados \[20\].

Erro comum: reranquear apenas 5 ou 10 chunks. Isso e top-k baixo demais para uma biblioteca grande. O correto e recuperar bastante antes: por exemplo, 100 BM25 + 100 dense + 20 sumarios, deduplicar, fundir e reranquear top 50-100. O limite final para o LLM pode ser 6 a 12 evidencias, mas o pool anterior precisa ser maior.

## 3.6 Geracao com citacoes

A geracao deve ser restringida por politica de evidencia. Para respostas sobre livros, o prompt deve exigir: usar apenas os trechos fornecidos; citar livro/capitulo/secao/paragrafo; separar inferencia de citacao direta; declarar insuficiencia quando os trechos nao sustentarem a resposta; nao preencher lacunas com conhecimento externo sem sinalizar.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>Politica de resposta recomendada:<br />
1. Responda somente com base nas evidencias fornecidas.<br />
2. Cite toda afirmacao material no formato [Livro &gt; Capitulo &gt; Secao &gt; bloco_id].<br />
3. Se a evidencia for insuficiente, diga exatamente o que falta.<br />
4. Nao misture livros diferentes sem indicar a fonte de cada ponto.<br />
5. Diferencie: 'o livro afirma', 'o trecho sugere' e 'minha inferencia'.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

# 4. Parte avancada - arquitetura modular, hierarquica, contextual e avaliavel

## 4.1 Naive RAG, Advanced RAG e Modular RAG

Naive RAG e o fluxo basico: chunk, embed, top-k, prompt. Advanced RAG adiciona pre-retrieval e post-retrieval: limpeza, query rewriting, filtros, reranking, compressao e avaliacao. Modular RAG vai alem: usa multiplos retrievers, roteadores, agentes, memoria, ferramentas, decomposicao de consulta, grafos e fluxos adaptativos. Essa classificacao aparece no survey de Gao et al. e e uma boa forma de pensar a evolucao do seu sistema \[2\].

| **Nivel**        | **Arquitetura**                                                 | **Qualidade esperada em livros**                                   |
|------------------|-----------------------------------------------------------------|--------------------------------------------------------------------|
| Naive            | Chunk fixo + vector top-k + prompt                              | Fraca a media; funciona apenas em perguntas simples                |
| Advanced         | Hybrid retrieval + rerank + filtros + citacao                   | Boa para QA local e busca de trechos                               |
| Hierarchical     | Child chunks + parent sections + sumarios                       | Superior para livros e manuais                                     |
| Modular/Agentic  | Planner, decomposicao, multiplos indices, ferramentas           | Superior para perguntas comparativas, resumos e multi-hop          |
| Graph/Book-aware | Hierarquia + entidades + relacoes + retrieval por granularidade | Melhor quando o corpus tem estrutura forte e conceitos recorrentes |

## 4.2 Query planner

O sistema nao deve tratar toda pergunta igual. “O que o livro diz sobre X?” nao e igual a “resuma o capitulo 3”, “compare dois autores” ou “onde aparece o termo Y?”. Um query planner classifica a pergunta e escolhe a rota de retrieval.

| **Tipo de pergunta**            | **Rota recomendada**                                                    |
|---------------------------------|-------------------------------------------------------------------------|
| Lookup factual                  | BM25 forte + dense + rerank + resposta curta com citacao                |
| Conceitual                      | Dense + contextual chunks + parent expansion                            |
| Resumo de capitulo              | Indice de secao/capitulo + sumario hierarquico + map-reduce/refine      |
| Comparacao entre livros         | Subconsultas por livro + evidencias separadas + sintese comparativa     |
| Pergunta ampla sobre biblioteca | Book-level summaries + Graph/Book index + topicos/entidades             |
| Pergunta com historico          | Reescrita da query usando historico, mantendo filtros de livro/capitulo |

## 4.3 Retrieval hierarquico e auto-merging

O padrao child-to-parent e especialmente forte para livros. Voce recupera chunks pequenos, porque eles sao mais precisos para busca, mas entrega ao gerador uma janela ou secao pai, porque ela contem coerencia. LlamaIndex documenta o AutoMergingRetriever usando uma hierarquia de nos em varios tamanhos, como 2048, 512 e 128 tokens por nivel, com links entre filhos e pais \[8\].

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>Regra pratica</strong><br />
Recupere pequeno, responda com contexto maior. O chunk pequeno serve para localizar. A secao pai serve para explicar.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 4.4 Contextual retrieval

Contextual retrieval adiciona contexto sintetico a cada chunk antes de indexar. Em vez de indexar apenas “Ele resolve esse problema em tres passos”, o sistema indexa algo como “Livro X, capitulo Y, secao Z: este trecho explica o metodo de avaliacao do RAG... Ele resolve esse problema em tres passos”. A Anthropic reportou reducao de falhas de recuperacao de 49%, e 67% quando combinado com reranking, em seu artigo sobre Contextual Retrieval \[11\].

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>Prompt de contextualizacao por chunk:<br />
Voce recebera o livro inteiro ou a secao pai e um chunk.<br />
Escreva 1-3 frases de contexto que situem o chunk no livro.<br />
Inclua tema, capitulo/secao, entidades principais e funcao do trecho.<br />
Nao resuma demais; nao acrescente informacao externa.<br />
Retorne apenas o contexto.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 4.5 HyDE, multi-query e decomposicao

HyDE gera um documento hipotetico a partir da pergunta e usa o embedding desse documento para recuperar textos semelhantes. A proposta original de Gao et al. busca melhorar dense retrieval zero-shot quando nao ha labels de relevancia \[10\]. Para livros, HyDE pode ajudar quando o usuario pergunta de forma vaga, mas deve ser usado com cuidado: ele aumenta custo e pode enviesar a busca para uma resposta imaginada. Use como rota opcional, nao como padrao universal.

Multi-query gera varias consultas alternativas para a mesma pergunta, por sinonimos, termos tecnicos, entidades e subperguntas. Em livros, e util para autores que usam vocabulario diferente. O risco e trazer ruido; por isso, precisa de fusao, deduplicacao, reranking e filtros por livro/capitulo quando existirem.

## 4.6 GraphRAG e BookRAG

GraphRAG usa grafos de entidades, relacoes e comunidades para responder perguntas globais sobre grandes corpus. O trabalho “From Local to Global” descreve construcao de grafo, hierarquia de comunidades e sumarios para perguntas de sentido global \[22\]. Para livros, isso e util quando a pergunta exige relacoes entre conceitos, personagens, autores, capitulos ou documentos. Nao deve substituir a busca por trecho exato; deve complementar.

A linha BookRAG vai na direcao correta para o seu caso: explorar hierarquia logica de documentos com estrutura de livro, usando uma arvore semelhante ao sumario e relacoes de entidades \[23\]. Mesmo que voce nao implemente o paper integralmente, o principio e decisivo: livros nao devem ser indexados como um monte de chunks independentes.

## 4.7 Avaliacao como parte da arquitetura

Sem avaliacao, voce nao sabe se melhorou ou apenas mudou o erro. RAGAS propoe metricas para avaliar retrieval e geracao sem depender sempre de anotacao humana \[14\]. ARES tambem avalia contexto, fidelidade e relevancia com dados sinteticos e pequena quantidade de anotacao humana \[15\]. TruLens popularizou a triade contexto relevante, groundedness e resposta relevante \[21\].

| **Metrica**               | **O que mede**                                        | **Meta inicial razoavel**      |
|---------------------------|-------------------------------------------------------|--------------------------------|
| Recall@20                 | Se algum trecho correto aparece entre os 20 primeiros | \>= 85% em golden set          |
| MRR@10                    | Quao cedo aparece o primeiro trecho correto           | Aumentar por iteracao          |
| nDCG@10                   | Qualidade da ordenacao com relevancia graduada        | Comparar pipelines             |
| Context precision         | Proporcao de contexto realmente util                  | Evitar prompt inflado          |
| Faithfulness/groundedness | Se a resposta e sustentada pelas evidencias           | Alto; qualquer queda e grave   |
| Citation accuracy         | Se a citacao aponta para o trecho certo               | \>= 95% em perguntas auditadas |
| Abstention accuracy       | Se o sistema sabe dizer “nao encontrei evidencia”     | Medir em perguntas impossiveis |
| Latency p95               | Tempo de resposta em percentil 95                     | Definir por produto            |

# 5. RAG de livros/EPUBs - melhores tecnicas para conversar com livros

## 5.1 Por que EPUB e diferente de PDF

EPUB e essencialmente um container de conteudo web estruturado. A especificacao W3C define EPUB como formato para distribuir conteudo web estruturado e semanticamente enriquecido em um unico arquivo \[16\]. Isso e uma vantagem para RAG: diferentemente de muitos PDFs, o EPUB ja traz HTML, metadados, recursos, ordem de leitura, sumario e marcacoes. Se voce extrair EPUB como texto bruto, esta desperdicando a parte mais valiosa do formato.

A documentacao do EbookLib indica suporte a EPUB2/EPUB3, metadados, TOC, spine, guide e outros componentes \[17\]. Para citacao precisa, EPUB CFI define uma forma padronizada de referenciar conteudo arbitrario dentro de uma publicacao EPUB \[18\].

## 5.2 Pipeline ideal de ingestao de EPUB

| **Etapa**                    | **Acao tecnica**                                                                | **Resultado esperado**               |
|------------------------------|---------------------------------------------------------------------------------|--------------------------------------|
| 1\. Validar arquivo          | Abrir EPUB, checar metadados, detectar idioma e encoding                        | Documento identificavel e rastreavel |
| 2\. Ler package/spine        | Usar ordem de leitura oficial, nao ordem de arquivos no zip                     | Texto na sequencia correta           |
| 3\. Extrair TOC/nav          | Mapear capitulos, secoes e anchors                                              | Arvore hierarquica do livro          |
| 4\. Parsear HTML             | BeautifulSoup/lxml; preservar h1-h6, p, blockquote, li, table, footnote         | Blocos semanticos                    |
| 5\. Limpar ruido             | Remover CSS/scripts, cabecalhos repetidos, paginas de copyright se irrelevantes | Texto limpo sem perder fonte         |
| 6\. Criar IDs                | book_id, chapter_id, section_id, block_id, char offsets e/ou CFI                | Citacao estavel                      |
| 7\. Enriquecer metadados     | Titulo, autor, capitulo, secao, topicos, entidades, resumo local                | Filtros e busca melhores             |
| 8\. Chunking hierarquico     | Filhos pequenos + pais grandes + janelas                                        | Busca precisa e resposta coerente    |
| 9\. Indexar multiplas visoes | Raw text, contextual text, summary, keywords/entities                           | Maior recall                         |
| 10\. Avaliar                 | Golden set por livro e por tipo de pergunta                                     | Melhoria mensuravel                  |

## 5.3 Modelo de dados recomendado

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>Book<br />
- book_id, title, author, language, publisher, year, source_hash, rights<br />
<br />
Node<br />
- node_id, book_id, parent_id, level<br />
- level: book | part | chapter | section | subsection | paragraph | sentence<br />
- title_path: [Livro, Parte, Capitulo, Secao]<br />
- text_raw, text_clean, text_contextual<br />
- cfi, href, anchor, char_start, char_end<br />
- summary_short, keywords, entities<br />
<br />
Chunk<br />
- chunk_id, node_id, parent_node_id, book_id<br />
- chunk_text, contextual_text, token_count<br />
- dense_vector, sparse_vector/BM25 field<br />
- citation_label, version, created_at</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 5.4 Estrategia de chunking superior para livros

A configuracao abaixo e um ponto de partida tecnico. Os numeros devem ser ajustados com avaliacao, mas a arquitetura e robusta.

| **Nivel**       | **Tamanho inicial**         | **Uso**                                               |
|-----------------|-----------------------------|-------------------------------------------------------|
| Sentence/bloco  | 1-3 frases ou 80-180 tokens | Busca extremamente precisa; nao enviar sozinho ao LLM |
| Child chunk     | 180-450 tokens              | Principal unidade de embedding e BM25                 |
| Parent section  | 800-2.000 tokens            | Contexto expandido para resposta                      |
| Chapter summary | 150-400 tokens              | Busca por perguntas amplas e roteamento               |
| Book summary    | 300-800 tokens              | Selecao de livros e perguntas globais                 |

## 5.5 Tecnicas que mais aumentam qualidade em livros

7.  **Contextual headers no texto indexado:** prefixar cada chunk com titulo do livro, autor, capitulo, secao e funcao do trecho. Isso melhora BM25 e embedding.

8.  **Parent-child retrieval:** buscar em chunks menores e expandir para secao pai antes de gerar resposta.

9.  **Indice de sumarios:** criar resumos de capitulos e secoes para perguntas amplas, resumos e selecao de area.

10. **Multi-index retrieval:** passage index, section index, chapter summary index e entity index.

11. **Entity-aware retrieval:** extrair personagens, conceitos, termos, autores, leis, tecnicas ou entidades e mapear para secoes.

12. **Query routing:** classificar pergunta em lookup, resumo, comparacao, explicacao, citacao exata ou global.

13. **Evidence packing:** montar contexto final com diversidade, ordem logica e deduplicacao, nao simplesmente top-k.

14. **Citations first:** gerar citacoes canonicas desde a ingestao; nao tentar inventar pagina depois.

15. **Abstencao:** quando o sistema nao encontra evidencia suficiente, responder isso claramente.

16. **Golden set:** medir por tipo de pergunta e por livro; sem isso, ajustes viram opiniao.

## 5.6 Por que seu RAG hibrido + reranking pode estar fraco

| **Problema**                        | **Teste rapido**                                           | **Correcao**                                        |
|-------------------------------------|------------------------------------------------------------|-----------------------------------------------------|
| Chunk sem contexto                  | Leia um chunk isolado. Ele faz sentido sem capitulo/secao? | Contextual retrieval + title_path no texto indexado |
| Top-k inicial baixo                 | O trecho correto aparece no top 100 antes do reranker?     | Aumentar candidate pool e usar RRF                  |
| Reranker recebe texto curto demais  | Ele ve apenas frase solta?                                 | Rerank com chunk contextualizado ou janela curta    |
| Resposta usa contexto errado        | O LLM recebeu trechos demais ou fora de ordem?             | Evidence packing + compressao por consulta          |
| Pergunta ampla tratada como pontual | Resumo de capitulo busca 5 chunks aleatorios?              | Rota de sumario hierarquico                         |
| Sem filtro de livro/capitulo        | Consulta sobre um livro retorna outro?                     | Metadados e filtros inferidos pelo query planner    |
| Citations ruins                     | A citacao aponta para local estavel?                       | CFI/anchor/block_id e quote offsets                 |

## 5.7 Conversar com livro: memoria de conversa correta

Em chatbot, o historico nao deve ser simplesmente concatenado ao prompt. O correto e reescrever a pergunta atual com base no historico, preservando filtros. Exemplo: se o usuario pergunta “e no capitulo seguinte?”, a query final precisa virar “No livro X, capitulo posterior ao capitulo Y, qual e a explicacao sobre Z?”. Depois disso, o retrieval roda sobre a query reescrita, nao sobre a frase curta.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>Entrada:<br />
Usuario: O que o autor diz sobre disciplina no capitulo 2?<br />
Assistente: ...<br />
Usuario: E no capitulo seguinte?<br />
<br />
Query reescrita:<br />
No mesmo livro, buscar no capitulo 3 a continuidade do tema 'disciplina' discutido no capitulo 2.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

# 6. Arquitetura recomendada em Python

## 6.1 Stack recomendada

| **Componente**      | **Opcao recomendada**                                                                   | **Justificativa**                                                  |
|---------------------|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Extracao EPUB       | EbookLib + BeautifulSoup/lxml                                                           | Acesso a spine, TOC, HTML e metadados                              |
| Normalizacao        | Python pipeline proprio                                                                 | Controle fino sobre blocos, IDs e limpeza                          |
| Banco relacional    | PostgreSQL                                                                              | Metadados, versoes, usuarios, permissoes e auditoria               |
| Vector/hybrid store | Qdrant ou Weaviate; Postgres/pgvector se simplicidade for prioridade                    | Qdrant/Weaviate tem recursos hibridos fortes; Postgres reduz infra |
| BM25                | Elasticsearch/OpenSearch, Qdrant sparse, Weaviate hybrid ou Postgres FTS/BM25 extension | Lexical e obrigatorio para livros                                  |
| Embeddings          | Benchmark entre OpenAI, BGE-M3, multilingual-e5, Jina                                   | Escolher por teste no corpus, nao por ranking generico             |
| Reranker            | Cross-encoder local ou API Cohere/Jina/Pinecone                                         | Melhora precisao final                                             |
| Orquestracao        | Pipeline proprio ou LlamaIndex para acelerar                                            | Evitar caixa-preta demais em producao                              |
| Avaliacao           | RAGAS/TruLens + scripts proprios                                                        | Medir retrieval, citacao, fidelidade e latencia                    |

## 6.2 Fluxo de ingestao - pseudocodigo

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>from ebooklib import epub<br />
from bs4 import BeautifulSoup<br />
<br />
book = epub.read_epub(path)<br />
metadata = extract_metadata(book)<br />
spine_items = resolve_spine_in_reading_order(book)<br />
toc_tree = extract_toc_tree(book)<br />
<br />
for item in spine_items:<br />
html = item.get_content()<br />
soup = BeautifulSoup(html, 'lxml')<br />
blocks = html_to_semantic_blocks(soup) # h1-h6, p, li, blockquote, table, footnote<br />
blocks = attach_toc_path(blocks, toc_tree)<br />
blocks = normalize_and_deduplicate(blocks)<br />
save_blocks_with_stable_ids(blocks)<br />
<br />
nodes = build_hierarchy(book_id) # book &gt; chapter &gt; section &gt; paragraph<br />
chunks = build_child_parent_chunks(nodes)<br />
chunks = generate_contextual_headers(chunks)<br />
embed_and_index(chunks)<br />
create_chapter_and_section_summaries(nodes)<br />
index_summaries_and_entities(nodes)</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 6.3 Fluxo de consulta - pseudocodigo

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>def answer(question, chat_history=None, filters=None):<br />
rewritten = rewrite_query(question, chat_history)<br />
route = classify_query(rewritten) # lookup | summary | comparison | global | quote<br />
subqueries = plan_subqueries(rewritten, route)<br />
<br />
candidates = []<br />
for q in subqueries:<br />
candidates += bm25_search(q, top_k=100, filters=filters)<br />
candidates += dense_search(q, top_k=100, filters=filters)<br />
candidates += summary_search(q, top_k=20, filters=filters)<br />
<br />
fused = rrf_fuse(candidates)<br />
diverse = dedupe_and_diversify(fused, by=['book_id', 'section_id'])<br />
reranked = cross_encoder_rerank(rewritten, diverse[:100])<br />
evidence = expand_to_parent_sections(reranked[:12])<br />
evidence = compress_context_by_query(rewritten, evidence)<br />
<br />
return generate_grounded_answer(<br />
question=question,<br />
rewritten_query=rewritten,<br />
evidence=evidence,<br />
citation_policy='strict'<br />
)</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 6.4 Evidence packing

Evidence packing e a montagem inteligente do contexto final. O objetivo nao e colocar os top 10 chunks cegamente. O objetivo e entregar ao LLM um pacote pequeno, diverso, ordenado e suficiente.

- Ordenar trechos por livro \> capitulo \> secao quando a pergunta pede explicacao sequencial.

- Manter diversidade: nao mandar 8 chunks quase iguais da mesma secao se 3 bastam.

- Expandir chunks filhos para uma janela/pai, mas remover partes irrelevantes por compressao.

- Adicionar citacao canonica antes de cada evidencia.

- Separar evidencias por subpergunta em perguntas comparativas.

## 6.5 Prompt de geracao recomendado

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>Voce e um assistente que responde com base em uma biblioteca de livros.<br />
<br />
Regras obrigatorias:<br />
- Use somente as evidencias fornecidas.<br />
- Toda afirmacao substantiva deve ter citacao.<br />
- Cite no formato [titulo &gt; capitulo &gt; secao &gt; bloco_id].<br />
- Se a pergunta pedir resumo, sintetize sem inventar topicos ausentes.<br />
- Se houver conflito entre fontes, mostre o conflito.<br />
- Se a evidencia for insuficiente, diga: 'Nao encontrei evidencia suficiente nos trechos recuperados'.<br />
- Nao use conhecimento externo sem marcar explicitamente como conhecimento externo.<br />
<br />
Pergunta original: {question}<br />
Pergunta reescrita para busca: {rewritten_query}<br />
Evidencias:<br />
{evidence_pack}<br />
<br />
Resposta:</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 6.6 Esquema SQL simplificado

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th>CREATE TABLE books (<br />
book_id UUID PRIMARY KEY,<br />
title TEXT NOT NULL,<br />
author TEXT,<br />
language TEXT,<br />
source_hash TEXT UNIQUE,<br />
metadata JSONB,<br />
created_at TIMESTAMP DEFAULT now()<br />
);<br />
<br />
CREATE TABLE nodes (<br />
node_id UUID PRIMARY KEY,<br />
book_id UUID REFERENCES books(book_id),<br />
parent_id UUID NULL,<br />
level TEXT NOT NULL,<br />
title TEXT,<br />
title_path TEXT[],<br />
text_clean TEXT,<br />
summary_short TEXT,<br />
href TEXT,<br />
cfi TEXT,<br />
char_start INT,<br />
char_end INT,<br />
metadata JSONB<br />
);<br />
<br />
CREATE TABLE chunks (<br />
chunk_id UUID PRIMARY KEY,<br />
node_id UUID REFERENCES nodes(node_id),<br />
parent_node_id UUID REFERENCES nodes(node_id),<br />
book_id UUID REFERENCES books(book_id),<br />
chunk_text TEXT NOT NULL,<br />
contextual_text TEXT NOT NULL,<br />
citation_label TEXT NOT NULL,<br />
token_count INT,<br />
metadata JSONB<br />
);</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

# 7. Plano de implementacao e checklist de qualidade

## 7.1 Roadmap recomendado

| **Fase** | **Objetivo**                     | **Entregavel**                                                              |
|----------|----------------------------------|-----------------------------------------------------------------------------|
| Fase 1   | Diagnosticar pipeline atual      | Golden set com 100-200 perguntas; baseline com Recall@20, MRR, faithfulness |
| Fase 2   | Reescrever ingestao EPUB         | Arvore livro \> capitulo \> secao \> bloco com IDs e citacoes               |
| Fase 3   | Implementar chunking hierarquico | Child chunks, parent sections, sentence windows e metadados                 |
| Fase 4   | Criar multi-index                | Passage index, section summary, chapter summary, entity index               |
| Fase 5   | Melhorar retrieval               | BM25 + dense + RRF + rerank top 50-100 + parent expansion                   |
| Fase 6   | Geracao e citacao                | Prompt estrito, evidence packing, abstencao e quote validation              |
| Fase 7   | Observabilidade                  | Logs por query: candidatos, scores, citacoes, custo, latencia, feedback     |
| Fase 8   | Otimizacao                       | A/B de embeddings, top-k, chunk size, reranker, contextual retrieval        |

## 7.2 Checklist de ingestao

- O texto segue a ordem do spine do EPUB, nao a ordem fisica dos arquivos.

- Cada bloco possui book_id, chapter_id, section_id, block_id e caminho de titulo.

- Notas de rodape, citacoes, listas, tabelas e imagens com alt text sao tratadas separadamente.

- O pipeline remove ruido sem apagar conteudo autoral relevante.

- Cada chunk tem versao, hash e localizador estavel.

- O texto usado para embedding inclui contexto suficiente, nao apenas o paragrafo solto.

- O texto usado para BM25 tambem inclui metadados contextuais uteis.

## 7.3 Checklist de retrieval

- BM25 e dense rodam em paralelo.

- O pool inicial e amplo: pelo menos 50-100 candidatos por retriever em corpus grande.

- Ha fusao por ranking, preferencialmente RRF, antes do reranker.

- Reranker recebe texto contextualizado e nao apenas fragmentos sem titulo.

- O sistema expande candidatos vencedores para pai/janela antes da resposta.

- Perguntas amplas usam indices de sumario, nao apenas chunks locais.

- Perguntas de conversa sao reescritas antes do retrieval.

## 7.4 Checklist de avaliacao

- Criar golden set com perguntas de lookup, resumo, comparacao, multi-hop e perguntas impossiveis.

- Medir Recall@10/20/50 antes do reranker.

- Medir nDCG/MRR depois da fusao e depois do reranker.

- Avaliar citacao: a fonte apontada realmente contem a afirmacao?

- Avaliar abstencao: sistema recusa quando nao ha evidencia?

- Registrar custo, latencia e numero de tokens por resposta.

- Rodar regressao sempre que mudar chunking, embedding, top-k ou prompt.

## 7.5 Configuracao inicial sugerida

| **Parametro**     | **Valor inicial**          | **Ajuste esperado**                                                     |
|-------------------|----------------------------|-------------------------------------------------------------------------|
| Child chunk       | 250-400 tokens             | Aumentar se contexto estiver fragmentado; reduzir se busca ficar difusa |
| Overlap           | 10-15% ou por paragrafo    | Evitar overlap cego alto que duplica resultados                         |
| BM25 top-k        | 100                        | Reduzir se latencia pesar; aumentar em corpus heterogeneo               |
| Dense top-k       | 100                        | Testar 50/100/200                                                       |
| Summary top-k     | 20                         | Usar para perguntas amplas                                              |
| Rerank pool       | 50-100                     | Depende de custo/latencia                                               |
| Final evidence    | 6-12 blocos expandidos     | Depende da janela do modelo e tipo de pergunta                          |
| Contextual header | 1-3 frases                 | Gerar na ingestao e guardar versionado                                  |
| Eval set          | 100-200 perguntas iniciais | Expandir para 500+ em producao                                          |

## 7.6 Ordem de experimentos recomendada

17. Baseline atual: medir sem mudar nada.

18. Adicionar citacoes e IDs de bloco se ainda nao existir.

19. Trocar chunking plano por parent-child sem mudar embedding.

20. Adicionar contextual headers e reindexar.

21. Aumentar candidate pool e aplicar RRF.

22. Rerank top 50-100 com texto contextualizado.

23. Adicionar indice de resumos de capitulo/secao.

24. Adicionar query planner e rotas por tipo de pergunta.

25. Comparar embeddings e rerankers com o mesmo golden set.

26. Adicionar Graph/Entity retrieval apenas se perguntas multi-hop/relacionais justificarem.

## 7.7 Conclusao tecnica

Para RAG de livros, a abordagem superior nao e “mais embedding” nem “mais prompt”. E preservar a estrutura do livro, recuperar em multiplas granularidades, reranquear com candidatos suficientes, responder com contexto expandido e medir tudo. Um RAG de EPUBs bem feito deve se comportar menos como um chatbot solto e mais como um leitor tecnico com indice, marcadores, memoria de capitulo e disciplina de citacao.

A sua stack em Python deve evoluir para um pipeline observavel: cada resposta precisa deixar rastro de query reescrita, rota escolhida, candidatos recuperados, scores, evidencia final, citacoes, custo, latencia e avaliacao. Esse e o ponto em que RAG deixa de ser prototipo e vira sistema confiavel.

# 8. Referencias principais

\[1\] Lewis, P. et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS/arXiv, 2020. [<u>https://arxiv.org/abs/2005.11401</u>](https://arxiv.org/abs/2005.11401)

\[2\] Gao, Y. et al. Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv, 2023/2024. [<u>https://arxiv.org/abs/2312.10997</u>](https://arxiv.org/abs/2312.10997)

\[3\] Karpukhin, V. et al. Dense Passage Retrieval for Open-Domain Question Answering. EMNLP, 2020. [<u>https://aclanthology.org/2020.emnlp-main.550/</u>](https://aclanthology.org/2020.emnlp-main.550/)

\[4\] Thakur, N. et al. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS Datasets and Benchmarks, 2021. [<u>https://openreview.net/forum?id=wCu6T5xFjeJ</u>](https://openreview.net/forum?id=wCu6T5xFjeJ)

\[5\] Muennighoff, N. et al. MTEB: Massive Text Embedding Benchmark. EACL/arXiv, 2022/2023. [<u>https://arxiv.org/abs/2210.07316</u>](https://arxiv.org/abs/2210.07316)

\[6\] Khattab, O.; Zaharia, M. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR/arXiv, 2020. [<u>https://arxiv.org/abs/2004.12832</u>](https://arxiv.org/abs/2004.12832)

\[7\] Gunther, M. et al. Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models. arXiv, 2024. [<u>https://arxiv.org/abs/2409.04701</u>](https://arxiv.org/abs/2409.04701)

\[8\] LlamaIndex. Auto Merging Retriever documentation. [<u>https://developers.llamaindex.ai/python/framework/integrations/retrievers/auto_merging_retriever/</u>](https://developers.llamaindex.ai/python/framework/integrations/retrievers/auto_merging_retriever/)

\[9\] LlamaIndex. Metadata Replacement and Sentence Window postprocessor documentation. [<u>https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/node_postprocessors/</u>](https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/node_postprocessors/)

\[10\] Gao, L. et al. Precise Zero-Shot Dense Retrieval without Relevance Labels / HyDE. arXiv, 2022. [<u>https://arxiv.org/abs/2212.10496</u>](https://arxiv.org/abs/2212.10496)

\[11\] Anthropic. Contextual Retrieval in AI Systems. 2024. [<u>https://www.anthropic.com/news/contextual-retrieval</u>](https://www.anthropic.com/news/contextual-retrieval)

\[12\] OpenAI. Vector embeddings documentation. [<u>https://developers.openai.com/api/docs/guides/embeddings</u>](https://developers.openai.com/api/docs/guides/embeddings)

\[13\] SentenceTransformers. Retrieve & Re-Rank Pipeline documentation. [<u>https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html</u>](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)

\[14\] Es, S. et al. RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv/EACL demo, 2023/2024. [<u>https://arxiv.org/abs/2309.15217</u>](https://arxiv.org/abs/2309.15217)

\[15\] Saad-Falcon, J. et al. ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. arXiv/NAACL, 2023/2024. [<u>https://arxiv.org/abs/2311.09476</u>](https://arxiv.org/abs/2311.09476)

\[16\] W3C. EPUB 3.4 specification. 2026. [<u>https://www.w3.org/TR/epub-34/</u>](https://www.w3.org/TR/epub-34/)

\[17\] EbookLib. Documentation. [<u>https://docs.sourcefabric.org/projects/ebooklib/en/latest/</u>](https://docs.sourcefabric.org/projects/ebooklib/en/latest/)

\[18\] IDPF/W3C. EPUB Canonical Fragment Identifiers 1.1. [<u>https://idpf.org/epub/linking/cfi/</u>](https://idpf.org/epub/linking/cfi/)

\[19\] Qdrant. Hybrid Queries documentation. [<u>https://qdrant.tech/documentation/search/hybrid-queries/</u>](https://qdrant.tech/documentation/search/hybrid-queries/)

\[20\] Weaviate. Hybrid search documentation. [<u>https://docs.weaviate.io/weaviate/search/hybrid</u>](https://docs.weaviate.io/weaviate/search/hybrid)

\[21\] Pinecone. Rerank results documentation. [<u>https://docs.pinecone.io/guides/search/rerank-results</u>](https://docs.pinecone.io/guides/search/rerank-results)

\[22\] TruLens. RAG Triad documentation. [<u>https://www.trulens.org/getting_started/core_concepts/rag_triad/</u>](https://www.trulens.org/getting_started/core_concepts/rag_triad/)

\[23\] Edge, D. et al. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv, 2024. [<u>https://arxiv.org/abs/2404.16130</u>](https://arxiv.org/abs/2404.16130)

\[24\] Wang, S. et al. BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents. arXiv, 2025. [<u>https://arxiv.org/abs/2512.03413</u>](https://arxiv.org/abs/2512.03413)

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>Observacao sobre referencias</strong><br />
As referencias foram usadas para fundamentar conceitos, arquitetura e decisoes tecnicas. Para implementacao final, mantenha uma matriz de avaliacao propria: a qualidade real do RAG depende do corpus, idioma, perguntas e requisitos de citacao.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>
