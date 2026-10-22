"""
Avaliação do retrieval sobre os 4 PDFs, sem Supabase e sem LLM.

Mede, para perguntas com nota esperada, se a nota certa aparece em 1º (hit@1) e entre
os 5 primeiros (hit@5), comparando a busca só vetorial com a busca híbrida.
Para comparar modelos de embedding:
    python avaliar_retrieval.py
    set EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 && python avaliar_retrieval.py   (Windows cmd)
"""
from data_pipeline import DocumentProcessor, OUTPUT_DIR, rank_local

TOP_K = 5
THRESHOLD = 0.35

# (pergunta, arquivo esperado). None = a resposta não existe na base.
PERGUNTAS = [
    ("Qual o valor total da nota do pet shop?", "NF_001_Gatinhos_Petshop.pdf"),
    ("Quanto custou a ração para gatos?", "NF_001_Gatinhos_Petshop.pdf"),
    ("Quem comprou o arranhador temático?", "NF_001_Gatinhos_Petshop.pdf"),
    ("Qual empresa tem o CNPJ 98.765.432/0001-21?", "NF_002_Servicos_Tech_Rag.pdf"),
    ("Quanto foi cobrado pela consultoria em arquitetura RAG?", "NF_002_Servicos_Tech_Rag.pdf"),
    ("Qual o endereço da Tech Corp?", "NF_002_Servicos_Tech_Rag.pdf"),
    ("Qual o preço da fantasia de gatinho preto?", "NF_003_Loja_Fantasias_Halloween.pdf"),
    ("Em que cidade fica a loja Fantasias & Feitiços?", "NF_003_Loja_Fantasias_Halloween.pdf"),
    ("Quanto custou o monitor ultrawide?", "NF_004_Equipamentos_Laboratorio.pdf"),
    ("Qual a nota 000.004.404?", "NF_004_Equipamentos_Laboratorio.pdf"),
    ("Qual a data de emissão da nota de equipamentos de laboratório?", "NF_004_Equipamentos_Laboratorio.pdf"),
    ("Qual o CPF da gatinha?", None),
    ("Quanto custou a passagem aérea para Paris?", None),
]


def busca_so_vetorial(query_embedding, chunks):
    """Comportamento antigo: só cosseno acima do threshold, sem score lexical."""
    # Pergunta vazia: sem termos, o score lexical é 0 e só o threshold de cosseno decide
    ranked = rank_local("", query_embedding, chunks, len(chunks), THRESHOLD)
    return sorted(ranked, key=lambda c: c["similarity"], reverse=True)[:TOP_K]


def main():
    processor = DocumentProcessor()
    _, chunks = processor.process_directory(OUTPUT_DIR)
    print(f"Modelo: {processor.model_name} | {len(chunks)} chunks | top_k={TOP_K} | threshold={THRESHOLD}\n")

    placar = {"vetorial": [0, 0], "híbrida": [0, 0]}
    positivas = sum(1 for _, esperado in PERGUNTAS if esperado)

    for pergunta, esperado in PERGUNTAS:
        q_emb = processor.generate_embeddings([pergunta])[0]
        resultados = {
            "vetorial": busca_so_vetorial(q_emb, chunks),
            "híbrida": rank_local(pergunta, q_emb, chunks, TOP_K, THRESHOLD),
        }
        print(f"? {pergunta}  (esperado: {esperado or 'nada'})")
        for nome, res in resultados.items():
            docs = [r["document_name"] for r in res]
            if esperado:
                placar[nome][0] += bool(docs) and docs[0] == esperado
                placar[nome][1] += esperado in docs
                marca = "1º" if docs and docs[0] == esperado else ("top5" if esperado in docs else "ERROU")
            else:
                marca = f"{len(docs)} trechos (a LLM precisa recusar)" if docs else "vazio, recusa direta"
            topo = f"{docs[0]} cos={res[0]['similarity']:.3f}" if res else "-"
            print(f"   {nome:9s} {marca:32s} topo: {topo}")
        print()

    for nome, (h1, h5) in placar.items():
        print(f"{nome:9s} hit@1 = {h1}/{positivas}   hit@5 = {h5}/{positivas}")


if __name__ == "__main__":
    main()
