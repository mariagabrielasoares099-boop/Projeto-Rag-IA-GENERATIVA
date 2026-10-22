"""
Avaliação de ponta a ponta: retrieval híbrido local + prompt aumentado + LLM via OpenRouter.

Sem Supabase e sem backend: os chunks ficam em memória. Para cada pergunta, confere se a resposta
traz o dado esperado da nota, ou se aplica a regra antialucinação quando a resposta não existe.
Precisa de OPENROUTER_API_KEY no .env. Faz uma chamada à LLM por pergunta.
    python avaliar_llm.py
"""
import os

import httpx
from dotenv import load_dotenv

from data_pipeline import DocumentProcessor, OUTPUT_DIR, rank_local
from rag_prompt import NO_DATA_ANSWER, build_messages, is_no_data_answer

load_dotenv()

TOP_K = 5
THRESHOLD = 0.35

# (pergunta, trecho que a resposta precisa conter). None = a LLM tem de recusar.
CASOS = [
    ("Qual o valor total da nota do pet shop?", "475,00"),
    ("Quanto custou a ração para gatos?", "180,00"),
    ("Quem comprou o arranhador temático?", "Gabriela Almeida"),
    ("Qual empresa tem o CNPJ 98.765.432/0001-21?", "Rag Soluções"),
    ("Quanto foi cobrado pela consultoria em arquitetura RAG?", "4.500,00"),
    ("Qual o endereço da Tech Corp?", "Paulista"),
    ("Qual o preço da fantasia de gatinho preto?", "130,00"),
    ("Em que cidade fica a loja Fantasias & Feitiços?", "Curitiba"),
    ("Quanto custou o monitor ultrawide?", "2.400,00"),
    ("Qual a nota 000.004.404?", "Hardware"),
    ("Qual a data de emissão da nota de equipamentos de laboratório?", "31/10/2026"),
    ("Qual o CPF da gatinha?", None),
    ("Quanto custou a passagem aérea para Paris?", None),
]


def perguntar_llm(client: httpx.Client, query: str, chunks) -> str:
    resp = client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json={
            "model": os.getenv("OPENROUTER_MODEL", "openrouter/auto"),
            "messages": build_messages(query, chunks),
            "temperature": 0.0,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY não está no .env")

    processor = DocumentProcessor()
    _, chunks = processor.process_directory(OUTPUT_DIR)
    print(f"Modelo LLM: {os.getenv('OPENROUTER_MODEL', 'openrouter/auto')} | embedding: {processor.model_name}\n")

    acertos = 0
    with httpx.Client(timeout=60.0, headers={"Authorization": f"Bearer {key}"}) as client:
        for pergunta, esperado in CASOS:
            q_emb = processor.generate_embeddings([pergunta])[0]
            recuperados = rank_local(pergunta, q_emb, chunks, TOP_K, THRESHOLD)
            # Mesma regra do backend: sem trechos, recusa sem chamar a LLM
            resposta = perguntar_llm(client, pergunta, recuperados) if recuperados else NO_DATA_ANSWER

            if esperado is None:
                ok = is_no_data_answer(resposta)
            else:
                ok = esperado.lower() in resposta.lower() and not is_no_data_answer(resposta)
            acertos += ok
            print(f"{'OK  ' if ok else 'FALHOU'} {pergunta}  (esperado: {esperado or 'recusa'})")
            print(f"       {' '.join(resposta.split())[:300]}\n")

    print(f"{acertos}/{len(CASOS)} respostas corretas")


if __name__ == "__main__":
    main()
