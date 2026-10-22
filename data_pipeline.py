import os
import re
import json
import math
import textwrap
import unicodedata
from typing import List, Dict, Any, Tuple
import pypdf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_PUBLISHABLE_KEY")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dados_notas_fiscais")
CACHE_FILE = os.path.join(os.path.dirname(__file__), "chunks_cache.json")

# Modelo multilíngue (as notas e as perguntas são em português). Tem 384 dimensões,
# então a tabela vector(384) do Supabase continua valendo. Ele lê até 128 tokens,
# por isso os trechos do texto integral são curtos (CHUNK_MAX_CHARS).
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CHUNK_MAX_CHARS = 300
CHUNK_OVERLAP_CHARS = 80

# Busca híbrida: candidatos vetoriais reordenados por RRF com um score lexical
RRF_K = 60
LEXICAL_MIN_SCORE = 0.5
STOPWORDS = {
    "qual", "quais", "que", "quem", "para", "com", "das", "dos", "uma", "umas", "uns", "nao",
    "nota", "notas", "fiscal", "fiscais", "valor", "sobre", "foi", "como", "onde", "quando",
    "esta", "essa", "esse", "este", "pela", "pelo", "tem", "sao", "nas", "nos", "mais", "entre",
    "item", "itens", "por", "seu", "sua", "meu", "minha",
}


def normalize_text(text: str) -> str:
    """Minúsculas, sem acentos, e números com separador (CNPJ, nº da nota) juntos num token só."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"(?<=\d)[.\-/](?=\d)", "", text)


def lexical_terms(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalize_text(text)) if len(t) >= 3 and t not in STOPWORDS]


def split_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap_chars: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """Divide o texto em trechos de até max_chars, quebrando por linha e repetindo o fim do trecho anterior."""
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            lines.extend(textwrap.wrap(line, max_chars) if len(line) > max_chars else [line])

    pieces: List[str] = []
    current: List[str] = []
    for line in lines:
        if current and len(" ".join(current + [line])) > max_chars:
            pieces.append("\n".join(current))
            # Sobreposição: carrega as últimas linhas do trecho anterior que cabem em overlap_chars
            tail: List[str] = []
            for prev in reversed(current):
                if len(" ".join([prev] + tail)) > overlap_chars:
                    break
                tail.insert(0, prev)
            current = tail
        current.append(line)
    if current:
        pieces.append("\n".join(current))
    return pieces


def rank_hybrid(query: str, candidates: List[Dict[str, Any]], top_k: int, match_threshold: float) -> List[Dict[str, Any]]:
    """
    Reordena candidatos vetoriais combinando a similaridade de cosseno com um score lexical
    ponderado por IDF (Reciprocal Rank Fusion). Um candidato é aceito se passar no threshold
    vetorial OU se cobrir a maior parte dos termos da pergunta (nomes, CNPJ, nº da nota).
    """
    terms = set(lexical_terms(query))
    chunk_terms = [set(lexical_terms(c["content"])) for c in candidates]
    n = len(candidates)
    # Termo que não aparece em nenhum candidato pesa mais: "gatinha" ausente derruba o score
    idf = {}
    for t in terms:
        df = sum(t in ct for ct in chunk_terms)
        idf[t] = math.log(1 + (n - df + 0.5) / (df + 0.5))
    total_weight = sum(idf.values())

    for c, ct in zip(candidates, chunk_terms):
        c["lexical_score"] = round(sum(idf[t] for t in terms & ct) / total_weight, 4) if total_weight else 0.0

    by_vector = sorted(candidates, key=lambda c: c["similarity"], reverse=True)
    by_lexical = sorted((c for c in candidates if c["lexical_score"] > 0), key=lambda c: c["lexical_score"], reverse=True)
    hybrid = {id(c): 0.0 for c in candidates}
    for rank, c in enumerate(by_vector, 1):
        hybrid[id(c)] += 1 / (RRF_K + rank)
    for rank, c in enumerate(by_lexical, 1):
        hybrid[id(c)] += 1 / (RRF_K + rank)
    for c in candidates:
        c["hybrid_score"] = round(hybrid[id(c)], 6)

    accepted = [c for c in candidates if c["similarity"] >= match_threshold or c["lexical_score"] >= LEXICAL_MIN_SCORE]
    accepted.sort(key=lambda c: c["hybrid_score"], reverse=True)
    return accepted[:top_k]


def rank_local(query: str, query_embedding: List[float], chunks: List[Dict[str, Any]], top_k: int, match_threshold: float) -> List[Dict[str, Any]]:
    """Busca híbrida sobre chunks em memória (cache local ou avaliação), sem Supabase."""
    if not chunks:
        return []
    matrix = np.array([c["embedding"] for c in chunks], dtype=float)
    q_vec = np.array(query_embedding, dtype=float)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_vec)
    sims = np.divide(matrix @ q_vec, norms, out=np.zeros(len(chunks)), where=norms > 0)

    candidates = [
        {
            "document_name": c["document_name"],
            "chunk_index": c["chunk_index"],
            "content": c["content"],
            "metadata": c.get("metadata", {}),
            "similarity": round(float(s), 4),
        }
        for c, s in zip(chunks, sims)
    ]
    return rank_hybrid(query, candidates, top_k, match_threshold)


class DocumentProcessor:
    """
    Pipeline de Ingestão, Extração de Texto (OCR/PDF),
    Estruturação Tabular, Chunking e Geração de Embeddings com FastEmbed.
    """
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.embedding_model = TextEmbedding(model_name=self.model_name)
        self.supabase_client: Client = None
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            except Exception as e:
                print(f"[!] Aviso ao inicializar Supabase: {e}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrai texto bruto de um arquivo PDF."""
        reader = pypdf.PdfReader(pdf_path)
        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
        return "\n".join(full_text)

    def parse_invoice_fields(self, filename: str, text: str) -> Dict[str, Any]:
        """Extrai campos estruturados de Notas Fiscais a partir do texto extraído."""
        # Extrações com regex tolerante
        num_match = re.search(r"N[ºo°]:\s*([0-9\.]+)", text)
        serie_match = re.search(r"S[ÉE]RIE:\s*([0-9]+)", text, re.IGNORECASE)
        data_match = re.search(r"DATA DE EMISS[ÃA]O:\s*([0-9]{2}/[0-9]{2}/[0-9]{4})", text, re.IGNORECASE)
        chave_match = re.search(r"CHAVE DE ACESSO:\s*([0-9]{44})", text, re.IGNORECASE)
        
        emit_razao = re.search(r"Raz[ãa]o Social:\s*([^\n\r]+)", text)
        # "(?<!/)" evita casar o "CPF/CNPJ:" do destinatário
        emit_cnpj = re.search(r"(?<!/)CNPJ:\s*([0-9\.\/\-]+)", text)

        dest_nome = re.search(r"Nome/Raz[ãa]o:\s*([^\n\r]+)", text)
        dest_doc = re.search(r"CPF/CNPJ:\s*([0-9\.\/\-]+)", text)
        total_match = re.search(r"VALOR TOTAL DA NOTA:\s*([^\n\r]+)", text)
        obs_match = re.search(r"INFORMA[ÇC][ÕO]ES COMPLEMENTARES:\s*([^\n\r]+)", text)

        data_dict = {
            "arquivo": filename,
            "numero_nota": num_match.group(1).strip() if num_match else "N/A",
            "serie": serie_match.group(1).strip() if serie_match else "1",
            "data_emissao": data_match.group(1).strip() if data_match else "N/A",
            "chave_acesso": chave_match.group(1).strip() if chave_match else "N/A",
            "emitente": emit_razao.group(1).strip() if emit_razao else "N/A",
            "emitente_cnpj": emit_cnpj.group(1).strip() if emit_cnpj else "N/A",
            "destinatario": dest_nome.group(1).strip() if dest_nome else "N/A",
            "destinatario_documento": dest_doc.group(1).strip() if dest_doc else "N/A",
            "valor_total": total_match.group(1).strip() if total_match else "N/A",
            "observacoes": obs_match.group(1).strip() if obs_match else "N/A",
            "conteudo_completo": text
        }
        return data_dict

    def create_chunks(self, invoice_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Divide o documento em chunks semânticos didáticos com metadados detalhados.
        """
        filename = invoice_info["arquivo"]
        full_text = invoice_info["conteudo_completo"]
        
        chunks = []

        # Chunk 1: Resumo Geral e Identificação
        # A chave de acesso (44 dígitos, muitos tokens) fica no fim: se o modelo truncar, perde só ela
        c1_text = (
            f"Nota Fiscal: {invoice_info['numero_nota']} (Série {invoice_info['serie']}). "
            f"Data de Emissão: {invoice_info['data_emissao']}. "
            f"Emitente: {invoice_info['emitente']} (CNPJ: {invoice_info['emitente_cnpj']}). "
            f"Destinatário: {invoice_info['destinatario']}. "
            f"Valor Total da Nota: {invoice_info['valor_total']}. "
            f"Chave de Acesso: {invoice_info['chave_acesso']}."
        )
        chunks.append({
            "document_name": filename,
            "chunk_index": 0,
            "content": c1_text,
            "metadata": {
                "tipo": "resumo_cabecalho",
                "numero_nota": invoice_info["numero_nota"],
                "data_emissao": invoice_info["data_emissao"],
                "emitente": invoice_info["emitente"],
                "destinatario": invoice_info["destinatario"],
                "valor_total": invoice_info["valor_total"]
            }
        })

        # Chunk 2: Detalhes das Partes e Observações
        c2_text = (
            f"Partes Envolvidas na NF {invoice_info['numero_nota']}: "
            f"Emitente: {invoice_info['emitente']} (CNPJ: {invoice_info['emitente_cnpj']}). "
            f"Destinatário/Cliente: {invoice_info['destinatario']} (CPF/CNPJ: {invoice_info['destinatario_documento']}). "
            f"Informações complementares e observações da operação: {invoice_info['observacoes']}."
        )
        chunks.append({
            "document_name": filename,
            "chunk_index": 1,
            "content": c2_text,
            "metadata": {
                "tipo": "partes_e_observacoes",
                "numero_nota": invoice_info["numero_nota"],
                "emitente": invoice_info["emitente"],
                "destinatario": invoice_info["destinatario"]
            }
        })

        # Chunks 3+: Conteúdo Integral (Itens e Valores) em trechos curtos com sobreposição.
        # Um chunk só com o PDF inteiro misturava tudo num vetor e passaria dos 128 tokens do modelo.
        pieces = split_text(full_text)
        for i, piece in enumerate(pieces, 1):
            chunks.append({
                "document_name": filename,
                "chunk_index": len(chunks),
                "content": f"NF {invoice_info['numero_nota']} ({invoice_info['emitente']}), trecho {i}/{len(pieces)}:\n{piece}",
                "metadata": {
                    "tipo": "detalhamento_itens",
                    "numero_nota": invoice_info["numero_nota"],
                    "valor_total": invoice_info["valor_total"],
                    "trecho": i
                }
            })

        for c in chunks:
            c["metadata"]["embedding_model"] = self.model_name

        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings vetoriais com FastEmbed (modelo Sentence Transformers)."""
        embeddings_iter = self.embedding_model.embed(texts)
        return [emb.tolist() for emb in embeddings_iter]

    def process_directory(self, dir_path: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Executa o pipeline completo:
        1. Ingestão dos PDFs
        2. Extração de texto e estruturação tabular
        3. Criação de Chunks semânticos
        4. Geração de Embeddings com FastEmbed
        """
        pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]
        invoices_list = []
        all_chunks = []

        for pdf_name in pdf_files:
            file_path = os.path.join(dir_path, pdf_name)
            raw_text = self.extract_text_from_pdf(file_path)
            parsed_data = self.parse_invoice_fields(pdf_name, raw_text)
            invoices_list.append(parsed_data)

            doc_chunks = self.create_chunks(parsed_data)
            all_chunks.extend(doc_chunks)

        # Tabular Dataframe
        df_invoices = pd.DataFrame(invoices_list).drop(columns=["conteudo_completo"], errors="ignore")

        # Gerar embeddings para todos os chunks
        chunk_texts = [c["content"] for c in all_chunks]
        if chunk_texts:
            embeddings = self.generate_embeddings(chunk_texts)
            for i, emb in enumerate(embeddings):
                all_chunks[i]["embedding"] = emb

        return df_invoices, all_chunks

    def upload_to_supabase(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Faz o upload dos chunks e embeddings para o Supabase (tabela document_chunks).
        Mantém também backup local para redundância.
        """
        status_report = {
            "total_chunks": len(chunks),
            "supabase_uploaded": 0,
            "supabase_status": "pendente",
            "error_detail": None
        }

        # Salva cópia local (cache rápido)
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[!] Erro ao salvar cache local: {e}")

        if not self.supabase_client:
            status_report["supabase_status"] = "erro_credenciais"
            status_report["error_detail"] = "Cliente Supabase não configurado"
            return status_report

        try:
            # Prepara registros para inserção
            records = []
            for c in chunks:
                records.append({
                    "document_name": c["document_name"],
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                    "metadata": c["metadata"],
                    "embedding": c["embedding"]
                })

            # Tenta limpar registros antigos para evitar duplicidade no teste didático
            try:
                self.supabase_client.table("document_chunks").delete().neq("id", 0).execute()
            except Exception:
                pass

            # Inserção em lote
            response = self.supabase_client.table("document_chunks").insert(records).execute()
            status_report["supabase_uploaded"] = len(records)
            status_report["supabase_status"] = "sucesso"
        except Exception as e:
            status_report["supabase_status"] = "falha"
            status_report["error_detail"] = str(e)
            print(f"[!] Upload Supabase falhou: {e}")

        return status_report

    def search_vector_store(self, query: str, top_k: int = 5, match_threshold: float = 0.35) -> List[Dict[str, Any]]:
        """
        Busca híbrida:
        1. Gera embedding da query com FastEmbed
        2. Busca candidatos vetoriais no Supabase (via RPC match_documents), sem threshold
        3. Reordena os candidatos com o score lexical (RRF) e aplica o threshold em rank_hybrid
        4. Se o Supabase não estiver disponível, faz o mesmo sobre o cache local de chunks
        """
        query_embedding = self.generate_embeddings([query])[0]

        # Tenta consulta no Supabase RPC
        if self.supabase_client:
            try:
                # Pool maior que top_k: o threshold e o corte final ficam com a etapa híbrida
                rpc_response = self.supabase_client.rpc("match_documents", {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.0,
                    "match_count": max(top_k * 4, 20)
                }).execute()

                if rpc_response.data:
                    candidates = [
                        {
                            "id": row.get("id"),
                            "document_name": row.get("document_name"),
                            "chunk_index": row.get("chunk_index"),
                            "content": row.get("content"),
                            "metadata": row.get("metadata") or {},
                            "similarity": float(row.get("similarity", 0.0))
                        }
                        for row in rpc_response.data
                    ]
                    self._warn_if_model_mismatch(candidates)
                    return rank_hybrid(query, candidates, top_k, match_threshold)
            except Exception as e:
                print(f"[!] RPC Supabase match_documents não disponível ou falhou: {e}. Usando busca local.")

        # Fallback local sobre o cache de chunks
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    cached_chunks = json.load(f)
                self._warn_if_model_mismatch(cached_chunks)
                return rank_local(query, query_embedding, cached_chunks, top_k, match_threshold)
            except Exception as e:
                print(f"[!] Erro no fallback local: {e}")

        return []

    def _warn_if_model_mismatch(self, chunks: List[Dict[str, Any]]) -> None:
        """Vetores de outro modelo têm a mesma dimensão (384), mas comparar com eles dá resultado sem sentido."""
        stored = {(c.get("metadata") or {}).get("embedding_model") for c in chunks}
        if stored != {self.model_name}:
            print(f"[!] Base indexada com outro modelo de embedding ({stored}). Rode a ingestão de novo para usar '{self.model_name}'.")

if __name__ == "__main__":
    processor = DocumentProcessor()
    invoices_dir = os.path.join(os.path.dirname(__file__), "dados_notas_fiscais")
    print(f"Executando teste do pipeline em: {invoices_dir}")
    df, chunks = processor.process_directory(invoices_dir)
    print("\n--- DATAFRAME ESTRUTURADO ---")
    print(df.to_string())
    print(f"\nTotal de Chunks gerados: {len(chunks)}")
    print(f"Tamanho do vetor do chunk 0: {len(chunks[0]['embedding'])}")
    
    print("\n--- TESTE DE UPLOAD ---")
    upload_res = processor.upload_to_supabase(chunks)
    print("Resultado do upload:", upload_res)

    print("\n--- TESTE DE RETRIEVAL VETORIAL ---")
    test_query = "Qual é o valor e itens da nota da Gabriela de gatinho?"
    retrieved = processor.search_vector_store(test_query, top_k=3, match_threshold=0.3)
    for r in retrieved:
        print(f"Documento: {r['document_name']} | Similaridade: {r['similarity']:.4f}")
        print(f"Conteúdo: {r['content'][:150]}...\n")
