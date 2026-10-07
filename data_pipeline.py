import os
import re
import json
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

class DocumentProcessor:
    """
    Pipeline de Ingestão, Extração de Texto (OCR/PDF),
    Estruturação Tabular, Chunking e Geração de Embeddings com FastEmbed.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
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
        emit_cnpj = re.search(r"CNPJ:\s*([0-9\.\/\-]+)", text)
        
        dest_nome = re.search(r"Nome/Raz[ãa]o:\s*([^\n\r]+)", text)
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
        c1_text = (
            f"Nota Fiscal: {invoice_info['numero_nota']} (Série {invoice_info['serie']}). "
            f"Data de Emissão: {invoice_info['data_emissao']}. "
            f"Chave de Acesso: {invoice_info['chave_acesso']}. "
            f"Emitente: {invoice_info['emitente']} (CNPJ: {invoice_info['emitente_cnpj']}). "
            f"Destinatário: {invoice_info['destinatario']}. "
            f"Valor Total da Nota: {invoice_info['valor_total']}."
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
            f"Destinatário/Cliente: {invoice_info['destinatario']}. "
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

        # Chunk 3: Conteúdo Integral extraído do documento (Itens e Valores)
        c3_text = f"Detalhamento completo e itens da NF {invoice_info['numero_nota']}:\n{full_text}"
        chunks.append({
            "document_name": filename,
            "chunk_index": 2,
            "content": c3_text,
            "metadata": {
                "tipo": "detalhamento_itens",
                "numero_nota": invoice_info["numero_nota"],
                "valor_total": invoice_info["valor_total"]
            }
        })

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
        cache_file = os.path.join(os.path.dirname(__file__), "chunks_cache.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
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
        Busca semântica vetorial:
        1. Gera embedding da query com FastEmbed
        2. Executa busca vetorial no Supabase (via RPC match_documents)
        3. Se tabela Supabase não estiver ativada, utiliza similaridade de cosseno local sobre os chunks
        """
        query_embedding = self.generate_embeddings([query])[0]
        results = []

        # Tenta consulta no Supabase RPC
        if self.supabase_client:
            try:
                rpc_response = self.supabase_client.rpc("match_documents", {
                    "query_embedding": query_embedding,
                    "match_threshold": match_threshold,
                    "match_count": top_k
                }).execute()

                if rpc_response.data and len(rpc_response.data) > 0:
                    for row in rpc_response.data:
                        results.append({
                            "id": row.get("id"),
                            "document_name": row.get("document_name"),
                            "chunk_index": row.get("chunk_index"),
                            "content": row.get("content"),
                            "metadata": row.get("metadata", {}),
                            "similarity": float(row.get("similarity", 0.0))
                        })
                    return results
            except Exception as e:
                print(f"[!] RPC Supabase match_documents não disponível ou falhou: {e}. Usando busca local.")

        # Fallback local sobre o cache de chunks
        cache_file = os.path.join(os.path.dirname(__file__), "chunks_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_chunks = json.load(f)

                q_vec = np.array(query_embedding)
                scored = []
                for chunk in cached_chunks:
                    c_vec = np.array(chunk["embedding"])
                    # Cosine similarity
                    norm_q = np.linalg.norm(q_vec)
                    norm_c = np.linalg.norm(c_vec)
                    if norm_q > 0 and norm_c > 0:
                        sim = float(np.dot(q_vec, c_vec) / (norm_q * norm_c))
                    else:
                        sim = 0.0

                    if sim >= match_threshold:
                        scored.append({
                            "document_name": chunk["document_name"],
                            "chunk_index": chunk["chunk_index"],
                            "content": chunk["content"],
                            "metadata": chunk.get("metadata", {}),
                            "similarity": round(sim, 4)
                        })

                # Ordena pelos mais similares
                scored.sort(key=lambda x: x["similarity"], reverse=True)
                results = scored[:top_k]
            except Exception as e:
                print(f"[!] Erro no fallback local: {e}")

        return results

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
