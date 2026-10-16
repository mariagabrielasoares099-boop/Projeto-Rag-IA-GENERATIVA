import os
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv

from data_pipeline import DocumentProcessor, OUTPUT_DIR

load_dotenv()

app = FastAPI(
    title="RAG Halloween Edition - API Backend RAG Educacional",
    description="Backend em FastAPI com FastEmbed, Supabase PGVector e OpenRouter",
    version="1.0.0"
)

# Inicializa o processador de documentos
processor = DocumentProcessor()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/auto")

# --- Modelos Pydantic ---
class IngestRequest(BaseModel):
    directory_path: Optional[str] = Field(default=OUTPUT_DIR, description="Caminho do diretório com os PDFs")

class IngestResponse(BaseModel):
    success: bool
    total_documents: int
    total_chunks: int
    embedding_dimension: int
    documents_table: List[Dict[str, Any]]
    supabase_status: Dict[str, Any]
    pipeline_steps: List[str]

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.35

class RetrievedChunk(BaseModel):
    document_name: str
    chunk_index: int
    content: str
    metadata: Dict[str, Any]
    similarity: float

class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    results: List[RetrievedChunk]
    execution_time_ms: float

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.35

class ChatResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[RetrievedChunk]
    pipeline_steps: List[str]
    has_relevant_data: bool
    execution_time_ms: float

# --- Rotas do Backend ---

@app.get("/api/health")
async def health_check():
    """Verifica a saúde do backend e das conexões externas."""
    supabase_configured = bool(os.getenv("SUPABASE_URL") and (os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY")))
    openrouter_configured = bool(OPENROUTER_API_KEY)
    
    return {
        "status": "online",
        "supabase_configured": supabase_configured,
        "openrouter_configured": openrouter_configured,
        "embedding_model": processor.model_name,
        "llm_model": OPENROUTER_MODEL
    }

@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Executa o pipeline completo de ingestão:
    INGESTÃO ➔ CHUNKING ➔ TOKENIZAÇÃO ➔ EMBEDDING ➔ ÍNDICE VETORIAL
    """
    start_time = time.time()
    steps = []

    target_dir = request.directory_path or OUTPUT_DIR
    if not os.path.exists(target_dir):
        raise HTTPException(status_code=404, detail=f"Diretório não encontrado: {target_dir}")

    steps.append(f"1. INGESTÃO: Localizados arquivos PDF no diretório '{os.path.basename(target_dir)}'")
    
    # Processa PDFs e extrai estrutura tabular
    df, chunks = processor.process_directory(target_dir)
    steps.append(f"2. CHUNKING: Gerados {len(chunks)} fragmentos semânticos estruturados a partir de {len(df)} notas fiscais")
    steps.append("3. TOKENIZAÇÃO: Textos tokenizados com Sentence Transformers")
    steps.append(f"4. EMBEDDING: Gerados vetores densos de 384 dimensões via FastEmbed ({processor.model_name})")

    # Upload para o Supabase PGVector
    upload_status = processor.upload_to_supabase(chunks)
    if upload_status.get("supabase_status") == "sucesso":
        steps.append(f"5. ÍNDICE VETORIAL: {upload_status.get('supabase_uploaded')} vetores indexados com sucesso no Supabase PGVector!")
    else:
        steps.append(f"5. ÍNDICE VETORIAL: Armazenados localmente em cache de vetores (Supabase remoto: {upload_status.get('supabase_status')})")

    return IngestResponse(
        success=True,
        total_documents=len(df),
        total_chunks=len(chunks),
        embedding_dimension=384,
        documents_table=df.to_dict(orient="records"),
        supabase_status=upload_status,
        pipeline_steps=steps
    )

@app.post("/api/retrieve", response_model=RetrieveResponse)
async def retrieve_chunks(request: RetrieveRequest):
    """
    Gera embedding da pergunta e busca os top_k fragmentos mais relevantes.
    """
    start_time = time.time()
    raw_results = processor.search_vector_store(
        query=request.query,
        top_k=request.top_k,
        match_threshold=request.threshold
    )
    elapsed_ms = (time.time() - start_time) * 1000

    chunks = [
        RetrievedChunk(
            document_name=r["document_name"],
            chunk_index=r["chunk_index"],
            content=r["content"],
            metadata=r.get("metadata", {}),
            similarity=r["similarity"]
        )
        for r in raw_results
    ]

    return RetrieveResponse(
        query=request.query,
        top_k=request.top_k,
        results=chunks,
        execution_time_ms=round(elapsed_ms, 2)
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_rag(request: ChatRequest):
    """
    Executa o fluxo RAG completo com regra antialucinação estrita:
    PERGUNTA ➔ TOKENIZAÇÃO ➔ EMBEDDING ➔ RETRIEVAL ➔ PROMPT AUMENTADO ➔ LLM via OpenRouter
    """
    start_time = time.time()
    steps = [
        f"1. PERGUNTA: '{request.query}' recebida",
        "2. TOKENIZAÇÃO: Query decomposta em sub-tokens",
        f"3. EMBEDDING: Gerado vetor da query (dim 384) via FastEmbed"
    ]

    # Recupera documentos relevantes
    retrieved = processor.search_vector_store(
        query=request.query,
        top_k=request.top_k,
        match_threshold=request.threshold
    )

    chunks = [
        RetrievedChunk(
            document_name=r["document_name"],
            chunk_index=r["chunk_index"],
            content=r["content"],
            metadata=r.get("metadata", {}),
            similarity=r["similarity"]
        )
        for r in retrieved
    ]

    steps.append(f"4. RETRIEVAL: {len(chunks)} fragmentos recuperados com score >= {request.threshold}")

    # REGRA ANTIALUCINAÇÃO ESTRITA
    # Se o retrieval não retornar candidatos ou se estiver vazio
    if not chunks:
        steps.append("5. TRATAMENTO DE EXCEÇÃO: Nenhum candidato relevante encontrado na base.")
        steps.append("6. REGRA ANTIALUCINAÇÃO: Resposta padronizada sem invenção de dados.")
        elapsed_ms = (time.time() - start_time) * 1000
        return ChatResponse(
            query=request.query,
            answer="não possuo essa informação na base de dados",
            retrieved_documents=[],
            pipeline_steps=steps,
            has_relevant_data=False,
            execution_time_ms=round(elapsed_ms, 2)
        )

    # Monta Contexto Aumentado
    context_str = "\n\n---\n\n".join([
        f"[Documento: {c.document_name} | Chunk {c.chunk_index} | Similaridade: {c.similarity:.2f}]\n{c.content}"
        for c in chunks
    ])

    steps.append("5. PROMPT AUMENTADO: Contexto concatenado e injetado com diretivas antialucinação")

    system_prompt = (
        "Você é o assistente técnico de consulta de Notas Fiscais do sistema RAG.\n"
        "Sua função é responder à pergunta do usuário utilizando ESTRITAMENTE as informações literais e explícitas contidas no CONTEXTO fornecido.\n\n"
        "DIRETRIZES INEGOCIÁVEIS:\n"
        "1. Responda APENAS quando a informação solicitada e as entidades mencionadas (nomes de pessoas, empresas, produtos) constarem de forma literal, inequívoca e explícita no contexto.\n"
        "2. NUNCA faça deduções, suposições, aproximações semânticas ou tentativas de adivinhação. Se o usuário perguntar por um nome, apelido, termo ou entidade que não exista exatamente no contexto (por exemplo, perguntar por 'gatinha' quando a base tem apenas 'Pet Shop Gatinho Mágico' para 'Gabriela Almeida', ou qualquer outro termo divergente), você NÃO deve associar ou assumir que é a mesma coisa.\n"
        "3. Em QUALQUER caso onde a informação exata solicitada não conste explicitamente na base de dados, sua resposta DEVE SER EXATAMENTE:\n"
        "\"não possuo essa informação na base de dados\"\n"
        "4. Quando a informação não existir, NÃO adicione saudações, justificativas ou textos adicionais: responda única e exclusivamente: 'não possuo essa informação na base de dados'.\n"
        "5. Quando a informação solicitada existir comprovadamente na base, forneça uma resposta precisa e objetiva com os dados da nota fiscal."
    )

    user_content = f"CONTEXTO RECUPERADO DA BASE VETORIAL:\n{context_str}\n\nPERGUNTA DO USUÁRIO:\n{request.query}"

    # Chamada assíncrona ao OpenRouter
    steps.append(f"6. LLM via OpenRouter: Enviando requisição para o modelo '{OPENROUTER_MODEL}'")
    
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY não configurada no .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "RAG Halloween Edition Education RAG"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.0
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code == 200:
                result_json = resp.json()
                assistant_answer = result_json["choices"][0]["message"]["content"]
            else:
                assistant_answer = f"[!] Erro ao consultar OpenRouter (Status {resp.status_code}): {resp.text}"
    except Exception as e:
        assistant_answer = f"[!] Falha na comunicação com OpenRouter: {str(e)}"

    elapsed_ms = (time.time() - start_time) * 1000

    return ChatResponse(
        query=request.query,
        answer=assistant_answer,
        retrieved_documents=chunks,
        pipeline_steps=steps,
        has_relevant_data=True,
        execution_time_ms=round(elapsed_ms, 2)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
