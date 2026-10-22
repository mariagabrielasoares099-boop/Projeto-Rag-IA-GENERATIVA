"""
Servidor MCP do RAG: expõe a busca e o chat sobre as notas fiscais como ferramentas
para clientes MCP (n8n, Claude Desktop, Claude Code...).

Ele não carrega o modelo de embedding: repassa as chamadas para o backend FastAPI,
que já mantém o FastEmbed e a conexão com o Supabase.

Uso:
    python mcp_server.py                 # HTTP streamable em http://0.0.0.0:8001/mcp
    python mcp_server.py --stdio         # stdio, para clientes locais
"""
import os
import sys
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://127.0.0.1:8000")

mcp = FastMCP(
    "rag-notas-fiscais",
    instructions=(
        "Ferramentas de consulta à base de Notas Fiscais do RAG Halloween Edition. "
        "Use perguntar_rag para respostas prontas e fundamentadas, ou buscar_trechos "
        "para ver os fragmentos recuperados e responder você mesmo. "
        "Se a base não tiver a informação, não invente dados."
    ),
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8001")),
)


async def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{BACKEND_URL}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def perguntar_rag(pergunta: str, top_k: int = 5) -> Dict[str, Any]:
    """Responde uma pergunta sobre as notas fiscais com o fluxo RAG completo
    (retrieval híbrido + LLM com regra antialucinação). Retorna a resposta e as fontes."""
    data = await _post("/api/chat", {"query": pergunta, "top_k": top_k})
    return {
        "resposta": data["answer"],
        "tem_dados_na_base": data["has_relevant_data"],
        "fontes": sorted({d["document_name"] for d in data["retrieved_documents"]}),
    }


@mcp.tool()
async def buscar_trechos(pergunta: str, top_k: int = 5) -> Dict[str, Any]:
    """Busca os fragmentos de notas fiscais mais relevantes para a pergunta, sem chamar a LLM.
    Cada trecho vem com o documento de origem e os scores de similaridade."""
    data = await _post("/api/retrieve", {"query": pergunta, "top_k": top_k})
    return {
        "trechos": [
            {
                "documento": r["document_name"],
                "chunk": r["chunk_index"],
                "conteudo": r["content"],
                "similaridade": round(r["similarity"], 3),
                "score_hibrido": round(r["hybrid_score"], 4),
            }
            for r in data["results"]
        ]
    }


@mcp.tool()
async def status_rag() -> Dict[str, Any]:
    """Verifica se o backend do RAG está online e quais modelos e conexões estão configurados."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{BACKEND_URL}/api/health")
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    mcp.run(transport="stdio" if "--stdio" in sys.argv else "streamable-http")
