-- ==============================================================================
-- Script de Configuração do Banco Vetorial Remoto (Supabase + PGVector)
-- Projeto: RAG Halloween Edition
-- ==============================================================================

-- 1. Habilita a extensão pgvector para suporte a operações com vetores
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Cria a tabela document_chunks para armazenar os fragmentos e vetores (dimensão 384 = BAAI/bge-small-en-v1.5)
CREATE TABLE IF NOT EXISTS public.document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_name TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(384) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Cria índice vetorial HNSW para buscas por similaridade de cosseno ultrarrápidas
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx 
ON public.document_chunks 
USING hnsw (embedding vector_cosine_ops);

-- 4. Função RPC para cálculo e recuperação de vizinhos mais próximos (Cosine Similarity)
CREATE OR REPLACE FUNCTION public.match_documents (
    query_embedding vector(384),
    match_threshold float DEFAULT 0.0,
    match_count int DEFAULT 5
)
RETURNS TABLE (
    id bigint,
    document_name text,
    chunk_index int,
    content text,
    metadata jsonb,
    similarity float
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_name,
        dc.chunk_index,
        dc.content,
        dc.metadata,
        (1 - (dc.embedding <=> query_embedding))::float AS similarity
    FROM public.document_chunks dc
    WHERE (1 - (dc.embedding <=> query_embedding)) >= match_threshold
    ORDER BY dc.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$;

-- 5. Configuração de Políticas de Segurança (Row Level Security)
ALTER TABLE public.document_chunks ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Acesso publico document_chunks" ON public.document_chunks;
CREATE POLICY "Acesso publico document_chunks" 
ON public.document_chunks 
FOR ALL 
USING (true) 
WITH CHECK (true);
