# 🎃 Rag + Gabi | Aplicação RAG Didática e Visual

Sistema RAG (Retrieval-Augmented Generation) educacional e interativo focado na compreensão de ponta a ponta do pipeline de dados, desenvolvido com **FastAPI**, **Streamlit**, **FastEmbed**, **Supabase (pgvector)** e **OpenRouter**.

---

## 🐱 Identidade Visual & Conceito
- **Tema:** Halloween Dark Glow (tons de roxo profundo `#120822` e laranja neon `#ff6e14`).
- **Mascote:** **Rag + Gabi** — Gatinho travesso e fofinho de Halloween com o machado cenográfico de Lizzie Borden.

---

## 🏗️ Arquitetura & Fluxo de Processamento

### 1. Pipeline de Ingestão de Dados (Página A)
```
[PDFs de Notas Fiscais]
         │
         ▼
     INGESTÃO (Extração de texto via pypdf)
         │
         ▼
     CHUNKING (Segmentação semântica e estruturação tabular)
         │
         ▼
   TOKENIZAÇÃO (Sentence Transformers Tokenizer)
         │
         ▼
    EMBEDDING (Vetores densos de 384 dimensões via FastEmbed)
         │
         ▼
 ÍNDICE VETORIAL (Supabase PGVector / Cache de redundância)
```

### 2. Pipeline de Consulta e Chat RAG (Página B)
```
[Pergunta do Usuário]
         │
         ▼
    PERGUNTA
         │
         ▼
   TOKENIZAÇÃO
         │
         ▼
    EMBEDDING (FastEmbed query vector)
         │
         ▼
    RETRIEVAL (Top-k=5 vizinhos mais próximos via Cosseno)
         │
    ┌────┴────────────────────────┐
    ▼                             ▼
[Candidatos Encontrados]    [Sem Candidatos / Score Baixo]
    │                             │
    ▼                             ▼
PROMPT AUMENTADO           REGRA ANTIALUCINAÇÃO ESTRITA
    │                             │
    ▼                             ▼
LLM via OpenRouter         "Sem dados disponíveis na base"
    │
    ▼
Resposta Fundamentada + Top-5 Expander
```

---

## 📂 Estrutura de Arquivos

```
Rag/
├── assets/
│   └── rag_gabi_mascot.jpg        # Mascote oficial Rag + Gabi Halloween
├── dados_notas_fiscais/            # 4 PDFs de Notas Fiscais fictícias geradas
│   ├── NF_001_Gatinhos_Petshop.pdf
│   ├── NF_002_Servicos_Tech_Rag.pdf
│   ├── NF_003_Loja_Fantasias_Halloween.pdf
│   └── NF_004_Equipamentos_Laboratorio.pdf
├── .env                            # Variáveis de ambiente reais (não versionadas)
├── .env.example                    # Modelo de configuração sem credenciais
├── setup_supabase_pgvector.sql     # Script SQL para habilitar pgvector e criar a tabela/função RPC
├── test_connection.py              # Teste isolado de sanidade de conexão
├── init_supabase.py                # Verificador de status do Supabase
├── generate_invoices.py            # Gerador dos 4 PDFs fiscais fictícios
├── data_pipeline.py                # Pipeline OCR, FastEmbed e busca vetorial
├── backend.py                      # Backend FastAPI com rotas assíncronas
├── app.py                          # Frontend Streamlit temático Halloween
└── README.md                       # Documentação da aplicação
```

---

## 🚀 Como Executar o Projeto

### 1. Ativação do Ambiente Virtual
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Teste de Sanidade de Conectividade
Antes de rodar a aplicação, execute o teste isolado:
```powershell
python test_connection.py
```

### 3. Configuração do Supabase (PGVector)
Para persistir os vetores no banco remoto:
1. Acesse o painel do Supabase do seu projeto.
2. Abra o **SQL Editor**.
3. Copie o conteúdo de `setup_supabase_pgvector.sql`, cole no editor e clique em **RUN**.
4. Teste a prontidão rodando:
   ```powershell
   python init_supabase.py
   ```

### 4. Inicialização do Backend (FastAPI)
```powershell
uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
```
Acesse a documentação interativa em: `http://127.0.0.1:8000/docs`

### 5. Inicialização do Frontend (Streamlit)
Em outro terminal:
```powershell
streamlit run app.py
```
Acesse a interface no navegador: `http://localhost:8501`

---

## 🛡️ Regra Antialucinação Estrita
Se o usuário fizer perguntas que não possuem respaldo documental nos fragmentos recuperados (ex: *"Qual o valor da passagem aérea para Paris?"*), o sistema intercepta e retorna estritamente:
> **"Sem dados disponíveis na base"**
