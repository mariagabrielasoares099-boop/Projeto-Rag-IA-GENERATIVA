import os
import time
import base64
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configuração da Página
st.set_page_config(
    page_title="Rag + Gabi 🎃 | Sistema RAG Educacional",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
MASCOT_PATH = os.path.join(os.path.dirname(__file__), "assets", "rag_gabi_mascot.jpg")

# --- CSS Customizado: Tema Halloween + Neon Dark ---
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Creepster&family=Inter:wght@300;400;600;700&display=swap');

    /* Barra Superior / Header do Streamlit na Cor Laranja */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #ff6e14 0%, #ff8c38 50%, #e65c00 100%) !important;
        background-color: #ff6e14 !important;
        border-bottom: 2px solid #b34700 !important;
        box-shadow: 0 4px 16px rgba(255, 110, 20, 0.4) !important;
    }

    /* Linha decorativa de topo */
    div[data-testid="stDecoration"] {
        background: #b34700 !important;
        height: 4px !important;
    }

    /* Elementos dentro do header */
    header[data-testid="stHeader"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    /* Caixa Superior de Destaque Laranja */
    .top-orange-box {
        background: linear-gradient(135deg, #ff6e14 0%, #ff8533 100%) !important;
        border-radius: 14px;
        padding: 16px 22px;
        box-shadow: 0 6px 20px rgba(255, 110, 20, 0.4);
        border: 2px solid #ffa35c;
    }

    /* Fundo da Aplicação e Textos Gerais de Alto Contraste */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #201035 0%, #0d0617 75%, #05020a 100%) !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }

    /* Parágrafos, labels, spans e itens de lista em branco nítido */
    p, span, label, li {
        color: #f7f3ff !important;
        font-size: 15px;
    }

    /* Legendas, subtextos e captions em tom claro visível */
    .stCaption, [data-testid="stCaptionContainer"], small {
        color: #e4d5fa !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
    }

    /* Barra Lateral de Alto Contraste */
    section[data-testid="stSidebar"] {
        background-color: #120822 !important;
        border-right: 1px solid #4a2179 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] div {
        color: #ffffff !important;
        font-weight: 450;
    }

    /* Cabeçalhos e Títulos */
    h1, h2, h3, h4 {
        color: #ffa35c !important;
        text-shadow: 0 0 12px rgba(255, 163, 92, 0.45);
        font-weight: 700 !important;
    }

    /* Tabs customizadas com alto contraste */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #190a30;
        padding: 8px 12px;
        border-radius: 12px;
        border: 1px solid #4f2387;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        color: #ffffff !important;
        background-color: #2e1454 !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border-radius: 8px;
        padding: 0 20px;
        border: 1px solid #5d289e;
        transition: all 0.25s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ff6e14 !important;
        color: #ffffff !important;
        border: 1px solid #ff9954 !important;
        box-shadow: 0 0 16px rgba(255, 110, 20, 0.6) !important;
    }

    /* Chat Messages */
    div[data-testid="stChatMessage"] {
        background-color: #190c30 !important;
        border: 1px solid #4a2179 !important;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
    }

    /* Balão do Usuário - Cor de texto preta (#000000) com fundo claro nítido */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: #f5efff !important;
        border-left: 6px solid #ff6e14 !important;
        color: #000000 !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) p,
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) span,
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 15.5px !important;
    }

    /* Balão do Assistente (Mascote Rag + Gabi) - Texto em branco nítido */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, #24113b 0%, #150826 100%) !important;
        border-left: 6px solid #ff7714 !important;
        border: 1px solid #5a2694 !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) p,
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) span,
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) li,
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) div {
        color: #ffffff !important;
        font-size: 15.5px !important;
        line-height: 1.6 !important;
        font-weight: 450 !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) strong {
        color: #ffa35c !important;
        font-weight: 700 !important;
    }

    /* Campo de Pesquisa / Chat Input do Usuário - Letra preta (#000000) */
    div[data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border-radius: 14px !important;
        border: 2px solid #ff7714 !important;
        box-shadow: 0 0 16px rgba(255, 119, 20, 0.4) !important;
    }

    div[data-testid="stChatInput"] textarea {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #ffffff !important;
        font-size: 15.5px !important;
        font-weight: 600 !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #555555 !important;
        -webkit-text-fill-color: #555555 !important;
        font-weight: 450 !important;
    }

    div[data-testid="stChatInput"] button {
        color: #ff6e14 !important;
    }

    /* Inputs de texto padrão */
    input[type="text"], .stTextInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Botões */
    div.stButton > button {
        background: linear-gradient(90deg, #ff6e14 0%, #9933ff 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        box-shadow: 0 4px 14px rgba(255, 110, 20, 0.35) !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 110, 20, 0.65) !important;
    }

    /* Expander com títulos e textos nítidos */
    div[data-testid="stExpander"] {
        background-color: #1a0b33 !important;
        border: 1px solid #662db0 !important;
        border-radius: 10px;
    }

    div[data-testid="stExpander"] summary span {
        color: #ffa35c !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }

    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] span,
    div[data-testid="stExpander"] div {
        color: #ffffff !important;
    }

    /* Métricas */
    div[data-testid="stMetricLabel"] p {
        color: #e5d6ff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffaa66 !important;
        font-weight: 800 !important;
    }

    /* Status Container */
    div[data-testid="stStatusWidget"] {
        background-color: #1f0c3d !important;
        border: 2px solid #8e3ff2 !important;
        border-radius: 12px;
    }

    div[data-testid="stStatusWidget"] p,
    div[data-testid="stStatusWidget"] span {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* Alertas e caixas de informação */
    div[data-testid="stAlert"] {
        background-color: #241042 !important;
        color: #ffffff !important;
        border: 1px solid #ff7714 !important;
    }

    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span {
        color: #ffffff !important;
    }

    /* Textos preformatados e blocos de código */
    pre, code {
        color: #ffffff !important;
        background-color: #160729 !important;
        border: 1px solid #572594 !important;
    }

    /* Badge Halloween */
    .halloween-badge {
        display: inline-block;
        background-color: #ff6e14;
        color: #ffffff;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 12.5px;
        margin-right: 6px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Helper para converter imagem local para base64 HTML
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# Inicialização de Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Olá! Eu sou o mascote **Rag + Gabi** 🐱🎃! "
                       "Estou pronto para navegar nas Notas Fiscais indexadas no **Supabase PGVector** com embeddings do **FastEmbed**. "
                       "Faça perguntas sobre produtos, valores ou emitentes!",
            "retrieved_docs": []
        }
    ]

if "indexed_data" not in st.session_state:
    st.session_state.indexed_data = None

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    if os.path.exists(MASCOT_PATH):
        st.image(MASCOT_PATH, use_container_width=True, caption="Rag + Gabi 🐱🪓 Machado de Lizzie Borden")
    else:
        st.header("🐱 Rag + Gabi")

    st.markdown("### 🎃 Painel de Controle")
    st.markdown("""
    **Stack Tecnológica:**
    - 🗄️ **Database:** Supabase (`pgvector`)
    - 🧠 **Embeddings:** `fastembed` (`BAAI/bge-small-en-v1.5`)
    - 🤖 **LLM:** OpenRouter (`openrouter/auto`)
    - ⚡ **Backend:** FastAPI Assíncrono
    - 🎨 **Frontend:** Streamlit Halloween Dark
    """)

    st.divider()
    
    # Verificação de status do Backend
    try:
        health_resp = requests.get(f"{BACKEND_URL}/api/health", timeout=2.0)
        if health_resp.status_code == 200:
            data = health_resp.json()
            st.success("🟢 Backend FastAPI Conectado!")
            st.caption(f"🧠 FastEmbed: `{data.get('embedding_model')}`")
            st.caption(f"🤖 LLM: `{data.get('llm_model')}`")
        else:
            st.warning("🟡 Backend respondeu com status inesperado.")
    except Exception:
        st.info("ℹ️ Backend FastAPI offline. Usando modo direto integrado.")

    st.divider()
    if st.button("🧹 Limpar Histórico de Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Histórico reiniciado! O que gostaria de consultar nas Notas Fiscais?",
                "retrieved_docs": []
            }
        ]
        st.rerun()

# --- Cabeçalho Principal ---
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if os.path.exists(MASCOT_PATH):
        st.image(MASCOT_PATH, width=110)
    else:
        st.title("🎃🐱")

with col_title:
    st.markdown("""
    <div class="top-orange-box">
        <h1 style="margin: 0; color: #ffffff !important; font-size: 2.2rem; text-shadow: 0 2px 8px rgba(0,0,0,0.35);">🎃 Rag + Gabi | RAG Educacional</h1>
        <p style="margin-top: 6px; margin-bottom: 0; color: #fff5ed !important; font-size: 15px;">
            <span class="halloween-badge" style="background-color: #200c36; color: #ffa05c; border: 1px solid #ffa05c;">HALLOWEEN SPECIAL</span>
            Pipeline Visual de Ingestão de Notas Fiscais, Supabase PGVector, FastEmbed & LLM Antialucinação
        </p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --- Divisão da Aplicação em Abas (Tabs) ---
tab_ingest, tab_chat = st.tabs([
    "📥 Página A: Painel de Ingestão de Dados",
    "💬 Página B: Painel de Consulta (Chat RAG)"
])

# ==============================================================================
# PÁGINA A: PAINEL DE INGESTÃO DE DADOS
# ==============================================================================
with tab_ingest:
    st.subheader("📁 Ingestão & Processamento de Documentos")
    st.markdown(
        "Selecione os documentos fiscais em PDF ou utilize a base de teste gerada. "
        "Acompanhe cada etapa didática do pipeline em tempo real."
    )

    col_btn, col_upload = st.columns([1, 1])
    
    with col_btn:
        st.markdown("#### 📄 Base Padrão de Notas Fiscais")
        st.caption("Contém 4 Notas Fiscais simuladas (Pet Shop, Tech RAG, Fantasias Halloween, Hardware).")
        exec_default_btn = st.button("🚀 Processar Base Fictícia Local", use_container_width=True)

    with col_upload:
        st.markdown("#### 📤 Upload de Novos PDFs")
        uploaded_files = st.file_uploader(
            "Selecione arquivos PDF para ingestão:",
            type=["pdf"],
            accept_multiple_files=True
        )
        exec_upload_btn = st.button("⚡ Processar Arquivos Enviados", disabled=(not uploaded_files), use_container_width=True)

    # Execução do Pipeline de Ingestão
    if exec_default_btn or (exec_upload_btn and uploaded_files):
        target_dir = os.path.join(os.path.dirname(__file__), "dados_notas_fiscais")
        
        # Se usuário enviou PDFs manuais, salva temporariamente
        if exec_upload_btn and uploaded_files:
            upload_dir = os.path.join(os.path.dirname(__file__), "uploaded_pdfs")
            os.makedirs(upload_dir, exist_ok=True)
            for up_file in uploaded_files:
                save_path = os.path.join(upload_dir, up_file.name)
                with open(save_path, "wb") as f:
                    f.write(up_file.getbuffer())
            target_dir = upload_dir

        # Exibição Visual Dinâmica do Pipeline com st.status
        with st.status("Iniciando Pipeline RAG Didático...", expanded=True) as status_box:
            st.write("📥 **Etapa 1:** INGESTÃO ➔ Lendo arquivos PDF e extraindo texto bruto com `pypdf`...")
            time.sleep(0.4)
            
            st.write("🧩 **Etapa 2:** CHUNKING ➔ Fragmentando documentos em blocos semânticos e extraindo metadados...")
            time.sleep(0.4)
            
            st.write("🔤 **Etapa 3:** TOKENIZAÇÃO ➔ Tokenizando vocabulário com Sentence Transformers...")
            time.sleep(0.4)
            
            st.write("🧠 **Etapa 4:** EMBEDDING ➔ Gerando representações vetoriais densas de 384 dimensões via `fastembed`...")
            time.sleep(0.4)
            
            st.write("🗄️ **Etapa 5:** ÍNDICE VETORIAL ➔ Sincronizando fragmentos e vetores com o Supabase PGVector...")
            
            # Tenta chamada via Backend FastAPI; se falhar, executa via módulo direto
            ingest_success = False
            result_data = None
            try:
                resp = requests.post(f"{BACKEND_URL}/api/ingest", json={"directory_path": target_dir}, timeout=60.0)
                if resp.status_code == 200:
                    result_data = resp.json()
                    ingest_success = True
            except Exception:
                pass

            if not ingest_success:
                # Fallback direto via pipeline python integrado
                from data_pipeline import DocumentProcessor
                proc = DocumentProcessor()
                df_res, chunks_res = proc.process_directory(target_dir)
                up_res = proc.upload_to_supabase(chunks_res)
                result_data = {
                    "total_documents": len(df_res),
                    "total_chunks": len(chunks_res),
                    "embedding_dimension": 384,
                    "documents_table": df_res.to_dict(orient="records"),
                    "supabase_status": up_res,
                    "pipeline_steps": [
                        "INGESTÃO Concluída", "CHUNKING Concluído", 
                        "TOKENIZAÇÃO Concluída", "EMBEDDING Concluído", "ÍNDICE VETORIAL Concluído"
                    ]
                }
                ingest_success = True

            st.session_state.indexed_data = result_data
            status_box.update(
                label="✨ Pipeline de Ingestão Finalizado com Êxito! 🎃",
                state="complete",
                expanded=False
            )

        st.success("✅ Documentos processados e vetorizados com sucesso!")

    # Exibição dos Dados Estruturados
    if st.session_state.indexed_data:
        data = st.session_state.indexed_data
        
        # Métricas Chave
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("📄 Documentos Ingeridos", data.get("total_documents", 0))
        col_m2.metric("🧩 Chunks Gerados", data.get("total_chunks", 0))
        col_m3.metric("📐 Dimensão Vetorial", f"{data.get('embedding_dimension', 384)} dims")
        sb_status = data.get("supabase_status", {}).get("supabase_status", "ativo")
        col_m4.metric("🗄️ Supabase PGVector", "Ativo / Cache" if sb_status else "Pronto")

        st.markdown("#### 📊 Tabela Estruturada das Notas Fiscais (Extração OCR)")
        df_display = pd.DataFrame(data.get("documents_table", []))
        if not df_display.empty:
            st.dataframe(df_display, use_container_width=True)

        with st.expander("🛠️ Detalhes Técnicos do Pipeline & Supabase"):
            st.json(data.get("supabase_status", {}))
            st.write("**Fluxo Percorrido:**")
            for step in data.get("pipeline_steps", []):
                st.write(f"- {step}")

# ==============================================================================
# PÁGINA B: PAINEL DE CONSULTA (CHAT RAG)
# ==============================================================================
with tab_chat:
    st.subheader("💬 Chat RAG com Antialucinação Estrita")
    st.caption("Consulte os dados das Notas Fiscais. Caso a informação não exista na base, o sistema responderá estritamente: **'não possuo essa informação na base de dados'**.")

    # Exibe Histórico de Mensagens
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Se houver fragmentos recuperados associados à resposta, exibe
            if msg.get("retrieved_docs"):
                with st.expander(f"🔍 Fragmentos Relevantes Recuperados ({len(msg['retrieved_docs'])} top-k)"):
                    for idx, doc in enumerate(msg["retrieved_docs"], 1):
                        st.markdown(
                            f"**#{idx} | Arquivo:** `{doc['document_name']}` — "
                            f"**Similaridade:** `{doc['similarity']:.4f}`"
                        )
                        st.text(doc["content"])
                        st.divider()

    # Input do Usuário
    user_query = st.chat_input("Digite sua pergunta sobre as Notas Fiscais (ex: 'Qual o valor total da nota da Gabriela?')")

    if user_query:
        # Registra a pergunta do usuário
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Resposta do Assistente
        with st.chat_message("assistant"):
            # Exibição Dinâmica do Fluxo de Execução com st.status
            with st.status("Processando fluxo RAG...", expanded=True) as chat_status:
                st.write("❓ **1. PERGUNTA:** Recebendo consulta do usuário...")
                time.sleep(0.3)
                
                st.write("🔤 **2. TOKENIZAÇÃO:** Analisando tokens semânticos...")
                time.sleep(0.3)
                
                st.write("🧠 **3. EMBEDDING:** Calculando vetor da consulta via FastEmbed...")
                time.sleep(0.3)
                
                st.write("🔍 **4. RETRIEVAL:** Buscando vizinhos mais próximos no Supabase PGVector (top_k=5)...")
                time.sleep(0.3)
                
                st.write("📝 **5. PROMPT AUMENTADO:** Injetando contexto recuperado e diretiva antialucinação...")
                time.sleep(0.3)
                
                st.write("🤖 **6. LLM via OpenRouter:** Gerando resposta baseada estritamente no contexto...")

                # Consulta ao Backend ou fallback direto
                chat_data = None
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/api/chat",
                        json={"query": user_query, "top_k": 5, "threshold": 0.35},
                        timeout=45.0
                    )
                    if resp.status_code == 200:
                        chat_data = resp.json()
                except Exception:
                    pass

                # Fallback direto se backend não responder
                if not chat_data:
                    from data_pipeline import DocumentProcessor
                    proc = DocumentProcessor()
                    docs = proc.search_vector_store(user_query, top_k=5, match_threshold=0.35)
                    
                    if not docs:
                        chat_data = {
                            "answer": "não possuo essa informação na base de dados",
                            "retrieved_documents": [],
                            "has_relevant_data": False
                        }
                    else:
                        # Chama OpenRouter diretamente
                        openrouter_key = os.getenv("OPENROUTER_API_KEY")
                        context_str = "\n\n".join([f"[{d['document_name']}]: {d['content']}" for d in docs])
                        prompt = f"CONTEXTO:\n{context_str}\n\nPERGUNTA: {user_query}"
                        system = (
                            "Você é o assistente técnico de consulta de Notas Fiscais do sistema RAG.\n"
                            "Sua função é responder à pergunta do usuário utilizando ESTRITAMENTE as informações literais e explícitas contidas no CONTEXTO fornecido.\n\n"
                            "DIRETRIZES INEGOCIÁVEIS:\n"
                            "1. Responda APENAS quando a informação solicitada e as entidades mencionadas constarem de forma literal, inequívoca e explícita no contexto.\n"
                            "2. NUNCA faça deduções, suposições, aproximações semânticas ou tentativas de adivinhação. Se o usuário perguntar por um nome, termo ou entidade que não exista exatamente no contexto (por exemplo, perguntar por 'gatinha' quando na base existe 'Pet Shop Gatinho Mágico' para 'Gabriela Almeida'), você NÃO deve associar ou presumir que se trata da mesma coisa.\n"
                            "3. Em QUALQUER situação onde a informação exata não conste explicitamente na base de dados, sua resposta DEVE SER EXATAMENTE:\n"
                            "\"não possuo essa informação na base de dados\"\n"
                            "4. Quando a informação não existir, NÃO adicione saudações, desculpas ou explicações adicionais: responda estritamente: 'não possuo essa informação na base de dados'."
                        )
                        
                        r_llm = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {openrouter_key}"},
                            json={
                                "model": os.getenv("OPENROUTER_MODEL", "openrouter/auto"),
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                                "temperature": 0.0
                            },
                            timeout=30.0
                        )
                        ans = r_llm.json()["choices"][0]["message"]["content"] if r_llm.status_code == 200 else "não possuo essa informação na base de dados"
                        chat_data = {
                            "answer": ans,
                            "retrieved_documents": docs,
                            "has_relevant_data": True
                        }

                chat_status.update(
                    label="Fluxo RAG Finalizado! 🐱🪓",
                    state="complete",
                    expanded=False
                )

            # Exibe a resposta final
            final_answer = chat_data.get("answer", "não possuo essa informação na base de dados")
            retrieved_items = chat_data.get("retrieved_documents", [])

            # Destaque se foi acionada a regra antialucinação
            if "não possuo essa informação na base de dados" in final_answer.lower() or "sem dados disponíveis" in final_answer.lower():
                st.warning(f"⚠️ **{final_answer}**")
            else:
                st.markdown(final_answer)

            # Exibe os top_k fragmentos com scores de similaridade
            if retrieved_items:
                with st.expander(f"🔍 Top-{len(retrieved_items)} Fragmentos Recuperados (Supabase PGVector & FastEmbed)"):
                    for idx, doc in enumerate(retrieved_items, 1):
                        sim_score = doc.get("similarity", 0.0)
                        st.markdown(
                            f"**#{idx} | Nota:** `{doc.get('document_name')}` — "
                            f"**Similaridade Cosseno:** `{sim_score:.4f}`"
                        )
                        st.info(doc.get("content"))

            # Salva na sessão
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "retrieved_docs": retrieved_items
            })
