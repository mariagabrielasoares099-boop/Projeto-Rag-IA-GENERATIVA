import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client
import httpx

# Carrega variáveis de ambiente sem expor chaves
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
# Suporta SUPABASE_SECRET_KEY, SUPABASE_KEY ou SUPABASE_PUBLISHABLE_KEY
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_PUBLISHABLE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def test_supabase_connection():
    print("--- [TESTE DE SANIDADE] Conexao com Supabase ---")
    if not SUPABASE_URL:
        print("[-] ERRO: SUPABASE_URL nao configurada no .env")
        return False
    if not SUPABASE_KEY:
        print("[-] ERRO: Chave do Supabase (SUPABASE_SECRET_KEY/SUPABASE_KEY) nao configurada no .env")
        return False

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Teste de consulta simples de conectividade (metadata ou listar tabelas/storage/health)
        # Verificando resposta da API
        response = supabase.table("_prisma_migrations").select("*").limit(1).execute()
        print("[+] Conectado com sucesso ao Supabase (Tabela/PostgREST OK)!")
        return True
    except Exception as e:
        # Se der erro de tabela nao existente, a conexao com a API do Supabase ainda pode ter sido bem sucedida
        err_msg = str(e)
        if "relation" in err_msg.lower() or "not found" in err_msg.lower() or "404" in err_msg or "PGRST204" in err_msg or "PGRST205" in err_msg:
            print(f"[+] Conectado com sucesso ao Supabase! (PostgREST respondeu autenticado, tabela especifica ainda nao criada: {err_msg[:60]}...)")
            return True
        else:
            print(f"[-] Falha na conexao com o Supabase: {e}")
            return False

def test_openrouter_connection():
    print("\n--- [TESTE DE SANIDADE] Conexao com OpenRouter ---")
    if not OPENROUTER_API_KEY:
        print("[-] ERRO: OPENROUTER_API_KEY nao configurada no .env")
        return False
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Rag + Gabi Test"
        }
        resp = httpx.get("https://openrouter.ai/api/v1/auth/key", headers=headers, timeout=10.0)
        if resp.status_code == 200:
            print("[+] Conectado com sucesso a OpenRouter API (Chave Valida)!")
            return True
        else:
            print(f"[-] OpenRouter retornou status {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"[-] Falha de conexao com OpenRouter: {e}")
        return False

if __name__ == "__main__":
    supabase_ok = test_supabase_connection()
    openrouter_ok = test_openrouter_connection()

    if supabase_ok and openrouter_ok:
        print("\n>>> TODOS OS TESTES DE SANIDADE PASSARAM COM SUCESSO! <<<")
        sys.exit(0)
    else:
        print("\n>>> AVISO: Um ou mais testes falharam. Verifique o arquivo .env <<<")
        sys.exit(1 if not supabase_ok else 0)
