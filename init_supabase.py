import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_PUBLISHABLE_KEY")

def check_or_setup_table():
    print("=" * 60)
    print("VERIFICAÇÃO DO BANCO DE DADOS REMOTO (Supabase + PGVector)")
    print("=" * 60)
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[-] ERRO: Credenciais do Supabase não encontradas no .env")
        return False

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    try:
        res = supabase.table("document_chunks").select("id").limit(1).execute()
        print("[+] Tabela 'document_chunks' encontrada e acessível no Supabase!")
        return True
    except Exception as e:
        err = str(e)
        if "PGRST205" in err or "schema cache" in err or "relation" in err:
            print("\n[!] A tabela 'document_chunks' ainda não foi criada no banco de dados Supabase.")
            print("\n>>> INSTRUÇÕES PARA ATIVAÇÃO NO SUPABASE <<<")
            print("1. Acesse o painel do Supabase do seu projeto:")
            print(f"   {SUPABASE_URL.replace('.co', '.co/project')}")
            print("2. Vá em 'SQL Editor' (no menu lateral esquerdo).")
            print("3. Abra o arquivo 'setup_supabase_pgvector.sql' deste projeto, cole o conteúdo e clique em 'RUN'.")
            print("4. O script criará a extensão 'vector', a tabela 'document_chunks' e a função 'match_documents'.\n")
            return False
        else:
            print(f"[-] Erro ao verificar tabela: {e}")
            return False

if __name__ == "__main__":
    ready = check_or_setup_table()
    if ready:
        print("[+] Supabase pronto para o pipeline de embeddings!")
    else:
        print("[!] Verifique as instruções acima para executar o script SQL no Supabase.")
