import os

LOG_FILE = r"c:\Users\Alexandre\Documents\Testes\Projeto01\logs\bot.log"

def scan_log():
    if not os.path.exists(LOG_FILE):
        print("Arquivo não existe.")
        return

    print(f"Tamanho do arquivo: {os.path.getsize(LOG_FILE)} bytes")
    
    count = 0
    with open(LOG_FILE, 'rb') as f:
        # Read file as binary and decode line by line loosely
        for line in f:
            try:
                decoded = line.decode('utf-8', errors='ignore')
                if "deepseek" in decoded.lower():
                    print(decoded.strip())
                    count += 1
            except:
                pass
            
    print(f"Encontradas {count} linhas com 'deepseek'.")

if __name__ == "__main__":
    scan_log()
