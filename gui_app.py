import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sys
import subprocess
import os
import time
import logging
import queue
import json
import queue
import json
from datetime import datetime
import pandas as pd # Force PyInstaller to bundle
from trader import BitcoinTrader # Force PyInstaller to bundle

# --- Redirect Stdout/Stderr to Queue ---
class QueueWriter:
    def __init__(self, q):
        self.q = q

    def write(self, message):
        self.q.put(message)

    def flush(self):
        pass

# --- Main App ---
class BitcoinAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bitcoin AI Trader - Painel de Controle v2.1")
        self.root.geometry("1100x750")
        
        # Configure Grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1) # Log area expands

        # --- Header ---
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.grid(row=0, column=0, sticky="ew")
        
        lbl_title = ttk.Label(header_frame, text="Sistema Bitcoin AI", font=("Helvetica", 16, "bold"))
        lbl_title.pack(side="left")

        # --- Status & Balance ---
        status_frame = ttk.Frame(header_frame)
        status_frame.pack(side="right")
        
        self.status_var = tk.StringVar(value="Status: Ocioso")
        lbl_status = ttk.Label(status_frame, textvariable=self.status_var, font=("Consolas", 10))
        lbl_status.pack(side="top", anchor="e")
        
        self.balance_var = tk.StringVar(value="Saldo: ---")
        lbl_balance = ttk.Label(status_frame, textvariable=self.balance_var, font=("Consolas", 10, "bold"))
        lbl_balance.pack(side="bottom", anchor="e")

        # --- Tabs ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=5)

        # Tab 1: Trading Console (Main)
        trading_tab = ttk.Frame(self.notebook)
        self.notebook.add(trading_tab, text="Console de Trading")

        # Tab 2: Settings (API Keys)
        settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(settings_tab, text="Configurações")

        # Tab 3: Hall of Fame
        hof_tab = ttk.Frame(self.notebook)
        self.notebook.add(hof_tab, text="Hall da Fama (Estratégias)")
        
        hof_frame = ttk.Frame(hof_tab, padding="10")
        hof_frame.pack(fill="both", expand=True)
        
        # Button
        btn_refresh_hof = ttk.Button(hof_frame, text="Atualizar Estratégias", command=self.load_strategies)
        btn_refresh_hof.pack(anchor="w", pady=5)
        
        # Treeview
        cols_hof = ('ID', 'Regime', 'Winrate', 'Trades', 'Origem', 'Lógica (Genes)')
        self.tree_hof = ttk.Treeview(hof_frame, columns=cols_hof, show='headings')
        self.tree_hof.heading('ID', text='ID')
        self.tree_hof.column('ID', width=40)
        self.tree_hof.heading('Regime', text='Regime')
        self.tree_hof.column('Regime', width=80)
        self.tree_hof.heading('Winrate', text='Winrate %')
        self.tree_hof.column('Winrate', width=80)
        self.tree_hof.heading('Trades', text='# Trades')
        self.tree_hof.column('Trades', width=60)
        self.tree_hof.heading('Origem', text='Origem')
        self.tree_hof.column('Origem', width=80)
        self.tree_hof.heading('Lógica (Genes)', text='Lógica')
        self.tree_hof.column('Lógica (Genes)', width=400)
        
        sb_hof = ttk.Scrollbar(hof_frame, orient=tk.VERTICAL, command=self.tree_hof.yview)
        self.tree_hof.configure(yscroll=sb_hof.set)
        self.tree_hof.pack(side="left", fill="both", expand=True)
        sb_hof.pack(side="right", fill="y")

        # --- SETTINGS TAB CONTENT ---
        settings_frame = ttk.LabelFrame(settings_tab, text="Configuração API Kraken", padding="20")
        settings_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(settings_frame, text="Kraken API Key:").pack(anchor="w")
        self.ent_api_key = ttk.Entry(settings_frame, width=60)
        self.ent_api_key.pack(fill="x", pady=(0, 10))
        
        ttk.Label(settings_frame, text="Kraken Secret:").pack(anchor="w")
        self.ent_secret = ttk.Entry(settings_frame, width=60, show="*")
        self.ent_secret.pack(fill="x", pady=(0, 10))
        
        ttk.Label(settings_frame, text="Nota: Chaves salvas em user_config.json após iniciar.", font=("Arial", 8, "italic")).pack(anchor="w")

        # Demo Mode Checkbox
        self.var_demo_mode = tk.BooleanVar(value=True) # Default to True for safety/sanity
        self.chk_demo = ttk.Checkbutton(settings_frame, text="Usar Modo Demo (Sandbox Futures)", variable=self.var_demo_mode)
        self.chk_demo.pack(anchor="w", pady=(5, 0))

        # --- TRADING TAB CONTENT ---
        content_frame = ttk.Frame(trading_tab, padding="5")
        content_frame.pack(fill="both", expand=True)
        content_frame.columnconfigure(1, weight=1)

        # Controls (Left)
        control_pane = ttk.LabelFrame(content_frame, text="Parâmetros de Trade", padding="10")
        control_pane.grid(row=0, column=0, sticky="ns", padx=5)
        
        params_frame = ttk.Frame(control_pane)
        params_frame.pack(fill="x", pady=5)
        
        # Row 1
        ttk.Label(params_frame, text="Valor Entrada (USD):").grid(row=0, column=0, sticky="w")
        self.ent_amount = ttk.Entry(params_frame, width=10)
        self.ent_amount.insert(0, "50.0") # Default $50
        self.ent_amount.grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Label(params_frame, text="Alavancagem (x):").grid(row=0, column=2, sticky="w")
        self.ent_leverage = ttk.Entry(params_frame, width=5)
        self.ent_leverage.insert(0, "50")
        self.ent_leverage.grid(row=0, column=3, sticky="w", padx=5)
        
        # Row 2
        ttk.Label(params_frame, text="Stop Loss (%):").grid(row=1, column=0, sticky="w", pady=5)
        self.ent_sl = ttk.Entry(params_frame, width=10)
        self.ent_sl.insert(0, "30.0") # Aggressive risk profile
        self.ent_sl.grid(row=1, column=1, sticky="w", padx=5)
        
        ttk.Label(params_frame, text="Take Profit (%):").grid(row=1, column=2, sticky="w", pady=5)
        self.ent_tp = ttk.Entry(params_frame, width=10)
        self.ent_tp.insert(0, "70.0") # High reward target
        self.ent_tp.grid(row=1, column=3, sticky="w", padx=5)
        
        ttk.Separator(control_pane, orient="horizontal").pack(fill="x", pady=15)

        # Buttons
        self.btn_run_trader = ttk.Button(control_pane, text="INICIAR Trading (Ao Vivo)", command=self.toggle_trader)
        self.btn_run_trader.pack(fill="x", pady=5)

        # This button replaces 'Open Dashboard'
        self.btn_backtest = ttk.Button(control_pane, text="Executar Backtest", command=self.run_backtest_handler)
        self.btn_backtest.pack(fill="x", pady=5)
        
        self.btn_exit = ttk.Button(control_pane, text="Sair", command=self.on_closing)
        self.btn_exit.pack(fill="x", pady=15, side="bottom")

        # --- Trade History (Right) ---
        history_frame = ttk.LabelFrame(content_frame, text="Histórico de Trades (Ao Vivo)", padding="5")
        history_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        
        cols = ('Hora', 'Símbolo', 'Lado', 'Qtd', 'Preço', 'Status')
        self.tree = ttk.Treeview(history_frame, columns=cols, show='headings', height=8)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80 if col != 'Hora' else 140)
        
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        scrollbar.pack(side="right", fill="y")

        # --- Log Area ---
        log_frame = ttk.LabelFrame(self.root, text="Logs do Sistema", padding="5")
        log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        
        self.log_area = scrolledtext.ScrolledText(log_frame, state='disabled', font=("Consolas", 9))
        self.log_area.pack(fill="both", expand=True)

        # --- Internal State ---
        self.trader_thread = None
        self.backtest_process = None
        self.running_trader = False
        self.stop_event = threading.Event()
        self.trader_instance = None # To access get_balance

        # --- Logging Setup ---
        self.log_queue = queue.Queue()
        sys.stdout = QueueWriter(self.log_queue)
        sys.stderr = QueueWriter(self.log_queue)
        
        # Start Polling Loops
        self.root.after(100, self.process_log_queue)
        self.root.after(5000, self.update_ui_data) # Update balance/history every 5s
        
        # Load Config
        self.load_config()

    def load_config(self):
        """Loads user settings from json."""
        if os.path.exists("user_config.json"):
            try:
                with open("user_config.json", "r") as f:
                    data = json.load(f)
                    self.ent_api_key.insert(0, data.get("api_key", ""))
                    self.ent_secret.insert(0, data.get("secret", ""))
                    
                    if "demo_mode" in data:
                        self.var_demo_mode.set(data["demo_mode"])

                    if "amount" in data:
                        self.ent_amount.delete(0, tk.END)
                        self.ent_amount.insert(0, str(data["amount"]))
                    if "leverage" in data:
                        self.ent_leverage.delete(0, tk.END)
                        self.ent_leverage.insert(0, str(data["leverage"]))
                    if "sl_pct" in data:
                        self.ent_sl.delete(0, tk.END)
                        self.ent_sl.insert(0, str(data["sl_pct"]))
                    if "tp_pct" in data:
                        self.ent_tp.delete(0, tk.END)
                        self.ent_tp.insert(0, str(data["tp_pct"]))
                        
                self.log("Configuração carregada de user_config.json")
            except Exception as e:
                self.log(f"Falha ao carregar config: {e}")

    def save_config(self, api_key, secret, settings):
        """Saves current settings to json."""
        data = {
            "api_key": api_key,
            "secret": secret,
            "demo_mode": self.var_demo_mode.get(),
            **settings
        }
        try:
            with open("user_config.json", "w") as f:
                json.dump(data, f, indent=4)
            self.log("Configuração salva em user_config.json")
        except Exception as e:
            self.log(f"Falha ao salvar config: {e}")

    def process_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_area.configure(state='normal')
                self.log_area.insert(tk.END, msg)
                self.log_area.see(tk.END)
                self.log_area.configure(state='disabled')
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_log_queue)

    def update_ui_data(self):
        """Polls for Balance and History updates if trader is active."""
        if self.trader_instance and self.trader_instance.dm:
             # Balance
            bal = self.trader_instance.dm.get_balance()
            if bal:
                self.balance_var.set(f"Saldo: ${bal:.2f}")
            
            # History
            trades = self.trader_instance.dm.get_recent_trades(limit=20)
            # Clear current items
            for item in self.tree.get_children():
                self.tree.delete(item)
            # Add new items
            for t in trades:
                self.tree.insert('', 'end', values=(
                    t['time'], t['symbol'], t['side'].upper(), t['amt'], f"${t['price']:.2f}", t['status']
                ))
                
        self.root.after(5000, self.update_ui_data)
        
    def load_strategies(self):
        """Loads strategies from DB."""
        try:
             # Crude check if we have a trader instance or need to make a temp DM
             if hasattr(self, 'trader_instance') and self.trader_instance:
                 dm = self.trader_instance.dm
             else:
                 from data_manager import DataManager
                 dm = DataManager()
                 
             strats = dm.get_top_strategies(limit=100)
             for item in self.tree_hof.get_children():
                 self.tree_hof.delete(item)
             
             for s in strats:
                 self.tree_hof.insert('', 'end', values=(
                     s['id'], s['regime'], f"{s['winrate']:.1f}%", s['trades'], s['origin'], s['genes']
                 ))
        except Exception as e:
            self.log(f"Erro ao carregar estratégias: {e}")

    def log(self, message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    # --- Actions ---
    def toggle_trader(self):
        if not self.running_trader:
            # Validate Inputs
            api_key = self.ent_api_key.get().strip()
            secret = self.ent_secret.get().strip()
            
            if not api_key or not secret:
                self.log("ERRO: API Key e Secret são obrigatórios.")
                return
                
            try:
                user_settings = {
                    'amount': float(self.ent_amount.get()),
                    'leverage': float(self.ent_leverage.get()),
                    'sl_pct': float(self.ent_sl.get()),
                    'tp_pct': float(self.ent_tp.get()),
                    'demo_mode': self.var_demo_mode.get()
                }
            except ValueError:
                self.log("ERRO: Valores numéricos inválidos.")
                return

            # Start
            self.running_trader = True
            
            # Save Config
            self.save_config(api_key, secret, user_settings)
            
            self.btn_run_trader.configure(text="PARAR Trading")
            self.status_var.set("Status: Trading Ativo")
            
            # Disable inputs
            self.ent_api_key.configure(state='disabled')
            self.ent_secret.configure(state='disabled')
            self.ent_amount.configure(state='disabled')
            
            self.trader_thread = threading.Thread(target=self.run_trader_safe, args=(api_key, secret, user_settings), daemon=True)
            self.trader_thread.start()
        else:
            # Stop (Soft)
            self.log("Solicitando Parada... (Reinicie o App para reset total)")
            self.running_trader = False
            self.btn_run_trader.configure(text="INICIAR Trading (Ao Vivo)")
            self.status_var.set("Status: Parado")
            
            # Re-enable inputs
            self.ent_api_key.configure(state='normal')
            self.ent_secret.configure(state='normal')
            self.ent_amount.configure(state='normal')

    def run_trader_safe(self, api_key, secret, user_settings):
        self.log("Inicializando Trader...")
        try:
            # BitcoinTrader is now imported at top level
            # Instantiate with params
            self.trader_instance = BitcoinTrader(api_key=api_key, secret=secret, user_settings=user_settings)
            
            self.log("Trader Inicializado. Iniciando Loop...")
            self.trader_instance.run() 
        except Exception as e:
            self.log(f"Trader Crash: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.running_trader = False
            self.trader_instance = None       

    def run_backtest_handler(self):
        """Runs the backtest.py script in a separate process."""
        self.log("Iniciando Backtest Acelerado...")
        try:
            # We use subprocess to run python backtest.py so it prints to the same stdout which we capture
            # But wait, sys.stdout is captured by QueueWriter in THIS process. 
            # Subprocess usually writes to its own pipe.
            # We need to read that pipe.
            
            thread = threading.Thread(target=self._monitor_backtest)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.log(f"Falha ao iniciar backtest: {e}")

    def _monitor_backtest(self):
        # Force UTF-8 encoding for the subprocess to handle emojis
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        process = subprocess.Popen(
            ["python", "-u", "backtest.py"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            encoding='utf-8', # Force read as UTF-8
            bufsize=1,
            env=env # Pass environment
        )
        self.backtest_process = process
        
        # Read lines
        for line in iter(process.stdout.readline, ''):
            if line:
                self.log(line.strip())
                
        # Also check stderr
        for line in iter(process.stderr.readline, ''):
            if line:
                self.log(f"ERRO BACKTEST: {line.strip()}")
                
        process.stdout.close()
        process.wait()
        self.log("O Backtest terminou.")

    def on_closing(self):
        if self.backtest_process:
            self.backtest_process.kill()
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    root = tk.Tk()
    app = BitcoinAIApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
