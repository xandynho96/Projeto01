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
import json
from datetime import datetime
import sys
import os

# Fix for PyInstaller ModuleNotFoundError
if getattr(sys, 'frozen', False):
    sys.path.append(sys._MEIPASS)

import ai_brain # Force PyInstaller to bundle explicitly (Import BEFORE trader)
import pandas as pd # Force PyInstaller to bundle
from trader import BitcoinTrader # Force PyInstaller to bundle
import config
import script_utils # Make sure this file exists or is handled
import evolution # Import the evolution module

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
        self.root.title("BitcoinAI Trader - Kraken Edition")
        self.root.geometry("1100x700")
        
        # Initialize Variables FIRST
        self.initialize_variables()
        self.loading = False
        
        # Setup Logger Queue
        self.log_queue = queue.Queue()
        self.queue_writer = QueueWriter(self.log_queue)
        
        # Redirect stdout/stderr
        sys.stdout = self.queue_writer
        sys.stderr = self.queue_writer
        
        # UI Setup
        self.create_widgets()
        
        # Start Log Polling
        self.root.after(100, self.poll_log_queue)
        
        # Load Config
        self.load_settings()

    def initialize_variables(self):
        # Settings Variables

        self.mode_var = tk.StringVar(value="Spot Margin (10x)")
        self.api_key_var = tk.StringVar()
        self.secret_var = tk.StringVar()
        self.deepseek_key_var = tk.StringVar()
        self.leverage_var = tk.StringVar(value=str(config.LEVERAGE))
        self.amount_var = tk.StringVar(value="20.0")
        self.sl_var = tk.StringVar(value="0.18")
        self.tp_var = tk.StringVar(value="0.24")
        self.demo_var = tk.BooleanVar(value=False)
        
        # UI Status Variables
        self.status_var = tk.StringVar(value="Status: Ocioso")
        self.balance_var = tk.StringVar(value="Saldo: ---")
        self.weekly_pl_var = tk.StringVar(value="PL Semanal: ---")

    def create_widgets(self):
        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Load Icon
        try:
            icon_path = script_utils.resource_path("app_icon.png")
            if os.path.exists(icon_path):
                img = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(False, img)
        except Exception:
            pass

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
        
        lbl_status = ttk.Label(status_frame, textvariable=self.status_var, font=("Consolas", 10))
        lbl_status.pack(side="top", anchor="e")
        
        lbl_balance = ttk.Label(status_frame, textvariable=self.balance_var, font=("Consolas", 10, "bold"))
        lbl_balance.pack(side="bottom", anchor="e")

        self.weekly_pl_var = tk.StringVar(value="PL Semanal: ---")
        lbl_weekly_pl = ttk.Label(status_frame, textvariable=self.weekly_pl_var, font=("Consolas", 9))
        lbl_weekly_pl.pack(side="bottom", anchor="e")


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
        # --- SETTINGS TAB CONTENT ---
        settings_frame = ttk.LabelFrame(settings_tab, text="Configuração API Kraken", padding="20")
        settings_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(settings_frame, text="Kraken API Key:").pack(anchor="w")
        self.ent_api_key = ttk.Entry(settings_frame, width=60, textvariable=self.api_key_var)
        self.ent_api_key.pack(fill="x", pady=(0, 10))
        
        ttk.Label(settings_frame, text="Kraken Secret:").pack(anchor="w")
        self.ent_secret = ttk.Entry(settings_frame, width=60, show="*", textvariable=self.secret_var)
        self.ent_secret.pack(fill="x", pady=(0, 10))
        
        ttk.Label(settings_frame, text="DeepSeek API Key (Opcional):").pack(anchor="w")
        self.ent_deepseek = ttk.Entry(settings_frame, width=60, show="*", textvariable=self.deepseek_key_var)
        self.ent_deepseek.pack(fill="x", pady=(0, 10))

        # Settings Controls
        opts_frame = ttk.Frame(settings_frame)
        opts_frame.pack(fill="x", pady=10)

        # Demo Mode
        tsk_demo = ttk.Checkbutton(opts_frame, text="Usar Modo Demo (Simulação / Dry Run)", variable=self.demo_var)
        tsk_demo.pack(anchor="w", side="left", padx=(0, 20))

        # Trading Mode
        ttk.Label(opts_frame, text="Modo:").pack(side="left")
        self.mode_combo_settings = ttk.Combobox(opts_frame, textvariable=self.mode_var, values=["Spot Margin (10x)", "Futures (50x)"], state="readonly", width=18)
        self.mode_combo_settings.pack(side="left", padx=5)
        self.mode_combo_settings.bind("<<ComboboxSelected>>", self.update_leverage_limit)

        # Save Button
        btn_save = ttk.Button(settings_frame, text="💾 Salvar Configurações", command=self.save_ui_settings)
        btn_save.pack(fill="x", pady=10)

        # --- TRADING TAB CONTENT ---
        content_frame = ttk.Frame(trading_tab, padding="5")
        content_frame.pack(fill="both", expand=True)
        content_frame.columnconfigure(1, weight=1)

        # Controls (Left)
        control_pane = ttk.LabelFrame(content_frame, text="Parâmetros de Trade", padding="10")
        control_pane.grid(row=0, column=0, sticky="ns", padx=5)
        
        params_frame = ttk.Frame(control_pane)
        params_frame.pack(fill="x", pady=5)

        # Row 0: Trading Mode (Added as requested)
        ttk.Label(params_frame, text="Modo:").grid(row=0, column=0, sticky="w", pady=5)
        self.mode_combo_main = ttk.Combobox(params_frame, textvariable=self.mode_var, values=["Spot Margin (10x)", "Futures (50x)"], state="readonly", width=18)
        self.mode_combo_main.grid(row=0, column=1, columnspan=3, sticky="we", padx=5, pady=5)
        self.mode_combo_main.bind("<<ComboboxSelected>>", self.update_leverage_limit)

        # Row 1
        ttk.Label(params_frame, text="Valor Entrada (USD):").grid(row=1, column=0, sticky="w")
        ttk.Entry(params_frame, textvariable=self.amount_var, width=10).grid(row=1, column=1, sticky="w", padx=5)
        
        ttk.Label(params_frame, text="Alavancagem (x):").grid(row=1, column=2, sticky="w")
        ttk.Entry(params_frame, textvariable=self.leverage_var, width=5).grid(row=1, column=3, sticky="w", padx=5)
        
        # Row 2
        ttk.Label(params_frame, text="Stop Loss (%):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(params_frame, textvariable=self.sl_var, width=10).grid(row=2, column=1, sticky="w", padx=5)
        
        ttk.Label(params_frame, text="Take Profit (%):").grid(row=2, column=2, sticky="w", pady=5)
        ttk.Entry(params_frame, textvariable=self.tp_var, width=10).grid(row=2, column=3, sticky="w", padx=5)
        
        ttk.Separator(control_pane, orient="horizontal").pack(fill="x", pady=15)

        # Buttons
        self.btn_run_trader = ttk.Button(control_pane, text="INICIAR Trading (Ao Vivo)", command=self.toggle_trader)
        self.btn_run_trader.pack(fill="x", pady=5)

        # This button replaces 'Open Dashboard'
        self.btn_backtest = ttk.Button(control_pane, text="Executar Backtest", command=self.run_backtest_handler)
        self.btn_backtest.pack(fill="x", pady=5)
        
        self.btn_evolution = ttk.Button(control_pane, text="🧬 Evoluir Estratégias (IA)", command=self.run_evolution_handler)
        self.btn_evolution.pack(fill="x", pady=5)
        
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
        
        # Defines tags for color
        self.log_area.tag_config("buy", foreground="green", font=("Consolas", 9, "bold"))
        self.log_area.tag_config("sell", foreground="red", font=("Consolas", 9, "bold"))
        self.log_area.tag_config("error", foreground="orange")
        self.log_area.tag_config("info", foreground="black")

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
        self.load_settings()

    def save_ui_settings(self):
        """Handler for the Save Button."""
        api_key = self.ent_api_key.get().strip()
        secret = self.ent_secret.get().strip()
        
        settings = {
            'amount': float(self.amount_var.get()),
            'leverage': float(self.leverage_var.get()),
            'sl_pct': float(self.sl_var.get()),
            'tp_pct': float(self.tp_var.get()),
        }
        self.save_settings(api_key, secret, settings)

    def load_settings(self):
        """Loads user settings from json."""
        if os.path.exists("user_config.json"):
            try:
                with open("user_config.json", "r") as f:
                    data = json.load(f)
                    # Set StringVars, Entries update automatically
                    if "api_key" in data: self.api_key_var.set(data["api_key"])
                    if "secret" in data: self.secret_var.set(data["secret"])
                    if "deepseek_key" in data: self.deepseek_key_var.set(data["deepseek_key"])
                    
                    if "demo_mode" in data:
                        self.demo_var.set(data["demo_mode"])

                    # Trading Mode
                    # Trading Mode
                    saved_mode = data.get('trading_mode', 'Spot Margin (10x)')
                    if saved_mode not in ["Spot Margin (10x)", "Futures (50x)"]:
                        # Migrate old 10x to 5x if needed
                        saved_mode = "Spot Margin (10x)"
                    self.mode_var.set(saved_mode)

                    if "amount" in data:
                        self.amount_var.set(str(data["amount"]))
                    if "leverage" in data:
                        self.leverage_var.set(str(data["leverage"]))
                    if "sl_pct" in data:
                        self.sl_var.set(str(data["sl_pct"]))
                    if "tp_pct" in data:
                        self.tp_var.set(str(data["tp_pct"]))
                        
                self.log("Configuração carregada de user_config.json")
            except Exception as e:
                self.log(f"Falha ao carregar config: {e}")

    def save_settings(self, api_key, secret, settings):
        """Saves current settings to json."""
        data = {
            "api_key": api_key,
            "secret": secret,
            "deepseek_key": self.deepseek_key_var.get().strip(),
            "demo_mode": self.demo_var.get(),
            "trading_mode": self.mode_var.get(),
            **settings
        }
        try:
            with open("user_config.json", "w") as f:
                json.dump(data, f, indent=4)
            self.log("Configuração salva em user_config.json")
        except Exception as e:
            self.log(f"Falha ao salvar config: {e}")


    def update_leverage_limit(self, event=None):
        """Enforces leverage limits based on selected mode."""
        mode = self.mode_var.get()
        # Update both combos to match (sync them)
        if self.mode_combo_main.get() != mode:
            self.mode_combo_main.set(mode)
        if self.mode_combo_settings.get() != mode:
            self.mode_combo_settings.set(mode)
            
        current_lev = float(self.leverage_var.get())
        if "Spot" in mode:
            if current_lev > 10:
                self.leverage_var.set("10")
        else: # Futures
            # Max 50
            if current_lev > 50:
                self.leverage_var.set("50")

    def process_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_area.configure(state='normal')
                
                # Check for keywords to colorize
                tag = "info"
                lower_msg = msg.lower()
                if "buy" in lower_msg or "compra" in lower_msg:
                    tag = "buy"
                elif "sell" in lower_msg or "venda" in lower_msg:
                    tag = "sell"
                elif "error" in lower_msg or "crash" in lower_msg or "falha" in lower_msg:
                    tag = "error"
                    
                self.log_area.insert(tk.END, msg, tag)
                self.log_area.see(tk.END)
                self.log_area.configure(state='disabled')
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_log_queue)

    def update_ui_data(self):
        """Polls for Balance and History updates if trader is active."""
        if self.trader_instance and getattr(self.trader_instance, 'dm', None):
             try:
                # Balance
                bal = self.trader_instance.dm.get_balance()
                if bal is not None:
                    self.balance_var.set(f"Saldo: ${bal:.2f}")

                # Weekly PL
                w_pl = self.trader_instance.dm.get_weekly_pnl()
                self.weekly_pl_var.set(f"PL Semanal: ${w_pl:.2f}")
                
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
             except Exception as e:
                 print(f"UI Update Error: {e}")
                
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
            api_key = self.api_key_var.get().strip()
            secret = self.secret_var.get().strip()
            
            if not api_key or not secret:
                self.log("ERRO: API Key e Secret são obrigatórios.")
                return
                
            try:
                user_settings = {
                    'amount': float(self.amount_var.get()),
                    'leverage': float(self.leverage_var.get()),
                    'sl_pct': float(self.sl_var.get()),
                    'tp_pct': float(self.tp_var.get()),
                    'demo_mode': self.demo_var.get(),
                    'deepseek_key': self.deepseek_key_var.get().strip(),
                    'trading_mode': self.mode_var.get()
                }
            except ValueError:
                self.log("ERRO: Valores numéricos inválidos.")
                return

            # Start
            self.running_trader = True
            
            # Save Config
            self.save_settings(api_key, secret, user_settings)
            
            self.btn_run_trader.configure(text="PARAR Trading")
            self.status_var.set(f"Status: Trading Ativo ({user_settings['trading_mode']})")
            
            # Disable inputs (Optional, skipping for now to avoid AttributeError if widgets aren't self.x)
            # You can re-enable this if you bind the widgets to self.ent_amount again in create_widgets
            
            self.trader_thread = threading.Thread(target=self.run_trader_safe, args=(api_key, secret, user_settings), daemon=True)
            self.trader_thread.start()
        else:
            # Stop (Soft)
            self.log("Solicitando Parada... (Reinicie o App para reset total)")
            self.running_trader = False
            self.btn_run_trader.configure(text="INICIAR Trading (Ao Vivo)")
            self.status_var.set("Status: Parado")
            
            # Re-enable inputs


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

    def run_evolution_handler(self):
        """Runs the evolution worker in a separate thread."""
        self.log("Iniciando Laboratório de Estratégias (IA + Genética) em Background...")
        try:
            # We run the worker directly. Since sys.stdout is redirected, prints should show in GUI.
            thread = threading.Thread(target=self._run_evolution_safe)
            thread.daemon = True
            thread.start()
        except Exception as e:
            self.log(f"Falha ao iniciar evolução: {e}")

    def _run_evolution_safe(self):
        try:
            # Check if data loads first, might block
            evolution.evolution_worker() # This runs an infinite loop
        except Exception as e:
            self.log(f"ERRO CRÍTICO no Worker de Evolução: {e}")
            import traceback
            self.log(traceback.format_exc())


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
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                self.queue_writer.write(line.strip())
                
        self.log("Backtest Finalizado.")
        
    def poll_log_queue(self):
        """Polls the log queue and updates the GUI."""
        while not self.log_queue.empty():
            try:
                record = self.log_queue.get_nowait()
                self.log_area.configure(state='normal')
                self.log_area.insert(tk.END, record + '\n')
                self.log_area.see(tk.END)
                self.log_area.configure(state='disabled')
            except queue.Empty:
                break
        
        # Schedule next poll
        self.root.after(100, self.poll_log_queue)

    def start_trading(self):
        self.toggle_trader()

    def stop_trading(self):
        self.toggle_trader()

    def on_closing(self):
        if self.running_trader and hasattr(self, 'trader_instance') and self.trader_instance:
             # Try to stop threads if possible or just kill
             pass
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    root = tk.Tk()
    app = BitcoinAIApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
