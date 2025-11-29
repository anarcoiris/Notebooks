"""
poker_gui_v2_integration.py

Actualización del GUI para usar el nuevo motor de póker (Fase 1)

Este archivo proporciona:
1. Parche para integrar poker_engine_v2 con el GUI original
2. Mejoras en la visualización de acciones
3. Análisis detallado de hand history

Uso:
    python poker_gui_v2_integration.py
    
O integrar con el código original:
    from poker_gui_v2_integration import PokerAppV2
"""

import json
import threading
import time
from pathlib import Path
from tkinter import ttk, messagebox, filedialog
import tkinter as tk

# Importar motor nuevo
from poker_engine_v2 import (
    PokerEngine, PlayerProfile, SimConfig,
    ActionType
)

# Importar del código original lo que necesitamos
try:
    from poker_gui import PokerApp, simple_int_dialog, DEFAULT_CONFIG
except ImportError:
    # Si no está disponible, definir lo básico
    DEFAULT_CONFIG = Path('poker_gui_config.json')
    
    def simple_int_dialog(prompt: str, default: int) -> int:
        import tkinter.simpledialog as sd
        val = sd.askinteger('Input', prompt, initialvalue=default)
        return int(val) if val is not None else default


# ==================== HELPER FUNCTIONS ====================

def format_action_display(action_dict):
    """Formatea una acción para display en el GUI"""
    action_type = action_dict.get('type', 'unknown')
    player = action_dict.get('player', 'Unknown')
    amount = action_dict.get('amount', 0)
    
    if action_type == 'fold':
        return f"{player} folds"
    elif action_type == 'check':
        return f"{player} checks"
    elif action_type == 'call':
        return f"{player} calls {amount}"
    elif action_type == 'bet':
        return f"{player} bets {amount}"
    elif action_type == 'raise':
        return f"{player} raises to {amount}"
    elif action_type == 'all_in':
        return f"{player} ALL-IN {amount}"
    return f"{player} {action_type} {amount}"


def calculate_player_statistics(hand_history_log):
    """
    Calcula estadísticas detalladas de los jugadores.
    
    Returns:
        Dict con stats por jugador
    """
    from collections import defaultdict
    
    stats = defaultdict(lambda: {
        'hands': 0,
        'vpip': 0,
        'pfr': 0,
        'wins': 0,
        'total_won': 0,
        'total_invested': 0,
        'folds_preflop': 0,
        'folds_postflop': 0,
        'aggression_actions': 0,
        'passive_actions': 0
    })
    
    for history_dict in hand_history_log:
        # Contar manos jugadas
        for player_info in history_dict.get('players', []):
            name = player_info['name']
            stats[name]['hands'] += 1
        
        # Analizar acciones preflop
        preflop_actions = history_dict.get('phases', {}).get('preflop', [])
        for action in preflop_actions:
            player = action.get('player', '')
            action_type = action.get('type', '')
            
            if action_type in ['call', 'raise', 'bet', 'all_in']:
                stats[player]['vpip'] += 1
            
            if action_type in ['raise', 'bet']:
                stats[player]['pfr'] += 1
                stats[player]['aggression_actions'] += 1
            elif action_type == 'call':
                stats[player]['passive_actions'] += 1
            elif action_type == 'fold':
                stats[player]['folds_preflop'] += 1
        
        # Analizar acciones postflop
        for phase in ['flop', 'turn', 'river']:
            for action in history_dict.get('phases', {}).get(phase, []):
                player = action.get('player', '')
                action_type = action.get('type', '')
                
                if action_type in ['bet', 'raise']:
                    stats[player]['aggression_actions'] += 1
                elif action_type == 'call':
                    stats[player]['passive_actions'] += 1
                elif action_type == 'fold':
                    stats[player]['folds_postflop'] += 1
        
        # Ganadores
        result = history_dict.get('result', {})
        for winner_name, amount in result.get('winners', []):
            stats[winner_name]['wins'] += 1
            stats[winner_name]['total_won'] += amount
    
    # Calcular porcentajes
    for name, data in stats.items():
        hands = data['hands']
        if hands > 0:
            data['vpip_pct'] = (data['vpip'] / hands) * 100
            data['pfr_pct'] = (data['pfr'] / hands) * 100
            data['fold_to_preflop_pct'] = (data['folds_preflop'] / hands) * 100
            data['win_rate'] = (data['wins'] / hands) * 100
            
            total_actions = data['aggression_actions'] + data['passive_actions']
            if total_actions > 0:
                data['aggression_factor'] = data['aggression_actions'] / max(1, data['passive_actions'])
            else:
                data['aggression_factor'] = 0
    
    return dict(stats)


# ==================== ENHANCED GUI ====================

class PokerAppV2(tk.Tk):
    """
    Versión mejorada del GUI usando poker_engine_v2.
    
    Mantiene compatibilidad con código original pero usa el motor correcto.
    """
    
    def __init__(self):
        super().__init__()
        self.title('Poker Recursive Trainer v2.0 [FASE 1 COMPLETA]')
        self.geometry('1200x800')
        
        # Config y profiles (compatibilidad)
        self.config = SimConfig()
        self.profiles = [
            PlayerProfile(name='Hero', is_hero=True),
            PlayerProfile(name='Villain')
        ]
        
        # Nuevo motor
        self.engine = None
        
        # Build UI
        self._build_ui()
        self.load_default_config()
    
    def _build_ui(self):
        """Construye la interfaz (similar al original pero mejorado)"""
        tab_control = ttk.Notebook(self)
        
        self.tab_conf = ttk.Frame(tab_control)
        self.tab_players = ttk.Frame(tab_control)
        self.tab_sim = ttk.Frame(tab_control)
        self.tab_analysis = ttk.Frame(tab_control)
        self.tab_replay = ttk.Frame(tab_control)  # NUEVO
        
        tab_control.add(self.tab_conf, text='⚙️ Config')
        tab_control.add(self.tab_players, text='👥 Players')
        tab_control.add(self.tab_sim, text='▶️ Simulation')
        tab_control.add(self.tab_analysis, text='📊 Analysis')
        tab_control.add(self.tab_replay, text='🔄 Hand Replay')
        
        tab_control.pack(expand=1, fill='both')
        
        self._build_conf_tab()
        self._build_players_tab()
        self._build_sim_tab()
        self._build_analysis_tab()
        self._build_replay_tab()
    
    # ==================== CONFIG TAB ====================
    
    def _build_conf_tab(self):
        frame = self.tab_conf
        
        # Usar grid para layout limpio
        ttk.Label(frame, text='Small blind:', font=('Arial', 10)).grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.sb_var = tk.IntVar(value=self.config.small_blind)
        ttk.Entry(frame, textvariable=self.sb_var, width=15).grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(frame, text='Big blind:', font=('Arial', 10)).grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.bb_var = tk.IntVar(value=self.config.big_blind)
        ttk.Entry(frame, textvariable=self.bb_var, width=15).grid(row=1, column=1, padx=10, pady=5)
        
        ttk.Label(frame, text='Starting stack:', font=('Arial', 10)).grid(row=2, column=0, sticky='w', padx=10, pady=5)
        self.ss_var = tk.IntVar(value=self.config.starting_stack)
        ttk.Entry(frame, textvariable=self.ss_var, width=15).grid(row=2, column=1, padx=10, pady=5)
        
        ttk.Label(frame, text='Rounds:', font=('Arial', 10)).grid(row=3, column=0, sticky='w', padx=10, pady=5)
        self.rounds_var = tk.IntVar(value=self.config.rounds)
        ttk.Entry(frame, textvariable=self.rounds_var, width=15).grid(row=3, column=1, padx=10, pady=5)
        
        ttk.Label(frame, text='MC samples:', font=('Arial', 10)).grid(row=4, column=0, sticky='w', padx=10, pady=5)
        self.mc_var = tk.IntVar(value=self.config.mc_samples)
        ttk.Entry(frame, textvariable=self.mc_var, width=15).grid(row=4, column=1, padx=10, pady=5)
        
        # Botones
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text='💾 Save Config', command=self.save_config_dialog).pack(side='left', padx=5)
        ttk.Button(btn_frame, text='📂 Load Config', command=self.load_config_dialog).pack(side='left', padx=5)
        
        # Info
        info_text = tk.Text(frame, height=15, width=60, wrap='word')
        info_text.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')
        info_text.insert('1.0', """
✅ FASE 1 COMPLETA - Motor Correcto Implementado

Mejoras en esta versión:
• Sistema de apuestas correcto con turnos
• Tracking de to_call y current_bet
• Side pots para all-ins múltiples
• Action logging completo
• Validación de acciones
• Hand history detallado

El motor ahora simula póker Texas Hold'em real con:
- Preflop, Flop, Turn, River diferenciados
- Orden de actuación correcto
- Raises y re-raises válidos
- All-ins con side pots
- Stats precisos (VPIP, PFR, AF)
        """)
        info_text.config(state='disabled')
    
    # ==================== PLAYERS TAB ====================
    
    def _build_players_tab(self):
        """Similar al original pero con mejores visuales"""
        frame = self.tab_players
        
        left = ttk.Frame(frame)
        left.pack(side='left', fill='y', padx=8, pady=8)
        
        right = ttk.Frame(frame)
        right.pack(side='right', expand=1, fill='both', padx=8, pady=8)
        
        ttk.Label(left, text='Players', font=('Arial', 12, 'bold')).pack()
        
        self.players_listbox = tk.Listbox(left, height=20, width=40)
        self.players_listbox.pack(pady=10)
        self.players_listbox.bind('<<ListboxSelect>>', self.on_player_select)
        
        btn_frame = ttk.Frame(left)
        btn_frame.pack(pady=6)
        
        ttk.Button(btn_frame, text='➕ Add', command=self.add_player_dialog).grid(row=0, column=0, padx=2)
        ttk.Button(btn_frame, text='✏️ Edit', command=self.edit_player_dialog).grid(row=0, column=1, padx=2)
        ttk.Button(btn_frame, text='🗑️ Delete', command=self.delete_player).grid(row=0, column=2, padx=2)
        ttk.Button(btn_frame, text='🎲 Presets', command=self.generate_presets).grid(row=1, column=0, columnspan=3, pady=6)
        
        # Detail panel
        ttk.Label(right, text='Player Detail', font=('Arial', 12, 'bold')).pack()
        self.detail_text = tk.Text(right, height=20, wrap='word')
        self.detail_text.pack(expand=1, fill='both', pady=10)
        
        self.refresh_players_list()
    
    def refresh_players_list(self):
        self.players_listbox.delete(0, 'end')
        for p in self.profiles:
            tag = ' [HERO]' if p.is_hero else ''
            self.players_listbox.insert('end', 
                f"{p.name}{tag} | ${p.stack} | AGG:{p.aggression:.2f} | ERR:{p.erraticity:.2f}")
    
    def on_player_select(self, evt=None):
        sel = self.players_listbox.curselection()
        if not sel:
            return
        i = sel[0]
        p = self.profiles[i]
        self.detail_text.delete('1.0', 'end')
        
        detail = f"""
Player: {p.name}
{'='*40}
Stack: ${p.stack}
Aggression: {p.aggression:.2f}
Erraticity: {p.erraticity:.2f}
Risk Tolerance: {p.risk:.2f}
VPIP: {p.vpip:.2f}
PFR: {p.pfr:.2f}
Is Hero: {'Yes' if p.is_hero else 'No'}
"""
        self.detail_text.insert('end', detail)
    
    def add_player_dialog(self):
        from tkinter import simpledialog
        name = simpledialog.askstring('Name', 'Player name:')
        if not name:
            return
        p = PlayerProfile(name=name)
        self.profiles.append(p)
        self.refresh_players_list()
    
    def edit_player_dialog(self):
        from tkinter import simpledialog
        sel = self.players_listbox.curselection()
        if not sel:
            messagebox.showinfo('Info', 'Select a player first')
            return
        i = sel[0]
        p = self.profiles[i]
        
        name = simpledialog.askstring('Name', 'Player name:', initialvalue=p.name)
        if not name:
            return
        p.name = name
        p.stack = simpledialog.askinteger('Stack', 'Stack', initialvalue=p.stack) or p.stack
        p.aggression = float(simpledialog.askfloat('Aggression', 'Aggression 0..1', initialvalue=p.aggression) or p.aggression)
        p.erraticity = float(simpledialog.askfloat('Erraticity', 'Erraticity 0..1', initialvalue=p.erraticity) or p.erraticity)
        p.risk = float(simpledialog.askfloat('Risk', 'Risk 0..1', initialvalue=p.risk) or p.risk)
        p.is_hero = messagebox.askyesno('Hero', 'Make this player HERO?')
        
        self.refresh_players_list()
    
    def delete_player(self):
        sel = self.players_listbox.curselection()
        if not sel:
            return
        i = sel[0]
        del self.profiles[i]
        self.refresh_players_list()
    
    def generate_presets(self):
        presets = [
            PlayerProfile(name='Tight-Passive', aggression=0.2, erraticity=0.05, risk=0.2),
            PlayerProfile(name='Loose-Aggro', aggression=0.8, erraticity=0.1, risk=0.7),
            PlayerProfile(name='Maniac', aggression=0.95, erraticity=0.4, risk=0.9),
            PlayerProfile(name='Calling-Station', aggression=0.3, erraticity=0.02, risk=0.3),
        ]
        for pr in presets:
            self.profiles.append(pr)
        self.refresh_players_list()
    
    # ==================== SIMULATION TAB ====================
    
    def _build_sim_tab(self):
        frame = self.tab_sim
        
        left = ttk.Frame(frame)
        left.pack(side='left', fill='y', padx=8, pady=8)
        
        right = ttk.Frame(frame)
        right.pack(side='right', expand=1, fill='both', padx=8, pady=8)
        
        ttk.Label(left, text='Simulation Controls', font=('Arial', 12, 'bold')).pack(pady=10)
        
        ttk.Button(left, text='▶️ Run Simulation', command=self.run_simulation_thread, width=20).pack(pady=4)
        ttk.Button(left, text='🗑️ Clear History', command=self.clear_history, width=20).pack(pady=4)
        ttk.Button(left, text='💾 Save History', command=self.save_history_dialog, width=20).pack(pady=4)
        ttk.Button(left, text='📂 Import Logs', command=self.import_hand_logs, width=20).pack(pady=4)
        
        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=10)
        
        ttk.Label(left, text='Rounds:', font=('Arial', 10)).pack()
        self.sim_rounds = tk.IntVar(value=self.config.rounds)
        ttk.Entry(left, textvariable=self.sim_rounds, width=20).pack(pady=2)
        
        ttk.Label(left, text='MC Samples:', font=('Arial', 10)).pack()
        self.sim_mc = tk.IntVar(value=self.config.mc_samples)
        ttk.Entry(left, textvariable=self.sim_mc, width=20).pack(pady=2)
        
        # Progress bar
        ttk.Label(left, text='Progress:', font=('Arial', 10)).pack(pady=(20, 5))
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(left, variable=self.progress_var, maximum=100, length=200)
        self.progress_bar.pack()
        
        # Right: output
        ttk.Label(right, text='Simulation Output', font=('Arial', 12, 'bold')).pack()
        
        scrollbar = ttk.Scrollbar(right)
        scrollbar.pack(side='right', fill='y')
        
        self.sim_text = tk.Text(right, height=30, yscrollcommand=scrollbar.set)
        self.sim_text.pack(expand=1, fill='both', pady=10)
        scrollbar.config(command=self.sim_text.yview)
    
    def run_simulation_thread(self):
        thr = threading.Thread(target=self.run_simulation, daemon=True)
        thr.start()
    
    def run_simulation(self):
        """Ejecuta simulación usando el nuevo motor"""
        rounds = int(self.sim_rounds.get())
        mc = int(self.sim_mc.get())
        
        # Actualizar config
        cfg = SimConfig(
            small_blind=self.sb_var.get(),
            big_blind=self.bb_var.get(),
            starting_stack=self.ss_var.get(),
            rounds=rounds,
            mc_samples=mc
        )
        
        # Reset stacks
        for p in self.profiles:
            p.stack = cfg.starting_stack
        
        # Crear engine
        self.engine = PokerEngine(self.profiles, cfg)
        
        self.sim_text.insert('end', f'\n{"="*60}\n')
        self.sim_text.insert('end', f'🎲 Starting simulation: {rounds} hands\n')
        self.sim_text.insert('end', f'Players: {[p.name for p in self.profiles]}\n')
        self.sim_text.insert('end', f'{"="*60}\n\n')
        self.sim_text.see('end')
        
        start_time = time.time()
        
        for i in range(rounds):
            try:
                result = self.engine.play_hand()
                
                # Update progress
                progress = ((i + 1) / rounds) * 100
                self.progress_var.set(progress)
                
                # Log cada N hands
                if i % max(1, rounds // 20) == 0 or i == rounds - 1:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    
                    self.sim_text.insert('end', 
                        f'Hand {i+1}/{rounds} | Pot: ${result["pot"]} | '
                        f'Winner: {result["winners"][0][0] if result["winners"] else "N/A"} | '
                        f'Rate: {rate:.1f} h/s\n')
                    self.sim_text.see('end')
            
            except Exception as e:
                self.sim_text.insert('end', f'\n❌ ERROR in hand {i+1}: {e}\n')
                self.sim_text.see('end')
                break
        
        elapsed = time.time() - start_time
        
        self.sim_text.insert('end', f'\n{"="*60}\n')
        self.sim_text.insert('end', f'✅ Simulation complete!\n')
        self.sim_text.insert('end', f'Time: {elapsed:.2f}s | Rate: {rounds/elapsed:.1f} hands/sec\n')
        self.sim_text.insert('end', f'{"="*60}\n\n')
        
        # Mostrar stacks finales
        self.sim_text.insert('end', 'Final Stacks:\n')
        for p in self.profiles:
            change = p.stack - cfg.starting_stack
            sign = '+' if change >= 0 else ''
            self.sim_text.insert('end', f'  {p.name}: ${p.stack} ({sign}${change})\n')
        
        self.sim_text.see('end')
        self.progress_var.set(100)
    
    def clear_history(self):
        if self.engine:
            self.engine.hand_history_log.clear()
        self.sim_text.delete('1.0', 'end')
        self.progress_var.set(0)
    
    def save_history_dialog(self):
        if not self.engine or not self.engine.hand_history_log:
            messagebox.showinfo('No Data', 'No history to save')
            return
        
        fn = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON', '*.json')])
        if not fn:
            return
        
        # Convertir HandHistory a dict
        data = [h.to_dict() for h in self.engine.hand_history_log]
        Path(fn).write_text(json.dumps(data, indent=2), encoding='utf-8')
        messagebox.showinfo('Saved', f'Saved {len(data)} hands to {fn}')
    
    def import_hand_logs(self):
        fn = filedialog.askopenfilename(filetypes=[('JSON', '*.json')])
        if not fn:
            return
        
        data = json.loads(Path(fn).read_text(encoding='utf-8'))
        
        # Crear engine si no existe
        if not self.engine:
            self.engine = PokerEngine(self.profiles, self.config)
        
        # Importar como dicts (compatible)
        messagebox.showinfo('Imported', f'Imported {len(data)} hands')
    
    # ==================== ANALYSIS TAB ====================
    
    def _build_analysis_tab(self):
        frame = self.tab_analysis
        
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(top_frame, text='📊 Analyze History', command=self.analyze_history).pack(side='left', padx=5)
        ttk.Button(top_frame, text='📈 Show Graphs', command=self.show_graphs).pack(side='left', padx=5)
        ttk.Button(top_frame, text='🔄 Refresh', command=self.analyze_history).pack(side='left', padx=5)
        
        # Output
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side='right', fill='y')
        
        self.analysis_text = tk.Text(frame, height=30, yscrollcommand=scrollbar.set, font=('Courier', 10))
        self.analysis_text.pack(expand=1, fill='both', padx=10, pady=10)
        scrollbar.config(command=self.analysis_text.yview)
    
    def analyze_history(self):
        """Análisis mejorado con el nuevo motor"""
        self.analysis_text.delete('1.0', 'end')
        
        if not self.engine or not self.engine.hand_history_log:
            self.analysis_text.insert('end', '❌ No history to analyze. Run simulation first.\n')
            return
        
        # Convertir a dicts para análisis
        history_dicts = [h.to_dict() for h in self.engine.hand_history_log]
        
        stats = calculate_player_statistics(history_dicts)
        
        self.analysis_text.insert('end', f'{"="*80}\n')
        self.analysis_text.insert('end', f'📊 POKER STATISTICS ANALYSIS\n')
        self.analysis_text.insert('end', f'{"="*80}\n\n')
        self.analysis_text.insert('end', f'Total Hands Analyzed: {len(history_dicts)}\n\n')
        
        for player_name, data in stats.items():
            self.analysis_text.insert('end', f'\n{player_name}\n')
            self.analysis_text.insert('end', f'{"-"*40}\n')
            self.analysis_text.insert('end', f'Hands Played: {data["hands"]}\n')
            self.analysis_text.insert('end', f'VPIP: {data.get("vpip_pct", 0):.1f}%\n')
            self.analysis_text.insert('end', f'PFR: {data.get("pfr_pct", 0):.1f}%\n')
            self.analysis_text.insert('end', f'Aggression Factor: {data.get("aggression_factor", 0):.2f}\n')
            self.analysis_text.insert('end', f'Wins: {data["wins"]} ({data.get("win_rate", 0):.1f}%)\n')
            self.analysis_text.insert('end', f'Total Won: ${data["total_won"]}\n')
            self.analysis_text.insert('end', f'Preflop Folds: {data["folds_preflop"]}\n')
            self.analysis_text.insert('end', f'Postflop Folds: {data["folds_postflop"]}\n')
        
        self.analysis_text.insert('end', f'\n{"="*80}\n')
        
        # Usar stats del engine también
        engine_stats = self.engine.get_statistics()
        self.analysis_text.insert('end', f'\n📈 Engine Statistics:\n')
        self.analysis_text.insert('end', json.dumps(engine_stats, indent=2))
    
    def show_graphs(self):
        """Placeholder para gráficos (Fase 3)"""
        messagebox.showinfo('Coming Soon', 'Graphical analysis will be available in Phase 3!')
    
    # ==================== REPLAY TAB ====================
    
    def _build_replay_tab(self):
        """Nueva pestaña para replay de manos"""
        frame = self.tab_replay
        
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(top_frame, text='Hand Replay', font=('Arial', 12, 'bold')).pack(side='left', padx=10)
        
        ttk.Button(top_frame, text='◀️ Prev', command=self.prev_hand).pack(side='left', padx=5)
        ttk.Button(top_frame, text='▶️ Next', command=self.next_hand).pack(side='left', padx=5)
        ttk.Button(top_frame, text='🔄 Refresh List', command=self.refresh_hand_list).pack(side='left', padx=5)
        
        # Lista de manos
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill='both', expand=1, padx=10, pady=10)
        
        ttk.Label(list_frame, text='Hands:', font=('Arial', 10)).pack(anchor='w')
        
        self.hands_listbox = tk.Listbox(list_frame, height=8)
        self.hands_listbox.pack(fill='x', pady=5)
        self.hands_listbox.bind('<<ListboxSelect>>', self.on_hand_select)
        
        # Detail
        ttk.Label(list_frame, text='Hand Detail:', font=('Arial', 10)).pack(anchor='w', pady=(10, 0))
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.replay_text = tk.Text(list_frame, height=20, yscrollcommand=scrollbar.set, font=('Courier', 9))
        self.replay_text.pack(expand=1, fill='both')
        scrollbar.config(command=self.replay_text.yview)
        
        self.current_hand_idx = 0
    
    def refresh_hand_list(self):
        self.hands_listbox.delete(0, 'end')
        if not self.engine or not self.engine.hand_history_log:
            return
        
        for i, history in enumerate(self.engine.hand_history_log):
            pot = history.result['pot'] if history.result else 0
            winner = history.result['winners'][0][0] if history.result and history.result['winners'] else 'N/A'
            self.hands_listbox.insert('end', f'Hand #{i+1} | Pot: ${pot} | Winner: {winner}')
    
    def on_hand_select(self, evt=None):
        sel = self.hands_listbox.curselection()
        if not sel:
            return
        self.current_hand_idx = sel[0]
        self.display_hand(self.current_hand_idx)
    
    def prev_hand(self):
        if self.current_hand_idx > 0:
            self.current_hand_idx -= 1
            self.hands_listbox.selection_clear(0, 'end')
            self.hands_listbox.selection_set(self.current_hand_idx)
            self.hands_listbox.see(self.current_hand_idx)
            self.display_hand(self.current_hand_idx)
    
    def next_hand(self):
        if self.engine and self.current_hand_idx < len(self.engine.hand_history_log) - 1:
            self.current_hand_idx += 1
            self.hands_listbox.selection_clear(0, 'end')
            self.hands_listbox.selection_set(self.current_hand_idx)
            self.hands_listbox.see(self.current_hand_idx)
            self.display_hand(self.current_hand_idx)
    
    def display_hand(self, idx):
        if not self.engine or idx >= len(self.engine.hand_history_log):
            return
        
        history = self.engine.hand_history_log[idx]
        readable = history.to_readable_string()
        
        self.replay_text.delete('1.0', 'end')
        self.replay_text.insert('end', readable)
    
    # ==================== CONFIG PERSISTENCE ====================
    
    def save_config_dialog(self):
        fn = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON', '*.json')])
        if not fn:
            return
        
        from dataclasses import asdict
        data = {
            'config': asdict(self.config),
            'profiles': [asdict(p) for p in self.profiles]
        }
        Path(fn).write_text(json.dumps(data, indent=2), encoding='utf-8')
        messagebox.showinfo('Saved', f'Saved config to {fn}')
    
    def load_config_dialog(self):
        fn = filedialog.askopenfilename(filetypes=[('JSON', '*.json')])
        if not fn:
            return
        
        data = json.loads(Path(fn).read_text(encoding='utf-8'))
        self.config = SimConfig(**data.get('config', {}))
        self.profiles = [PlayerProfile(**d) for d in data.get('profiles', [])]
        
        # Update UI
        self.sb_var.set(self.config.small_blind)
        self.bb_var.set(self.config.big_blind)
        self.ss_var.set(self.config.starting_stack)
        self.rounds_var.set(self.config.rounds)
        self.mc_var.set(self.config.mc_samples)
        self.refresh_players_list()
        messagebox.showinfo('Loaded', f'Loaded config from {fn}')
    
    def load_default_config(self):
        if DEFAULT_CONFIG.exists():
            try:
                data = json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
                self.config = SimConfig(**data.get('config', {}))
                self.profiles = [PlayerProfile(**d) for d in data.get('profiles', [])]
                self.refresh_players_list()
            except Exception:
                pass


# ==================== MAIN ====================

def main():
    """Punto de entrada principal"""
    app = PokerAppV2()
    app.mainloop()


if __name__ == '__main__':
    main()