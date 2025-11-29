"""
poker_gui.py

GUI-driven Poker Recursive Simulator + Trainer (Tkinter)

Features:
- Tkinter GUI with tabs: Config, Players, Simulation, Analysis, Training, Optimization
- Persist/load configuration (JSON)
- Create/edit player profiles, generate preset/synthetic players
- Run simulations (hands) with configurable rounds, blinds, recursion depth and MC samples
- Analyze played hands to estimate opponent stats (aggression, fold rate, bet freq)
- Train a small opponent model (SmallMLP via PyTorch) from simulated or imported hand logs
- Optimize hero profile parameters via an evolutionary strategy using simulation feedback
- Optional: uses eval7 for correct equity calculations if installed; torch optional for model training

Usage:
  python poker_gui.py

Requirements (recommended):
  pip install numpy rich tokenizers ftfy eval7 torch matplotlib

If torch or eval7 are not installed the app will run with degraded functionality (heuristic equity & no training).

This file is a single self-contained example; adapt & extend for your needs.
"""

from __future__ import annotations
import json
import random
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

# optional imports
try:
    import numpy as np
except Exception:
    np = None
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False
try:
    import eval7
    HAVE_EVAL7 = True
except Exception:
    HAVE_EVAL7 = False

# Compatibility wrapper for eval7.evaluate across versions
# Some eval7 versions expect a single list of cards (hand+board) while others accept two args (hand, board).
# Use this wrapper everywhere to be robust.
def eval7_evaluate(hand_cards, board_cards):
    """Evaluate a hand given lists of eval7.Card objects or card strings.

    If eval7 is not installed this function should not be called.
    Tries the two-argument API first and falls back to the single-list API on TypeError.
    """
    if not HAVE_EVAL7:
        raise RuntimeError("eval7 is not available")
    try:
        # try the two-argument signature (some versions support this)
        return eval7.evaluate(hand_cards, board_cards)
    except TypeError:
        # fallback: provide a single combined list
        return eval7.evaluate(hand_cards + board_cards)

    try:
        # try the two-argument signature (some versions support this)
        return eval7_evaluate(hand_cards, board_cards)
    except TypeError:
        # fallback: provide a single combined list
        return eval7.evaluate(hand_cards + board_cards)

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# ---------- Basic poker utilities (lightweight) ----------
RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = [r + s for r in RANKS for s in SUITS]

def make_deck():
    d = DECK.copy()
    random.shuffle(d)
    return d

# equity estimator (eval7 if available, else heuristic)
def estimate_equity(hero: List[str], board: List[str], n=200):
    if HAVE_EVAL7:
        hero_cards = [eval7.Card(c) for c in hero]
        board_cards = [eval7.Card(c) for c in board]
        deck = eval7.Deck()
        for c in hero_cards + board_cards:
            try:
                deck.cards.remove(c)
            except ValueError:
                pass
        wins = 0
        ties = 0
        n = max(10, n)
        for _ in range(n):
            opp = deck.sample(2)
            # fill board
            deck.shuffle()
            fill = []
            while len(board_cards) + len(fill) < 5:
                card = deck.sample(1)[0]
                if card not in hero_cards and card not in board_cards and card not in opp:
                    fill.append(card)
            full = board_cards + fill
            hscore = eval7_evaluate(hero_cards, full)
            oscore = eval7_evaluate(opp, full)
            if hscore > oscore:
                wins += 1
            elif hscore == oscore:
                ties += 1
        return (wins + 0.5 * ties) / n
    else:
        # fallback heuristic
        score = 0.5
        if hero[0][0] == hero[1][0]:
            score += 0.12
        for b in board:
            if b[0] in (hero[0][0], hero[1][0]):
                score += 0.05
        return min(0.99, max(0.01, score))

# ---------- Data classes for profiles and config ----------
@dataclass
class PlayerProfile:
    name: str = "P"
    stack: int = 1000
    aggression: float = 0.5
    erraticity: float = 0.0
    risk: float = 0.5
    vpip: float = 0.2
    pfr: float = 0.1
    is_hero: bool = False

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]):
        return PlayerProfile(**d)

@dataclass
class SimConfig:
    small_blind: int = 5
    big_blind: int = 10
    starting_stack: int = 1000
    rounds: int = 200
    recursion_depth: int = 1
    mc_samples: int = 150
    bet_sizes: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]):
        return SimConfig(**d)

# ---------- Simple in-memory poker engine (heads-up / multi simplified) ----------
class PlayerState:
    def __init__(self, profile: PlayerProfile):
        self.profile = profile
        self.hole: List[str] = []
        self.committed = 0
        self.folded = False

class SimplePokerEngine:
    def __init__(self, profiles: List[PlayerProfile], config: SimConfig):
        self.config = config
        self.profiles = [PlayerState(p) for p in profiles]
        self.deck: List[str] = []
        self.board: List[str] = []
        self.pot = 0
        self.hand_history: List[Dict[str, Any]] = []

    def reset_hand(self):
        self.deck = make_deck()
        self.board = []
        self.pot = 0
        for p in self.profiles:
            p.hole = []
            p.committed = 0
            p.folded = False

    def post_blinds(self):
        # simple: player0 = dealer, player1 = SB, player2 = BB etc
        n = len(self.profiles)
        if n < 2:
            return
        sb_idx = 1 % n
        bb_idx = 2 % n
        sb = min(self.profiles[sb_idx].profile.stack, self.config.small_blind)
        bb = min(self.profiles[bb_idx].profile.stack, self.config.big_blind)
        self.profiles[sb_idx].profile.stack -= sb
        self.profiles[bb_idx].profile.stack -= bb
        self.profiles[sb_idx].committed += sb
        self.profiles[bb_idx].committed += bb
        self.pot += sb + bb

    def deal_hole(self):
        for p in self.profiles:
            p.hole = [self.deck.pop(), self.deck.pop()]

    def deal_flop(self):
        # burn
        self.deck.pop()
        self.board += [self.deck.pop(), self.deck.pop(), self.deck.pop()]

    def deal_turn(self):
        self.deck.pop(); self.board.append(self.deck.pop())

    def deal_river(self):
        self.deck.pop(); self.board.append(self.deck.pop())

    def simulate_betting_simple(self):
        # each active player either fold/call/bet based on heuristic
        for p in self.profiles:
            if p.folded or p.profile.stack <= 0:
                continue
            fv = {"pot": self.pot, "to_call": 0, "equity": estimate_equity(p.hole, self.board, n=self.config.mc_samples//4)}
            probs = heuristic_policy_probabilities(fv, p.profile)
            act = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
            if act == 'fold':
                p.folded = True
            elif act == 'call':
                commit = min(p.profile.stack, 0)
                p.profile.stack -= commit
                p.committed += commit
                self.pot += commit
            elif act == 'bet':
                bet_amount = int(max(1, self.pot * random.choice(self.config.bet_sizes)))
                bet_amount = min(bet_amount, p.profile.stack)
                p.profile.stack -= bet_amount
                p.committed += bet_amount
                self.pot += bet_amount

    def showdown(self):
        active = [p for p in self.profiles if not p.folded]
        if len(active) == 1:
            winner = active[0]
            winner.profile.stack += self.pot
            return [(winner.profile.name, self.pot)]
        # else naive ranking
        scores = []
        for p in active:
            if HAVE_EVAL7:
                h = [eval7.Card(c) for c in p.hole]
                b = [eval7.Card(c) for c in self.board]
                s = eval7_evaluate(h, b)
            else:
                s = sum(RANKS.index(c[0]) for c in p.hole)
            scores.append((s, p))
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[0][0]
        winners = [p for s,p in scores if s==top]
        split = self.pot // len(winners)
        for w in winners:
            w.profile.stack += split
        return [(w.profile.name, split) for w in winners]

    def play_hand(self):
        self.reset_hand()
        self.post_blinds()
        self.deal_hole()
        self.simulate_betting_simple()
        self.deal_flop()
        self.simulate_betting_simple()
        self.deal_turn()
        self.simulate_betting_simple()
        self.deal_river()
        self.simulate_betting_simple()
        res = self.showdown()
        record = {"board": self.board.copy(), "holes": [p.hole.copy() for p in self.profiles], "pot": self.pot, "result": res}
        self.hand_history.append(record)
        return record

# ---------- heuristic opponent policy ----------
def heuristic_policy_probabilities(fv: Dict[str, float], profile: PlayerProfile):
    eq = fv.get('equity', 0.5)
    pot = fv.get('pot', 100)
    to_call = fv.get('to_call', 0)
    agg = profile.aggression
    risk = profile.risk
    err = profile.erraticity
    pot_odds = 0.0
    if to_call > 0:
        pot_odds = to_call / (pot + to_call + 1e-9)
    call_pref = max(0.01, min(0.99, 0.2 + (eq - pot_odds) * 2 + agg * 0.2))
    bet_pref = max(0.01, min(0.99, 0.05 + (eq - 0.5) * 2 + agg * 0.3 + risk * 0.2))
    fold_pref = max(0.01, 1.0 - call_pref - bet_pref)
    probs = {'fold': fold_pref, 'call': call_pref, 'bet': bet_pref}
    total = sum(probs.values())
    for k in probs:
        probs[k] = probs[k] / (total + 1e-9)
        probs[k] = probs[k] * (1 - err) + (err / 3.0)
    s = sum(probs.values())
    for k in probs:
        probs[k] /= s
    return probs

# ---------- Simple PyTorch model for opponent (optional) ----------
if HAVE_TORCH:
    class SmallMLP(nn.Module):
        def __init__(self, inp: int, hidden: int = 128, out: int = 3):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))
        def forward(self,x):
            return self.net(x)

# ---------- GUI Application ----------
class PokerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Poker Recursive Trainer')
        self.geometry('1000x700')
        self.config = SimConfig()
        self.profiles: List[PlayerProfile] = [PlayerProfile(name='Hero', is_hero=True), PlayerProfile(name='Villain')]
        self.engine: Optional[SimplePokerEngine] = None
        self.model = None
        self._build_ui()
        self.load_default_config()

    def _build_ui(self):
        tab_control = ttk.Notebook(self)
        self.tab_conf = ttk.Frame(tab_control)
        self.tab_players = ttk.Frame(tab_control)
        self.tab_sim = ttk.Frame(tab_control)
        self.tab_analysis = ttk.Frame(tab_control)
        self.tab_train = ttk.Frame(tab_control)
        self.tab_opt = ttk.Frame(tab_control)
        tab_control.add(self.tab_conf, text='Config')
        tab_control.add(self.tab_players, text='Players')
        tab_control.add(self.tab_sim, text='Simulation')
        tab_control.add(self.tab_analysis, text='Analysis')
        tab_control.add(self.tab_train, text='Training')
        tab_control.add(self.tab_opt, text='Optimize Hero')
        tab_control.pack(expand=1, fill='both')
        self._build_conf_tab()
        self._build_players_tab()
        self._build_sim_tab()
        self._build_analysis_tab()
        self._build_train_tab()
        self._build_opt_tab()

    # ---------- Config Tab ----------
    def _build_conf_tab(self):
        frame = self.tab_conf
        ttk.Label(frame, text='Small blind:').grid(row=0, column=0, sticky='w')
        self.sb_var = tk.IntVar(value=self.config.small_blind)
        ttk.Entry(frame, textvariable=self.sb_var).grid(row=0, column=1)
        ttk.Label(frame, text='Big blind:').grid(row=1, column=0, sticky='w')
        self.bb_var = tk.IntVar(value=self.config.big_blind)
        ttk.Entry(frame, textvariable=self.bb_var).grid(row=1, column=1)
        ttk.Label(frame, text='Starting stack:').grid(row=2, column=0, sticky='w')
        self.ss_var = tk.IntVar(value=self.config.starting_stack)
        ttk.Entry(frame, textvariable=self.ss_var).grid(row=2, column=1)
        ttk.Label(frame, text='Rounds:').grid(row=3, column=0, sticky='w')
        self.rounds_var = tk.IntVar(value=self.config.rounds)
        ttk.Entry(frame, textvariable=self.rounds_var).grid(row=3, column=1)
        ttk.Label(frame, text='Recursion depth (K):').grid(row=4, column=0, sticky='w')
        self.depth_var = tk.IntVar(value=self.config.recursion_depth)
        ttk.Entry(frame, textvariable=self.depth_var).grid(row=4, column=1)
        ttk.Label(frame, text='MC samples:').grid(row=5, column=0, sticky='w')
        self.mc_var = tk.IntVar(value=self.config.mc_samples)
        ttk.Entry(frame, textvariable=self.mc_var).grid(row=5, column=1)
        ttk.Button(frame, text='Save config', command=self.save_config_dialog).grid(row=6, column=0)
        ttk.Button(frame, text='Load config', command=self.load_config_dialog).grid(row=6, column=1)

    # ---------- Players Tab ----------
    def _build_players_tab(self):
        frame = self.tab_players
        left = ttk.Frame(frame)
        left.pack(side='left', fill='y', padx=8, pady=8)
        right = ttk.Frame(frame)
        right.pack(side='right', expand=1, fill='both', padx=8, pady=8)
        ttk.Label(left, text='Players').pack()
        self.players_listbox = tk.Listbox(left, height=20)
        self.players_listbox.pack()
        self.players_listbox.bind('<<ListboxSelect>>', self.on_player_select)
        btn_frame = ttk.Frame(left)
        btn_frame.pack(pady=6)
        ttk.Button(btn_frame, text='Add', command=self.add_player_dialog).grid(row=0, column=0)
        ttk.Button(btn_frame, text='Edit', command=self.edit_player_dialog).grid(row=0, column=1)
        ttk.Button(btn_frame, text='Delete', command=self.delete_player).grid(row=0, column=2)
        ttk.Button(btn_frame, text='Generate presets', command=self.generate_presets).grid(row=1, column=0, columnspan=3, pady=6)
        # right: detail
        ttk.Label(right, text='Player detail').pack()
        self.detail_text = tk.Text(right, height=20)
        self.detail_text.pack(expand=1, fill='both')
        self.refresh_players_list()

    def refresh_players_list(self):
        self.players_listbox.delete(0, 'end')
        for p in self.profiles:
            tag = ' (H)' if p.is_hero else ''
            self.players_listbox.insert('end', f"{p.name}{tag} | stack={p.stack} aggr={p.aggression:.2f} err={p.erraticity:.2f}")

    def on_player_select(self, evt=None):
        sel = self.players_listbox.curselection()
        if not sel:
            return
        i = sel[0]
        p = self.profiles[i]
        self.detail_text.delete('1.0', 'end')
        self.detail_text.insert('end', json.dumps(p.to_dict(), indent=2))

    def add_player_dialog(self):
        name = simpledialog.askstring('Name', 'Player name:')
        if not name:
            return
        p = PlayerProfile(name=name)
        self.profiles.append(p)
        self.refresh_players_list()

    def edit_player_dialog(self):
        sel = self.players_listbox.curselection()
        if not sel:
            messagebox.showinfo('Info', 'Select a player first')
            return
        i = sel[0]
        p = self.profiles[i]
        # simple edits using dialogs
        name = simpledialog.askstring('Name', 'Player name:', initialvalue=p.name)
        if not name:
            return
        p.name = name
        p.stack = simpledialog.askinteger('Stack', 'Stack', initialvalue=p.stack)
        p.aggression = float(simpledialog.askfloat('Aggression', 'Aggression 0..1', initialvalue=p.aggression))
        p.erraticity = float(simpledialog.askfloat('Erraticity', 'Erraticity 0..1', initialvalue=p.erraticity))
        p.risk = float(simpledialog.askfloat('Risk', 'Risk 0..1', initialvalue=p.risk))
        hero_choice = messagebox.askyesno('Hero', 'Make this player HERO?')
        p.is_hero = hero_choice
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

    # ---------- Simulation Tab ----------
    def _build_sim_tab(self):
        frame = self.tab_sim
        left = ttk.Frame(frame)
        left.pack(side='left', fill='y', padx=8, pady=8)
        right = ttk.Frame(frame)
        right.pack(side='right', expand=1, fill='both', padx=8, pady=8)
        ttk.Label(left, text='Simulation Controls').pack()
        ttk.Button(left, text='Run simulation', command=self.run_simulation_thread).pack(pady=4)
        ttk.Button(left, text='Clear history', command=self.clear_history).pack(pady=4)
        ttk.Button(left, text='Save history', command=self.save_history_dialog).pack(pady=4)
        ttk.Label(left, text='Rounds').pack()
        self.sim_rounds = tk.IntVar(value=self.config.rounds)
        ttk.Entry(left, textvariable=self.sim_rounds).pack()
        ttk.Label(left, text='MC samples').pack()
        self.sim_mc = tk.IntVar(value=self.config.mc_samples)
        ttk.Entry(left, textvariable=self.sim_mc).pack()
        ttk.Button(left, text='Import hand logs (JSON)', command=self.import_hand_logs).pack(pady=4)
        # right: quick stats
        ttk.Label(right, text='Simulation status').pack()
        self.sim_text = tk.Text(right, height=20)
        self.sim_text.pack(expand=1, fill='both')

    def run_simulation_thread(self):
        thr = threading.Thread(target=self.run_simulation)
        thr.start()

    def run_simulation(self):
        rounds = int(self.sim_rounds.get())
        mc = int(self.sim_mc.get())
        cfg = SimConfig(small_blind=self.sb_var.get(), big_blind=self.bb_var.get(), starting_stack=self.ss_var.get(), rounds=rounds, recursion_depth=self.depth_var.get(), mc_samples=mc)
        # set player stacks to starting
        for p in self.profiles:
            p.stack = cfg.starting_stack
        self.engine = SimplePokerEngine(self.profiles, cfg)
        self.sim_text.insert('end', f'Starting sim {rounds} hands...\n')
        self.sim_text.see('end')
        for i in range(rounds):
            rec = self.engine.play_hand()
            if i % max(1, rounds//10) == 0:
                self.sim_text.insert('end', f'Hand {i+1}/{rounds} done. Pot {rec["pot"] if "pot" in rec else rec.get("pot", 0)}\n')
                self.sim_text.see('end')
        self.sim_text.insert('end', 'Simulation finished.\n')
        self.sim_text.see('end')

    def clear_history(self):
        if self.engine:
            self.engine.hand_history.clear()
        self.sim_text.delete('1.0', 'end')

    def save_history_dialog(self):
        if not self.engine or not self.engine.hand_history:
            messagebox.showinfo('No data', 'No history to save')
            return
        fn = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON', '*.json')])
        if not fn:
            return
        Path(fn).write_text(json.dumps(self.engine.hand_history, indent=2), encoding='utf-8')
        messagebox.showinfo('Saved', f'Saved to {fn}')

    def import_hand_logs(self):
        fn = filedialog.askopenfilename(filetypes=[('JSON', '*.json')])
        if not fn:
            return
        data = json.loads(Path(fn).read_text(encoding='utf-8'))
        # naive: treat as engine history
        if not self.engine:
            self.engine = SimplePokerEngine(self.profiles, self.config)
        self.engine.hand_history = data
        messagebox.showinfo('Imported', f'Imported {len(data)} hands')

    # ---------- Analysis Tab ----------
    def _build_analysis_tab(self):
        frame = self.tab_analysis
        ttk.Button(frame, text='Analyze history', command=self.analyze_history).pack(pady=6)
        self.analysis_text = tk.Text(frame, height=30)
        self.analysis_text.pack(expand=1, fill='both')

    def analyze_history(self):
        self.analysis_text.delete('1.0', 'end')
        if not self.engine or not self.engine.hand_history:
            self.analysis_text.insert('end', 'No history to analyze. Run sim or import logs.\n')
            return
        freq = {}
        total = 0
        for hand in self.engine.hand_history:
            holes = hand.get('holes', [])
            for i, h in enumerate(holes):
                key = f'player_{i}'
                freq.setdefault(key, {'hands':0})
                freq[key]['hands'] += 1
            total += 1
        self.analysis_text.insert('end', f'Hands analyzed: {total}\n')
        # simple per-player stats (fold count, bet presence, etc)
        # note: our engine doesn't store actions; preliminary metrics are limited
        stacks = {p.profile.name: p.profile.stack for p in self.profiles}
        self.analysis_text.insert('end', f'Final stacks snapshot: {stacks}\n')
        # If data included bets or actions, we would compute aggression/fold rates

    # ---------- Training Tab ----------
    def _build_train_tab(self):
        frame = self.tab_train
        ttk.Label(frame, text='Opponent Model Training (supervised from logs or synthetic data)').pack()
        ttk.Button(frame, text='Generate synthetic training data (from sim)', command=self.generate_synth_training).pack(pady=4)
        ttk.Button(frame, text='Train model (PyTorch required)', command=self.train_model_thread).pack(pady=4)
        self.train_text = tk.Text(frame, height=20)
        self.train_text.pack(expand=1, fill='both')

    def generate_synth_training(self):
        # run some quick sims and convert to simple (features, action) dataset
        if not self.engine:
            self.engine = SimplePokerEngine(self.profiles, self.config)
        dataset = []
        N = simple_int_dialog('Number of hands for synth dataset', 500)
        self.train_text.insert('end', f'Generating {N} hands for synthetic dataset...\n')
        for i in range(N):
            rec = self.engine.play_hand()
            # derive fake label: randomly choose an action based on heuristic for each player
            for idx, p in enumerate(self.engine.profiles):
                fv = {'pot': rec.get('pot',0), 'equity': estimate_equity(p.hole, rec.get('board',[]), n=max(10,self.config.mc_samples//10))}
                probs = heuristic_policy_probabilities(fv, p.profile)
                action = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
                dataset.append({'features': fv, 'action': action})
        # save to memory
        self.synthetic_dataset = dataset
        self.train_text.insert('end', f'Synthetic dataset size: {len(dataset)}\n')

    def train_model_thread(self):
        if not HAVE_TORCH:
            messagebox.showerror('Torch missing', 'PyTorch not installed. Install torch to train models.')
            return
        thr = threading.Thread(target=self.train_model)
        thr.start()

    def train_model(self):
        self.train_text.insert('end', 'Training model...\n')
        # use synthetic dataset if present, else error
        data = getattr(self, 'synthetic_dataset', None)
        if not data:
            self.train_text.insert('end', 'No dataset: generate synthetic data first.\n')
            return
        # simple feature extraction: [equity, pot_norm]
        X = []
        y = []
        for rec in data:
            fv = rec['features']
            eq = fv.get('equity',0.5)
            pot = fv.get('pot',0)/1000.0
            X.append([eq, pot])
            act = rec['action']
            y.append({'fold':0,'call':1,'bet':2}[act])
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        model = SmallMLP(inp=2, hidden=64, out=3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        epochs = 10
        for ep in range(epochs):
            model.train()
            logits = model(X)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            self.train_text.insert('end', f'epoch {ep+1}/{epochs} loss={loss.item():.4f}\n')
        self.model = model
        torch.save(model.state_dict(), 'opp_model.pt')
        self.train_text.insert('end', 'Training finished and saved to opp_model.pt\n')

    # ---------- Optimization Tab ----------
    def _build_opt_tab(self):
        frame = self.tab_opt
        ttk.Label(frame, text='Optimize HERO profile via evolutionary search').pack()
        ttk.Button(frame, text='Run optimization', command=self.optimize_hero_thread).pack(pady=4)
        self.opt_text = tk.Text(frame, height=20)
        self.opt_text.pack(expand=1, fill='both')

    def optimize_hero_thread(self):
        thr = threading.Thread(target=self.optimize_hero)
        thr.start()

    def optimize_hero(self):
        # find hero index
        heroes = [p for p in self.profiles if p.is_hero]
        if not heroes:
            messagebox.showerror('No hero', 'Mark a player as hero in Players tab')
            return
        hero = heroes[0]
        pop = 10
        gens = simple_int_dialog('Generations', 6)
        rounds = simple_int_dialog('Rounds per candidate', 200)
        self.opt_text.insert('end', f'Starting optimization: pop={pop} gens={gens} rounds={rounds}\n')
        # param vector: aggression, erraticity, risk
        def eval_candidate(vec):
            # set hero profile temporarily
            original = (hero.aggression, hero.erraticity, hero.risk)
            hero.aggression, hero.erraticity, hero.risk = vec
            # run quick sim
            cfg = SimConfig(small_blind=self.sb_var.get(), big_blind=self.bb_var.get(), starting_stack=self.ss_var.get(), rounds=rounds, recursion_depth=self.depth_var.get(), mc_samples=self.mc_var.get())
            engine = SimplePokerEngine(self.profiles, cfg)
            # reset stacks
            for p in self.profiles:
                p.profile.stack = cfg.starting_stack
            total_profit = 0
            for _ in range(rounds):
                engine.play_hand()
            # find hero profit
            hp = [p for p in engine.profiles if p.profile.is_hero][0]
            profit = hp.profile.stack - cfg.starting_stack
            # restore original
            hero.aggression, hero.erraticity, hero.risk = original
            return profit
        # initialize population around current hero
        cur = [hero.aggression, hero.erraticity, hero.risk]
        population = []
        for i in range(pop):
            candidate = [min(1.0,max(0.0, cur[j] + random.uniform(-0.2,0.2))) for j in range(3)]
            population.append(candidate)
        for g in range(gens):
            scores = []
            for cand in population:
                score = eval_candidate(cand)
                scores.append((score, cand))
                self.opt_text.insert('end', f'cand {cand} -> profit {score}\n')
                self.opt_text.see('end')
            scores.sort(key=lambda x: x[0], reverse=True)
            best = scores[0]
            self.opt_text.insert('end', f'Gen {g+1} best {best}\n')
            # breed new population
            newpop = [best[1]]
            while len(newpop) < pop:
                a = random.choice(scores)[1]
                b = random.choice(scores)[1]
                child = [(a[i]+b[i])/2.0 + random.uniform(-0.05,0.05) for i in range(3)]
                child = [min(1.0, max(0.0, x)) for x in child]
                newpop.append(child)
            population = newpop
        champion = population[0]
        self.opt_text.insert('end', f'Optimization finished. Champion: {champion}\n')
        # apply champion
        hero.aggression, hero.erraticity, hero.risk = champion
        self.refresh_players_list()

    # ---------- Config persistence ----------
    def save_config_dialog(self):
        fn = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON','*.json')])
        if not fn:
            return
        data = {'config': self.config.to_dict(), 'profiles': [p.to_dict() for p in self.profiles]}
        Path(fn).write_text(json.dumps(data, indent=2), encoding='utf-8')
        messagebox.showinfo('Saved', f'Saved config to {fn}')

    def load_config_dialog(self):
        fn = filedialog.askopenfilename(filetypes=[('JSON','*.json')])
        if not fn:
            return
        data = json.loads(Path(fn).read_text(encoding='utf-8'))
        self.config = SimConfig.from_dict(data.get('config', {}))
        self.profiles = [PlayerProfile.from_dict(d) for d in data.get('profiles', [])]
        # update UI
        self.sb_var.set(self.config.small_blind)
        self.bb_var.set(self.config.big_blind)
        self.ss_var.set(self.config.starting_stack)
        self.rounds_var.set(self.config.rounds)
        self.depth_var.set(self.config.recursion_depth)
        self.mc_var.set(self.config.mc_samples)
        self.refresh_players_list()
        messagebox.showinfo('Loaded', f'Loaded config from {fn}')

    def load_default_config(self):
        if DEFAULT_CONFIG.exists():
            try:
                data = json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
                self.config = SimConfig.from_dict(data.get('config', {}))
                self.profiles = [PlayerProfile.from_dict(d) for d in data.get('profiles', [])]
                self.refresh_players_list()
            except Exception:
                pass

# helper dialogs
DEFAULT_CONFIG = Path('poker_gui_config.json')

def simple_int_dialog(prompt: str, default: int) -> int:
    try:
        val = simpledialog.askinteger('Input', prompt, initialvalue=default)
        return int(val) if val is not None else default
    except Exception:
        return default

# ---------- main ----------
if __name__ == '__main__':
    app = PokerApp()
    app.mainloop()
