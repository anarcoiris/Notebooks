#!/usr/bin/env python3
"""
poker_recursive_cli.py

Simulador/CLI de decisiones recursivas en Texas Hold'em (heads-up y multi-opponent).
- Menús guiados (InquirerPy optional)
- Config guard/load (JSON)
- Player profiles (aggression, erraticity, risk_margin)
- Recursive reasoning (depth K) for hero decisions at river (or any street)
- Simplified betting model: discrete bet sizes, call, fold, check
- Logging and basic visualization with rich / matplotlib

Nota: Este motor es intencionalmente simple para ser extensible. Recomendado usar `eval7`
para equity Monte Carlo si lo tienes instalado. Si no, el script usará heurística.
"""
from __future__ import annotations
import random
import math
import json
import csv
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any

# Optional deps
try:
    from InquirerPy import inquirer
    HAVE_INQUIRER = True
except Exception:
    HAVE_INQUIRER = False

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

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    from rich import box
    console = Console()
    HAVE_RICH = True
except Exception:
    HAVE_RICH = False
    console = None

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

import numpy as np

# -------------------------
# Card / Deck Utilities
# -------------------------
RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = [r + s for r in RANKS for s in SUITS]

def make_deck():
    return DECK.copy()

def sample_deck_excluding(exclude: List[str]):
    pool = [c for c in DECK if c not in exclude]
    return pool

# -------------------------
# Simple equity estimator (eval7 if available)
# -------------------------
def estimate_equity_mc(hero: List[str], board: List[str], n_samples: int = 300) -> float:
    """
    Estimate hero's equity vs one random opponent using Monte Carlo.
    Uses eval7 if available, otherwise heuristic fallback.
    """
    if HAVE_EVAL7:
        hero_cards = [eval7.Card(c) for c in hero]
        board_cards = [eval7.Card(c) for c in board]
        deck = eval7.Deck()
        # remove known cards
        for c in hero_cards + board_cards:
            try:
                deck.cards.remove(c)
            except ValueError:
                pass
        wins = 0
        ties = 0
        n_samples = max(10, n_samples)
        for _ in range(n_samples):
            opp = deck.sample(2)
            # fill board
            deck.shuffle()
            fill = []
            while len(board_cards) + len(fill) < 5:
                card = deck.sample(1)[0]
                if card not in hero_cards and card not in board_cards and card not in opp:
                    fill.append(card)
            full_board = board_cards + fill
            hscore = eval7.evaluate(hero_cards, full_board)
            oscore = eval7.evaluate(opp, full_board)
            if hscore > oscore:
                wins += 1
            elif hscore == oscore:
                ties += 1
        return (wins + 0.5 * ties) / n_samples
    else:
        # fallback heuristic: check pairs / shared ranks / high cards
        score = 0.5
        # pair in hand
        if hero[0][0] == hero[1][0]:
            score += 0.12
        # match board ranks
        for b in board:
            if b[0] in (hero[0][0], hero[1][0]):
                score += 0.05
        return min(0.99, max(0.01, score))

# -------------------------
# Player profile and state
# -------------------------
@dataclass
class PlayerProfile:
    name: str = "P"
    stack: int = 1000
    seat: int = 0
    vpip: float = 0.2        # propensity to enter pots
    pfr: float = 0.1         # preflop raise freq
    aggression: float = 0.5  # 0..1
    erraticity: float = 0.0  # 0..1 (probability of random action)
    risk: float = 0.5        # 0..1 how risk-seeking
    model_weight: float = 0.0 # how much to rely on a learned model vs heuristic
    is_hero: bool = False

    def to_dict(self):
        return asdict(self)

@dataclass
class PlayerState:
    profile: PlayerProfile
    hole: List[str] = field(default_factory=list)
    committed: int = 0  # chips in pot for this round
    active: bool = True
    folded: bool = False

# -------------------------
# Opponent heuristic policy
# -------------------------
def heuristic_policy_probabilities(fv: Dict[str, float], profile: PlayerProfile) -> Dict[str, float]:
    """
    Return probabilities for actions ['fold','call','bet'] based on numeric features.
    fv can contain: pot, to_call, hero_equity_estimate, last_bet_size, effective_stack
    """
    eq = fv.get("equity", 0.5)
    pot = fv.get("pot", 100)
    to_call = fv.get("to_call", 0)
    eff = fv.get("eff_stack", 1000)
    # base tendencies from profile
    agg = profile.aggression
    risk = profile.risk
    err = profile.erraticity
    # simple logic: stronger equity -> more likely to call/raise
    win_prob = eq
    # pot odds influence
    pot_odds = 0.0
    if to_call > 0:
        pot_odds = to_call / (pot + to_call + 1e-9)
    # approximate expected value of calling: if win_prob > pot_odds -> call
    call_pref = max(0.01, min(0.99, 0.2 + (win_prob - pot_odds) * 2 + agg * 0.2))
    bet_pref = max(0.01, min(0.99, 0.05 + (win_prob - 0.5) * 2 + agg * 0.3 + risk * 0.2))
    fold_pref = max(0.01, 1.0 - call_pref - bet_pref)
    # normalize and include erraticity
    probs = {"fold": fold_pref, "call": call_pref, "bet": bet_pref}
    total = sum(probs.values())
    for k in probs:
        probs[k] = probs[k] / (total + 1e-9)
        # mix random behavior
        probs[k] = probs[k] * (1 - err) + (err / 3.0)
    # renormalize
    s = sum(probs.values())
    for k in probs:
        probs[k] /= s
    return probs

# -------------------------
# Torch networks (optional)
# -------------------------
if HAVE_TORCH:
    class SmallMLP(nn.Module):
        def __init__(self, inp: int, hidden: int = 128, out: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(inp, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out)
            )
        def forward(self, x):
            return self.net(x)

# -------------------------
# Game Engine (simplified)
# -------------------------
@dataclass
class GameConfig:
    small_blind: int = 5
    big_blind: int = 10
    ante: int = 0
    starting_stack: int = 1000
    bet_sizes: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])  # fractions of pot (bet sizing)
    max_rounds: int = 1000
    recursion_depth: int = 1
    mc_samples: int = 200

@dataclass
class GameResult:
    hero_profit_history: List[int]
    summary: Dict[str, Any]

class PokerGame:
    def __init__(self, profiles: List[PlayerProfile], config: GameConfig):
        self.config = config
        self.players: List[PlayerState] = [PlayerState(p) for p in profiles]
        self.dealer_idx = 0
        self.pot = 0
        self.board: List[str] = []
        self.deck: List[str] = []
        self.community_cards: List[str] = []
        self.bets_this_round = 0
        self.hand_history = []

    def rotate_dealer(self):
        self.dealer_idx = (self.dealer_idx + 1) % len(self.players)

    def reset_hand_state(self):
        for p in self.players:
            p.hole = []
            p.committed = 0
            p.active = True
            p.folded = False
        self.board = []
        self.pot = 0
        self.community_cards = []
        self.deck = make_deck()
        random.shuffle(self.deck)

    def post_blinds(self):
        # small / big blind to next players after dealer
        sb_idx = (self.dealer_idx + 1) % len(self.players)
        bb_idx = (self.dealer_idx + 2) % len(self.players)
        sb = min(self.players[sb_idx].profile.stack, self.config.small_blind)
        bb = min(self.players[bb_idx].profile.stack, self.config.big_blind)
        self.players[sb_idx].profile.stack -= sb
        self.players[bb_idx].profile.stack -= bb
        self.players[sb_idx].committed = sb
        self.players[bb_idx].committed = bb
        self.pot += sb + bb
        return sb_idx, bb_idx

    def deal_hole(self):
        for p in self.players:
            p.hole = [self.deck.pop(), self.deck.pop()]

    def deal_flop(self):
        # burn card
        self.deck.pop()
        self.community_cards.extend([self.deck.pop(), self.deck.pop(), self.deck.pop()])

    def deal_turn(self):
        self.deck.pop()
        self.community_cards.append(self.deck.pop())

    def deal_river(self):
        self.deck.pop()
        self.community_cards.append(self.deck.pop())

    def active_players(self):
        return [p for p in self.players if p.active and not p.folded and p.profile.stack > 0]

    # simplified betting: players choose fold/call/bet (bet is fixed % of pot sizes)
    def betting_round(self, to_call: int, street: str, hero_idx: int, opp_models: Optional[List]=None):
        """
        to_call: chips hero needs to put to call current bet (for action context)
        street: 'preflop'|'flop'|'turn'|'river'
        hero_idx: index of hero in self.players
        opp_models: optional list of functions that take features and return probs
        returns updated pot and per-player commits
        """
        # For each active player in seat order starting after dealer:
        order = list(range(self.dealer_idx + 1, self.dealer_idx + 1 + len(self.players)))
        order = [i % len(self.players) for i in order]
        for idx in order:
            pstate = self.players[idx]
            if pstate.folded or pstate.profile.stack <= 0:
                continue
            # build lightweight feature vector for decision
            fv = {
                "pot": self.pot,
                "to_call": to_call,
                "eff_stack": min(pstate.profile.stack, max(1, max(pl.profile.stack for pl in self.players))),
                "equity": estimate_equity_mc(pstate.hole if pstate.hole else ["As","Ks"], self.community_cards, n_samples=self.config.mc_samples // 4)
            }
            # decide action
            if pstate.profile.erraticity > 0 and random.random() < pstate.profile.erraticity:
                action = random.choice(["fold","call","bet"])
            else:
                if opp_models and self.players.index(pstate) < len(opp_models):
                    model = opp_models[self.players.index(pstate)]
                    try:
                        probs = model(fv) if callable(model) else heuristic_policy_probabilities(fv, pstate.profile)
                    except Exception:
                        probs = heuristic_policy_probabilities(fv, pstate.profile)
                else:
                    probs = heuristic_policy_probabilities(fv, pstate.profile)
                action = random.choices(population=list(probs.keys()), weights=list(probs.values()), k=1)[0]
            # enact action simplified
            if action == "fold":
                pstate.folded = True
                pstate.active = False
            elif action == "call":
                # commit to_call or all-in
                commit = min(pstate.profile.stack, to_call)
                pstate.profile.stack -= commit
                pstate.committed += commit
                self.pot += commit
            elif action == "bet":
                # bet fraction of pot choices
                bet_frac = random.choice(self.config.bet_sizes)
                bet_amount = int(max(1, self.pot * bet_frac))
                bet_amount = min(bet_amount, pstate.profile.stack)
                pstate.profile.stack -= bet_amount
                pstate.committed += bet_amount
                to_call = bet_amount  # next players must call this
                self.pot += bet_amount
        # return pot
        return self.pot

    def showdown(self):
        # find non-folded players, compare via eval7 if available else random
        active = [p for p in self.players if not p.folded]
        if len(active) == 1:
            winner = active[0]
            winner.profile.stack += self.pot
            return [(winner.profile.name, self.pot)]
        else:
            # naive: compute winner by sampling or eval7
            scores = []
            for p in active:
                if HAVE_EVAL7:
                    h = [eval7.Card(c) for c in p.hole]
                    b = [eval7.Card(c) for c in self.community_cards]
                    s = eval7.evaluate(h, b)
                else:
                    # heuristic ranking: sum of ranks indices
                    s = sum(RANKS.index(c[0]) for c in p.hole)
                scores.append((s, p))
            scores.sort(key=lambda x: x[0], reverse=True)
            top_score = scores[0][0]
            winners = [p for s, p in scores if s == top_score]
            split = self.pot // len(winners)
            for w in winners:
                w.profile.stack += split
            return [(w.profile.name, split) for w in winners]

    # One hand simulation, returns dict about hand
    def play_hand(self, hero_idx: int = 0, opp_models: Optional[List]=None, run_betting_full: bool = False):
        self.reset_hand_state()
        sb_idx, bb_idx = self.post_blinds()
        self.deal_hole()
        # preflop betting simplified
        self.betting_round(to_call=self.players[bb_idx].committed - self.players[sb_idx].committed, street="preflop", hero_idx=hero_idx, opp_models=opp_models)
        # flop
        self.deal_flop()
        self.betting_round(to_call=0, street="flop", hero_idx=hero_idx, opp_models=opp_models)
        # turn
        self.deal_turn()
        self.betting_round(to_call=0, street="turn", hero_idx=hero_idx, opp_models=opp_models)
        # river
        self.deal_river()
        self.betting_round(to_call=0, street="river", hero_idx=hero_idx, opp_models=opp_models)
        # showdown
        result = self.showdown()
        # record history
        self.hand_history.append({
            "board": self.community_cards.copy(),
            "holes": [p.hole.copy() for p in self.players],
            "pot": self.pot,
            "result": result
        })
        return result

# -------------------------
# Recursive decision routine for hero
# -------------------------
def hero_recursive_decision(hero_state: PlayerState, game: PokerGame, config: GameConfig,
                            opp_profiles: List[PlayerProfile], depth: int = 1, mc_samples: int = 200,
                            bet_sizes: Optional[List[float]] = None):
    """
    For hero at river (or any street), evaluate EV of candidate actions by sampling opponent hands and opponent actions.
    Returns best_action (str) and dict of EV per action.
    Candidate actions: 'fold', 'call', 'bet_X' (for each bet size)
    """
    if bet_sizes is None:
        bet_sizes = config.bet_sizes
    actions = ["fold", "call"] + [f"bet_{int(100*bf)}pct" for bf in bet_sizes]
    evs = {a: 0.0 for a in actions}
    hero = hero_state.hole
    board = game.community_cards
    pot = game.pot
    to_call = max(0, max(p.committed for p in game.players) - hero_state.committed)
    # sample opponent unknown hands from deck
    used = [c for p in game.players for c in p.hole] + board
    for _ in range(mc_samples):
        # sample a random opponent hand for each opponent (simplify: only sample one opponent for heads-up; for multi, sample each)
        sampled_hands = {}
        pool = [c for c in DECK if c not in used]
        random.shuffle(pool)
        # assign 2 cards per opponent (excluding hero)
        opp_index = 0
        for i, p in enumerate(game.players):
            if p is hero_state:  # skip hero
                continue
            sampled_hands[i] = [pool.pop(), pool.pop()]
        # For each action compute payoff vs opponent policies (approx)
        for action in actions:
            # for simplicity evaluate vs each opponent independently and average
            payoff_sum = 0.0
            for i, opp_profile in enumerate(opp_profiles):
                # simulate opponent response probabilities
                # compute fv for opponent
                fv = {"pot": pot, "to_call": to_call, "eff_stack": min(opp_profile.stack, hero_state.profile.stack),
                      "equity": estimate_equity_mc(sampled_hands[i], board, n_samples=20)}
                probs = heuristic_policy_probabilities(fv, opp_profile)
                # consider expected payoff = sum_p(prob_action_opp * payoff(action, opp_action))
                sub_ev = 0.0
                for opp_act, p_prob in probs.items():
                    payoff_val = payoff_model_simple(hero, sampled_hands[i], board, action, opp_act, pot, to_call, hero_state.profile, config)
                    sub_ev += p_prob * payoff_val
                payoff_sum += sub_ev
            payoff_avg = payoff_sum / max(1, len(opp_profiles))
            evs[action] += payoff_avg
    # average
    for a in evs:
        evs[a] /= mc_samples
    # pick best
    best_action = max(evs.items(), key=lambda kv: kv[1])[0]
    return best_action, evs

def payoff_model_simple(hero_cards, opp_cards, board, hero_action, opp_action, pot, to_call, hero_profile, config: GameConfig):
    """
    Very simplified payoff model:
    - if hero folds => -0 (we assume decision point before committing)
    - if opp folds => hero wins pot (+ any hero bet)
    - if showdown => use equity to compute expected value
    - hero_action strings: 'fold','call','bet_Xpct'
    """
    if hero_action == "fold":
        return -0.0
    # parse hero bet
    bet_amount = 0
    if hero_action.startswith("bet_"):
        pct = int(hero_action.split("_")[1].replace("pct","")) / 100.0
        bet_amount = int(max(1, pot * pct))
    if opp_action == "fold":
        return pot + bet_amount
    # showdown: use equity of hero vs opp
    eq = estimate_equity_mc(hero_cards, board, n_samples=60)
    hero_put = bet_amount if hero_action.startswith("bet_") else 0.0
    # EV approx:
    ev = eq * (pot + hero_put) - (1 - eq) * hero_put
    return ev

# -------------------------
# CLI helpers (menu / config)
# -------------------------
DEFAULT_CONFIG_PATH = Path("poker_sim_config.json")

def prompt_menu(title: str, choices: List[str]) -> str:
    if HAVE_INQUIRER:
        q = inquirer.select(message=title, choices=choices, default=choices[0]).execute()
        return q
    else:
        # fallback simple prompt
        print(title)
        for i, c in enumerate(choices):
            print(f"{i+1}) {c}")
        choice = input("Choice: ").strip()
        try:
            idx = int(choice) - 1
            return choices[idx]
        except Exception:
            return choices[0]

def prompt_number(prompt: str, default: int = 0):
    try:
        val = input(f"{prompt} [{default}]: ").strip()
        if val == "":
            return default
        return int(val)
    except Exception:
        return default

def prompt_float(prompt: str, default: float = 0.0):
    try:
        val = input(f"{prompt} [{default}]: ").strip()
        if val == "":
            return default
        return float(val)
    except Exception:
        return default

def save_config(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved config to {path}")

def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

# -------------------------
# Simulation orchestration
# -------------------------
def run_simulation_interactive():
    console_print("Welcome to Poker Recursive Simulator", accent=True)
    # quick config
    n_players = prompt_number("Número de jugadores (incluyendo hero)", default=2)
    profiles = []
    for i in range(n_players):
        name = input(f"Nombre jugador {i+1} [P{i+1}]: ").strip() or f"P{i+1}"
        stack = prompt_number(f"Stack inicial para {name}", default=1000)
        aggression = prompt_float(f"Aggression (0..1) para {name}", default=0.5)
        err = prompt_float(f"Erraticity (0..1) para {name}", default=0.0)
        risk = prompt_float(f"Risk (0..1) para {name}", default=0.5)
        is_hero = False
        if i == 0:
            # by default first is hero; let user choose
            use_hero = input("Hacer este jugador el HERO? (y/n) [y]: ").strip().lower() or "y"
            is_hero = use_hero.startswith("y")
        prof = PlayerProfile(name=name, stack=stack, seat=i, aggression=aggression, erraticity=err, risk=risk, is_hero=is_hero)
        profiles.append(prof)
    # game config
    sb = prompt_number("Small blind", default=5)
    bb = prompt_number("Big blind", default=10)
    starting_stack = prompt_number("Stack inicial general", default=1000)
    rounds = prompt_number("Número de manos a simular", default=100)
    depth = prompt_number("Profundidad de razonamiento (K)", default=1)
    mc = prompt_number("MC samples por evaluación (ajusta según velocidad)", default=200)
    # create game config and players
    gconf = GameConfig(small_blind=sb, big_blind=bb, starting_stack=starting_stack, max_rounds=rounds, recursion_depth=depth, mc_samples=mc)
    # ensure starting stack applied
    for p in profiles:
        p.stack = starting_stack
    # instantiate game
    game = PokerGame(profiles, gconf)
    # find hero index
    hero_idx = 0
    for i, p in enumerate(game.players):
        if p.profile.is_hero:
            hero_idx = i
            break
    # run rounds
    hero_profit = []
    for r in range(rounds):
        res = game.play_hand(hero_idx=hero_idx, opp_models=None)
        # compute hero profit for this hand
        hero_name = game.players[hero_idx].profile.name
        # find hero stack change from starting
        hero = game.players[hero_idx].profile
        # For simplicity track stack relative to starting_stack
        profit = hero.stack - starting_stack
        hero_profit.append(profit)
        if HAVE_RICH:
            console.print(f"Hand {r+1}/{rounds} done. Hero stack: {hero.stack} profit: {profit}")
        else:
            print(f"Hand {r+1}/{rounds} done. Hero stack: {hero.stack} profit: {profit}")
        game.rotate_dealer()
    # report
    sr = GameResult(hero_profit_history=hero_profit, summary={"final_stacks": {p.profile.name: p.profile.stack for p in game.players}})
    if HAVE_MPL:
        plt.plot(range(1, len(hero_profit)+1), hero_profit)
        plt.xlabel("Hand")
        plt.ylabel("Hero profit (relative)")
        plt.title("Hero profit over hands")
        plt.show()
    else:
        print("Final stacks:", sr.summary["final_stacks"])
    # save history optional
    out = input("Guardar historial? (filename or empty to skip): ").strip()
    if out:
        Path(out).write_text(json.dumps({"hands": game.hand_history, "summary": sr.summary}, default=str, indent=2), encoding="utf-8")
        print("Guardado.")
    return sr

def console_print(msg: str, accent: bool = False):
    if HAVE_RICH:
        if accent:
            console.rule(f"[bold green]{msg}")
        else:
            console.print(msg)
    else:
        print(msg)

# -------------------------
# Main entry
# -------------------------
def main():
    console_print("Poker Recursive Simulator", accent=True)
    while True:
        choice = prompt_menu("Selecciona acción", ["Run interactive sim", "Run quick sim (headsup)", "Exit"])
        if choice == "Exit":
            break
        if choice == "Run interactive sim":
            run_simulation_interactive()
        elif choice == "Run quick sim (headsup)":
            # quick heads-up example using predefined profiles
            p1 = PlayerProfile(name="Hero", stack=1000, seat=0, aggression=0.5, erraticity=0.0, risk=0.5, is_hero=True)
            p2 = PlayerProfile(name="Opp", stack=1000, seat=1, aggression=0.6, erraticity=0.1, risk=0.6)
            conf = GameConfig()
            game = PokerGame([p1,p2], conf)
            hero_idx = 0
            hero_profit = []
            for i in range(50):
                game.play_hand(hero_idx=hero_idx, opp_models=None)
                hero_profit.append(game.players[hero_idx].profile.stack - conf.starting_stack)
                game.rotate_dealer()
            console_print("Final stacks: " + str({p.profile.name: p.profile.stack for p in game.players}))
            if HAVE_MPL:
                plt.plot(hero_profit); plt.show()
        else:
            continue

if __name__ == "__main__":
    main()
