"""
decision_recursive_poker.py
Ejemplo demostrativo: placeholders y flujo completo.
Requisitos opcionales: pip install torch eval7
"""
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# try import eval7 for real hand equity; si no, usaremos fallback
try:
    import eval7
    HAVE_EVAL7 = True
except Exception:
    HAVE_EVAL7 = False

# -------------------------
# Helpers (cartas, encoding)
# -------------------------
RANKS = "23456789TJQKA"
SUITS = "cdhs"

def card_str_to_int(card: str) -> int:
    """Ej: 'As' -> index 51? We'll use eval7 style if available (rank+suit)."""
    # If eval7: use eval7.Card
    if HAVE_EVAL7:
        c = eval7.Card(card)
        return c
    # fallback: encode as integer 0..51 by rank*4 + suit_index
    r = RANKS.index(card[0].upper())
    s = SUITS.index(card[1].lower())
    return r * 4 + s

def one_hot_card_idx(idx: int) -> np.ndarray:
    out = np.zeros(52, dtype=np.float32)
    out[idx] = 1.0
    return out

def encode_cards(hole: list, board: list):
    """hole: ['As','Kh'], board: ['2d','5h',...]. Devuelve vector flat simples."""
    vec = []
    for i in range(2):
        if i < len(hole):
            idx = card_str_to_int(hole[i])
            if HAVE_EVAL7:
                # cannot one-hot eval7.Card, but return simple features; fallback to rank-suit
                r = RANKS.index(hole[i][0].upper())
                s = SUITS.index(hole[i][1].lower())
                v = np.zeros(52); v[r*4 + s] = 1.0
            else:
                v = one_hot_card_idx(idx)
        else:
            v = np.zeros(52, dtype=np.float32)
        vec.append(v)
    # board 5 slots
    for j in range(5):
        if j < len(board):
            idx = card_str_to_int(board[j])
            v = one_hot_card_idx(idx) if not HAVE_EVAL7 else (np.zeros(52); v := np.zeros(52); v[idx] = 1.0)
        else:
            v = np.zeros(52, dtype=np.float32)
        vec.append(v)
    return np.concatenate(vec)  # shape 7*52 = 364

# -------------------------
# Hand equity estimator
# -------------------------
def estimate_equity(hole, board, n_samples=200):
    """Estimación Monte Carlo de equity del hero vs un single random opponent hand.
       Si eval7 está disponible, se usa para evaluar; si no, devolvemos heurística aleatoria.
    """
    if HAVE_EVAL7:
        hero = [eval7.Card(c) for c in hole]
        board_cards = [eval7.Card(c) for c in board]
        deck = eval7.Deck()
        for c in hero + board_cards:
            deck.cards.remove(c)
        wins = 0
        ties = 0
        for _ in range(n_samples):
            opp = deck.sample(2)
            # completar board si necesario
            deck_cards = list(deck.cards)
            # shuffle copy deck for fill
            deck.shuffle()
            fill = []
            while len(board_cards) + len(fill) < 5:
                card = deck.sample(1)[0]
                if card not in hero and card not in board_cards and card not in opp:
                    fill.append(card)
            full_board = board_cards + fill
            hscore = eval7.evaluate(hero, full_board)
            oscore = eval7.evaluate(opp, full_board)
            if hscore > oscore:
                wins += 1
            elif hscore == oscore:
                ties += 1
        equity = (wins + ties * 0.5) / n_samples
        return equity
    else:
        # fallback: very rough heuristic (mejor reemplazar por eval7)
        # Si pareja en la mano o board => mayor equity, etc.
        text = " ".join(hole + board).lower()
        score = 0.5
        # boost for pocket pair
        if hole[0][0] == hole[1][0]:
            score += 0.12
        # boost if board pairs with our ranks
        for b in board:
            if b[0] in (hole[0][0], hole[1][0]):
                score += 0.05
        return max(0.01, min(0.99, score))

# -------------------------
# Small neural nets (policy + opponent model)
# -------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden=128, n_out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Feature builder
# -------------------------
def build_feature_vector(hole, board, pot, to_call, hero_stack, opp_stack,
                         position=0, last_action=0.0, vpip=0.2, pfr=0.1, agg=0.5, tilt=0.0):
    card_vec = encode_cards(hole, board)  # 364
    equity = estimate_equity(hole, board, n_samples=100)  # scalar
    eff_stack = min(hero_stack, opp_stack)
    pot_odds = to_call / (pot + to_call + 1e-9)
    # build numeric vector
    meta = np.array([pot, to_call, hero_stack, opp_stack, eff_stack, pot_odds, position, last_action,
                     vpip, pfr, agg, tilt, equity], dtype=np.float32)
    vec = np.concatenate([card_vec.astype(np.float32), meta])
    return vec  # shape ~364 + 13

# -------------------------
# Payoff simulator (one step)
# -------------------------
def payoff_terminal(hole, opp_hand_sample, board, hero_action, opp_action, pot, to_call, bet_amount):
    """Retorna payoff desde la perspectiva del hero (fichas ganadas, puede ser negativo).
       hero_action: 'check/call' or 'bet' (we assume bet amount = bet_amount) or 'fold' (represented separately)
       opp_action: 'fold', 'call', 'raise' (we simplify)
       Esta función asume un sóla ronda (river) y showdown si hay call.
    """
    # Simplified semantics:
    # If hero folds -> payoff = -0 (shouldn't happen on hero fold decision)
    # If hero checks and opp checks -> showdown
    # If hero bets and opp folds -> hero wins pot
    # If call and showdown -> compare hands
    # We'll use estimate via eval7 if available
    if hero_action == 'fold':
        return -0.0
    if opp_action == 'fold':
        # opponent folds, hero wins current pot (plus any bet hero put)
        # assume hero bet bet_amount if action=bet, otherwise 0
        hero_contrib = bet_amount if hero_action == 'bet' else 0.0
        return pot + hero_contrib
    # showdown: compare hero vs opp hand
    if HAVE_EVAL7:
        h = [eval7.Card(c) for c in hole]
        o = [eval7.Card(c) for c in opp_hand_sample]
        b = [eval7.Card(c) for c in board]
        hscore = eval7.evaluate(h, b)
        oscore = eval7.evaluate(o, b)
        if hscore > oscore:
            return pot + bet_amount
        elif hscore == oscore:
            return (pot + bet_amount) * 0.5
        else:
            return - (bet_amount if hero_action=='bet' else 0.0)
    else:
        # fallback: random tie-break using naive equity
        eq = estimate_equity(hole, board, n_samples=80)
        # approx: EV = eq*(pot+bet) - (1-eq)*(amount_hero_put)
        hero_put = bet_amount if hero_action == 'bet' else 0.0
        return eq * (pot + hero_put) - (1-eq) * hero_put

# -------------------------
# Recursive decision function
# -------------------------
ACTIONS = ['check_call', 'bet', 'fold']  # map to semantics as above

def recursive_decision(hole, board, state_meta, policy_net, opp_net,
                       depth=1, n_opponent_samples=200, bet_amount=100):
    """
    depth: cuántos niveles de 'razonamiento' (1 = usar OppModel para simular respuestas).
    strategy: calculamos EV(a) = E_{opp_hand, opp_action}[ payoff ]
    """
    # build base state vector
    fv = build_feature_vector(hole, board, state_meta['pot'], state_meta['to_call'],
                              state_meta['hero_stack'], state_meta['opp_stack'],
                              position=state_meta.get('position',0),
                              last_action=state_meta.get('last_action',0.0),
                              vpip=state_meta.get('vpip',0.2),
                              pfr=state_meta.get('pfr',0.1),
                              agg=state_meta.get('agg',0.5),
                              tilt=state_meta.get('tilt', 0.0))
    device = next(policy_net.parameters()).device
    x = torch.tensor(fv, dtype=torch.float32, device=device).unsqueeze(0)
    # We'll approximate EV by sampling opponent hands and opponent actions
    evs = {a: 0.0 for a in ACTIONS}
    # sample opponent hands uniformly (naive)
    for _ in range(n_opponent_samples):
        # sample random opponent hand disjoint from board/hole
        # (here we fallback to string placeholders)
        # In real code, sample from deck using eval7.Deck if available.
        opp_hand = sample_random_hand_excluding(hole+board)
        # build state for opponent model: we may include hero action as part of input
        for a in ACTIONS:
            # create a state variant including hero action (one-hot)
            # Here we cheat: we feed same fv plus encoded hero action to opp_net
            hero_action_idx = ACTIONS.index(a)
            x_opp = torch.cat([x, F.one_hot(torch.tensor([hero_action_idx], device=device), num_classes=len(ACTIONS)).float()], dim=1) \
                    if hasattr(opp_net, 'requires_action') else x
            # predict opponent action logits
            with torch.no_grad():
                logits = opp_net(x_opp) if hasattr(opp_net, 'requires_action') else opp_net(x)
                probs = F.softmax(logits.squeeze(0), dim=0).cpu().numpy()
            # map probs to actual opp actions: assume same ACTIONS list; for real model map appropriately
            # compute expected payoff: sum_over_opp_actions prob * payoff
            payoff_sum = 0.0
            for i, opp_act in enumerate(ACTIONS):
                p_oa = probs[i]
                payoff_val = payoff_terminal(hole, opp_hand, board, a, opp_act,
                                            pot=state_meta['pot'], to_call=state_meta['to_call'], bet_amount=bet_amount)
                payoff_sum += p_oa * payoff_val
            evs[a] += payoff_sum
    # average over samples
    for a in evs:
        evs[a] /= n_opponent_samples
    # if depth > 1 we could re-estimate opponent model conditioned on hero's best response etc.
    # For this demo we return argmax EV
    best_action = max(evs.items(), key=lambda kv: kv[1])[0]
    return best_action, evs

# -------------------------
# Utilities: sample random hand (naive)
# -------------------------
_all_cards = [r+s for r in RANKS for s in SUITS]
def sample_random_hand_excluding(used_cards):
    used = set([c.upper() for c in used_cards])
    pool = [c for c in _all_cards if c.upper() not in used]
    h = random.sample(pool, 2)
    return h

# -------------------------
# Demo usage (placeholder)
# -------------------------
if __name__ == "__main__":
    # instantiate networks with correct input dim estimate
    dummy_hole = ['As','Kh']
    dummy_board = ['2d','5h','9c','Td','Jh']
    fv = build_feature_vector(dummy_hole, dummy_board, pot=300, to_call=50, hero_stack=1200, opp_stack=800)
    input_dim = fv.shape[0]
    policy = SimpleMLP(input_dim, hidden=128, n_out=len(ACTIONS))
    opp = SimpleMLP(input_dim, hidden=128, n_out=len(ACTIONS))  # opponent model
    # (opp could be trained to predict actual opponent actions)
    state_meta = {'pot':300, 'to_call':50, 'hero_stack':1200, 'opp_stack':800, 'position':1, 'vpip':0.25, 'agg':0.6}
    act, evs = recursive_decision(dummy_hole, dummy_board, state_meta, policy, opp, depth=1, n_opponent_samples=300, bet_amount=100)
    print("Decisión:", act)
    print("EVs:", evs)
