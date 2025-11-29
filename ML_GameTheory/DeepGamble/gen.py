#!/usr/bin/env python3
"""
create_poker_project.py

Genera la estructura del proyecto `poker_engine_v2/` con archivos esqueleto:
poker_engine_v2/
├── __init__.py
├── actions.py          # Action, ActionType, ActionValidator
├── betting.py          # BettingRound, SidePotManager
├── hand.py             # PokerHand
├── engine.py           # PokerEngine (principal)
├── history.py          # HandHistory
└── tests/
    ├── test_actions.py
    ├── test_betting.py
    ├── test_side_pots.py
    ├── test_hand.py
    └── test_integration.py

Uso:
    python create_poker_project.py
"""
from pathlib import Path
import textwrap
import os
import sys

ROOT = Path("poker_engine_v2")

FILES = {
    "__init__.py": '''"""
poker_engine_v2 package
"""
__version__ = "0.1.0"
''',

    "actions.py": '''"""
actions.py
Define ActionType, Action, ActionValidator.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"

@dataclass
class Action:
    actor_id: int
    type: ActionType
    amount: Optional[int] = None  # for bet/raise

class ActionValidationError(Exception):
    pass

class ActionValidator:
    \"\"\"Validate that an action is legal in the current betting context.
    This is a placeholder: integrate with BettingRound for full checks.\"\"\"
    @staticmethod
    def validate(action: Action, to_call: int, player_stack: int) -> bool:
        if action.type in (ActionType.BET, ActionType.RAISE):
            if action.amount is None or action.amount <= 0:
                raise ActionValidationError("Bet/raise requires positive amount")
            if action.amount > player_stack:
                raise ActionValidationError("Bet/raise exceeds player stack")
        if action.type == ActionType.CALL:
            if to_call > player_stack:
                # allow all-in call
                return True
        return True
''',

    "betting.py": '''"""
betting.py
Contains BettingRound (simple discrete model) and SidePotManager skeleton.
"""
from typing import List, Dict, Optional
from collections import defaultdict
from .actions import Action, ActionType

class SidePotManager:
    \"\"\"Simplified manager for side pots. Tracks contributions and computes shares at showdown.\"\"\"
    def __init__(self):
        self.contributions = defaultdict(int)  # player_id -> contributed amount

    def add_contribution(self, player_id: int, amount: int):
        self.contributions[player_id] += amount

    def compute_pots(self) -> List[Dict]:
        # Returns list of pots: [{\"amount\": int, \"eligible\": [player_ids]}]
        pots = []
        if not self.contributions:
            return pots
        # naive algorithm: build side pots sorted by contributions
        sorted_players = sorted(self.contributions.items(), key=lambda kv: kv[1])
        remaining_players = set(self.contributions.keys())
        prev = 0
        for pid, contrib in sorted_players:
            layer = contrib - prev
            if layer > 0:
                pot_amount = layer * len(remaining_players)
                pots.append({\"amount\": pot_amount, \"eligible\": sorted(remaining_players)})
                prev = contrib
            remaining_players.remove(pid)
        return pots

class BettingRound:
    \"\"\"Simple betting round stateful object (not full poker rules).\"\"\"
    def __init__(self, player_order: List[int], stacks: Dict[int,int], pot: int = 0):
        self.player_order = player_order[:]  # rotation order for actions
        self.stacks = dict(stacks)
        self.to_call = 0
        self.pot = pot
        self.committed = {pid: 0 for pid in player_order}
        self.actions = []

    def apply_action(self, action: Action):
        if action.type == ActionType.FOLD:
            self.actions.append(action)
            # mark as folded externally by engine
        elif action.type == ActionType.CALL:
            commit = min(self.to_call - self.committed[action.actor_id], self.stacks[action.actor_id])
            self.stacks[action.actor_id] -= commit
            self.committed[action.actor_id] += commit
            self.pot += commit
            self.actions.append(action)
        elif action.type in (ActionType.BET, ActionType.RAISE):
            amt = action.amount or 0
            # Simplified: set to_call to amt
            self.to_call = max(self.to_call, amt)
            commit = min(amt - self.committed[action.actor_id], self.stacks[action.actor_id])
            self.stacks[action.actor_id] -= commit
            self.committed[action.actor_id] += commit
            self.pot += commit
            self.actions.append(action)
        elif action.type == ActionType.CHECK:
            self.actions.append(action)
        else:
            raise ValueError(\"Unknown action type\")
''',

    "hand.py": '''"""
hand.py
Representation of a poker hand: hole cards, board, simple evaluation placeholder.
"""
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PokerHand:
    player_holes: Dict[int, List[str]]  # player_id -> list of 2 cards (e.g. 'As')
    board: List[str]  # up to 5 community cards

    def is_showdown(self) -> bool:
        # placeholder: if no one left folded (higher-level engine should decide)
        return len(self.board) == 5

    def evaluate(self) -> Dict[int, float]:
        \"\"\"Return a placeholder evaluation score per player (higher means stronger).\"\"\"
        scores = {pid: (len(''.join(hole)) % 100) for pid, hole in self.player_holes.items()}
        return scores
''',

    "engine.py": '''"""
engine.py
Top-level PokerEngine skeleton. Coordinates dealing, betting rounds and showdowns.
"""
from typing import List, Dict, Any
from .hand import PokerHand
from .betting import BettingRound, SidePotManager
from .actions import Action, ActionType

class PokerEngine:
    def __init__(self, players: List[int], starting_stacks: Dict[int,int]):
        self.players = players[:]  # list of player ids
        self.stacks = dict(starting_stacks)
        self.pot = 0
        self.history = []

    def new_hand(self, hole_cards: Dict[int, List[str]], board: List[str]):
        hand = PokerHand(player_holes=hole_cards, board=board)
        return hand

    def run_betting_round(self, betting_round: BettingRound, actions: List[Action]):
        for act in actions:
            betting_round.apply_action(act)

    def resolve_showdown(self, hand: PokerHand) -> Dict[int,int]:
        scores = hand.evaluate()
        # naive winner: max score
        winner = max(scores.items(), key=lambda kv: kv[1])[0]
        return {winner: sum(betting_round.committed.get(winner, 0) for betting_round in [])}
''',

    "history.py": '''"""
history.py
HandHistory class that stores hand-level records for analysis.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class HandHistory:
    records: List[Dict[str, Any]] = field(default_factory=list)

    def append(self, record: Dict[str, Any]):
        self.records.append(record)

    def to_json(self) -> str:
        import json
        return json.dumps(self.records, indent=2)
''',

    "tests/test_actions.py": '''import pytest
from poker_engine_v2.actions import Action, ActionType, ActionValidator, ActionValidationError

def test_action_creation():
    a = Action(actor_id=1, type=ActionType.BET, amount=50)
    assert a.actor_id == 1
    assert a.type == ActionType.BET
    assert a.amount == 50

def test_validation_fail_amount():
    a = Action(actor_id=1, type=ActionType.BET, amount=0)
    with pytest.raises(ActionValidationError):
        ActionValidator.validate(a, to_call=0, player_stack=100)
''',

    "tests/test_betting.py": '''import pytest
from poker_engine_v2.betting import BettingRound
from poker_engine_v2.actions import Action, ActionType

def test_betting_round_apply_call():
    players = [1,2,3]
    stacks = {1:100,2:100,3:100}
    br = BettingRound(players, stacks, pot=0)
    a = Action(actor_id=1, type=ActionType.CALL)
    br.to_call = 10
    br.apply_action(a)
    assert br.committed[1] >= 0
''',

    "tests/test_side_pots.py": '''from poker_engine_v2.betting import SidePotManager

def test_side_pots_basic():
    sp = SidePotManager()
    sp.add_contribution(1, 50)
    sp.add_contribution(2, 100)
    sp.add_contribution(3, 200)
    pots = sp.compute_pots()
    assert isinstance(pots, list)
    assert sum(p['amount'] for p in pots) > 0
''',

    "tests/test_hand.py": '''from poker_engine_v2.hand import PokerHand

def test_hand_evaluate():
    hand = PokerHand({1: ['As','Kh'], 2: ['2c','3d']}, board=['4s','5h','6d','7c','8s'])
    scores = hand.evaluate()
    assert 1 in scores and 2 in scores
''',

    "tests/test_integration.py": '''from poker_engine_v2.engine import PokerEngine
from poker_engine_v2.actions import Action, ActionType
from poker_engine_v2.betting import BettingRound

def test_simple_integration():
    players = [1,2]
    stacks = {1:1000,2:1000}
    eng = PokerEngine(players, stacks)
    hand = eng.new_hand({1:['As','Ks'], 2:['2c','3d']}, board=['4s','5h','6d','7c','8s'])
    br = BettingRound(players, stacks)
    eng.run_betting_round(br, [Action(1, ActionType.BET, amount=50)])
    assert br.pot >= 0
''',
}

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    print(f"Written: {path}")

def main():
    print("Creating project structure at:", ROOT.resolve())
    ensure_dir(ROOT)
    # write package files
    for fname, content in FILES.items():
        full = ROOT / fname
        write_file(full, content)
    # create __init__.py in tests package to treat as module (optional)
    test_init = ROOT / "tests" / "__init__.py"
    write_file(test_init, '"""\nTest package init\n"""')
    print("Project created. To run tests (pytest recommended):")
    print("  pip install pytest")
    print("  pytest -q poker_engine_v2/tests")

if __name__ == "__main__":
    main()
