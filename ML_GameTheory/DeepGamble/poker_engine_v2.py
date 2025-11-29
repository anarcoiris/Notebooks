"""
poker_engine_v2.py

Motor de Póker Corregido - FASE 1 COMPLETA

Características:
- Sistema de apuestas correcto con turnos
- Fases del juego completas (preflop, flop, turn, river)
- Action logging detallado
- Side pots para all-ins
- Validación de acciones legales
- Hand history en formato legible

Uso:
    from poker_engine_v2 import PokerEngine, PlayerProfile, SimConfig
    
    profiles = [PlayerProfile(name='Hero'), PlayerProfile(name='Villain')]
    config = SimConfig()
    engine = PokerEngine(profiles, config)
    result = engine.play_hand()
"""

from __future__ import annotations
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import defaultdict

# Importar utilities del código original
RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = [r + s for r in RANKS for s in SUITS]

def make_deck() -> List[str]:
    """Crea un deck barajado"""
    d = DECK.copy()
    random.shuffle(d)
    return d


# ==================== ACTION SYSTEM ====================

class ActionType(Enum):
    """Tipos de acciones válidas en póker"""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass
class Action:
    """Representa una acción de póker con contexto completo"""
    type: ActionType
    amount: int  # 0 para fold/check, monto para call/bet/raise/all_in
    player_name: str
    phase: str
    to_call: int = 0  # cuánto tenía que igualar antes de actuar
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self):
        if self.type == ActionType.FOLD:
            return f"{self.player_name} folds"
        elif self.type == ActionType.CHECK:
            return f"{self.player_name} checks"
        elif self.type == ActionType.CALL:
            return f"{self.player_name} calls {self.amount}"
        elif self.type == ActionType.BET:
            return f"{self.player_name} bets {self.amount}"
        elif self.type == ActionType.RAISE:
            return f"{self.player_name} raises to {self.amount}"
        elif self.type == ActionType.ALL_IN:
            return f"{self.player_name} is all-in {self.amount}"
        return f"{self.player_name} {self.type.value} {self.amount}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'amount': self.amount,
            'player': self.player_name,
            'phase': self.phase,
            'to_call': self.to_call,
            'timestamp': self.timestamp
        }


class ActionValidator:
    """Valida acciones y genera acciones legales"""
    
    @staticmethod
    def get_legal_actions(
        player_stack: int,
        player_committed: int,
        current_bet: int,
        min_raise: int,
        can_check: bool
    ) -> List[ActionType]:
        """
        Retorna lista de acciones legales para un jugador.
        
        Args:
            player_stack: fichas restantes del jugador
            player_committed: fichas ya comprometidas en esta ronda
            current_bet: apuesta actual que debe igualar
            min_raise: mínimo incremento para raise
            can_check: si puede check (no hay apuesta que igualar)
        """
        legal = [ActionType.FOLD]  # siempre puede foldear
        
        to_call = current_bet - player_committed
        
        if can_check:
            legal.append(ActionType.CHECK)
        
        if to_call > 0:
            # Debe igualar una apuesta
            if player_stack >= to_call:
                legal.append(ActionType.CALL)
            
            # Puede raise si tiene suficiente
            min_raise_total = current_bet + min_raise
            if player_stack > to_call + min_raise:
                legal.append(ActionType.RAISE)
        else:
            # No hay apuesta, puede bet
            if player_stack > 0:
                legal.append(ActionType.BET)
        
        # All-in siempre es válido si tiene stack
        if player_stack > 0:
            legal.append(ActionType.ALL_IN)
        
        return legal
    
    @staticmethod
    def validate_action(
        action: Action,
        player_stack: int,
        player_committed: int,
        current_bet: int,
        min_raise: int
    ) -> Tuple[bool, str]:
        """
        Valida si una acción es legal.
        
        Returns:
            (is_valid, reason)
        """
        to_call = current_bet - player_committed
        
        if action.type == ActionType.FOLD:
            return True, "OK"
        
        if action.type == ActionType.CHECK:
            if to_call > 0:
                return False, "Cannot check, must call or raise"
            return True, "OK"
        
        if action.type == ActionType.CALL:
            if to_call == 0:
                return False, "Nothing to call"
            if action.amount != to_call:
                return False, f"Call amount must be {to_call}"
            if player_stack < to_call:
                return False, "Not enough stack to call"
            return True, "OK"
        
        if action.type == ActionType.BET:
            if current_bet > 0:
                return False, "Cannot bet, there's already a bet (use raise)"
            if action.amount > player_stack:
                return False, "Bet exceeds stack"
            return True, "OK"
        
        if action.type == ActionType.RAISE:
            if to_call == 0:
                return False, "Cannot raise, no bet to raise (use bet)"
            min_raise_total = current_bet + min_raise
            if action.amount < min_raise_total:
                return False, f"Raise must be at least {min_raise_total}"
            if action.amount > current_bet + player_stack:
                return False, "Raise exceeds stack"
            return True, "OK"
        
        if action.type == ActionType.ALL_IN:
            if action.amount != player_stack:
                return False, f"All-in amount must equal stack ({player_stack})"
            return True, "OK"
        
        return False, "Unknown action type"


# ==================== PLAYER STATE ====================

@dataclass
class PlayerProfile:
    """Perfil de jugador (compatible con código original)"""
    name: str = "Player"
    stack: int = 1000
    aggression: float = 0.5
    erraticity: float = 0.0
    risk: float = 0.5
    vpip: float = 0.2
    pfr: float = 0.1
    is_hero: bool = False


class PlayerState:
    """Estado de jugador durante una mano"""
    
    def __init__(self, profile: PlayerProfile, seat: int):
        self.profile = profile
        self.seat = seat
        self.hole: List[str] = []
        self.committed_this_round = 0  # comprometido en la ronda actual
        self.committed_total = 0  # total comprometido en toda la mano
        self.folded = False
        self.all_in = False
        self.acted_this_round = False
    
    def reset_for_betting_round(self):
        """Reset para nueva ronda de apuestas"""
        self.committed_this_round = 0
        self.acted_this_round = False
    
    def commit(self, amount: int):
        """Commitea fichas del stack"""
        actual = min(amount, self.profile.stack)
        self.profile.stack -= actual
        self.committed_this_round += actual
        self.committed_total += actual
        if self.profile.stack == 0:
            self.all_in = True
        return actual
    
    def is_active(self) -> bool:
        """Puede actuar en esta mano"""
        return not self.folded and not self.all_in
    
    def __repr__(self):
        return f"PlayerState({self.profile.name}, stack={self.profile.stack}, seat={self.seat})"


# ==================== SIDE POT MANAGER ====================

class SidePotManager:
    """Calcula side pots cuando hay all-ins"""
    
    @staticmethod
    def calculate_pots(players: List[PlayerState]) -> List[Dict[str, Any]]:
        """
        Calcula main pot y side pots.
        
        Returns:
            Lista de pots: [
                {'amount': 300, 'eligible': ['P1', 'P2', 'P3']},
                {'amount': 200, 'eligible': ['P2', 'P3']},
            ]
        """
        # Solo consideramos jugadores que no foldearon
        active = [p for p in players if not p.folded]
        
        if not active:
            return []
        
        # Casos simples
        if len(active) == 1:
            # Solo uno activo, se lleva todo
            total = sum(p.committed_total for p in players)
            return [{'amount': total, 'eligible': [active[0].profile.name]}]
        
        # Agrupar por monto comprometido
        levels = {}
        for p in active:
            amt = p.committed_total
            if amt not in levels:
                levels[amt] = []
            levels[amt].append(p)
        
        # Ordenar niveles
        sorted_levels = sorted(levels.keys())
        
        pots = []
        prev_level = 0
        eligible_players = active.copy()
        
        for level in sorted_levels:
            if level == 0:
                continue
            
            # Calcular cuánto aporta cada jugador elegible a este pot
            contribution_per_player = level - prev_level
            num_eligible = len(eligible_players)
            pot_amount = contribution_per_player * num_eligible
            
            # Añadir contribuciones de jugadores foldeados hasta este nivel
            for p in players:
                if p.folded and p.committed_total >= prev_level:
                    pot_amount += min(contribution_per_player, p.committed_total - prev_level)
            
            if pot_amount > 0:
                pots.append({
                    'amount': pot_amount,
                    'eligible': [p.profile.name for p in eligible_players]
                })
            
            # Remover jugadores que están all-in en este nivel
            eligible_players = [p for p in eligible_players if p.committed_total > level]
            prev_level = level
        
        return pots


# ==================== BETTING ROUND ====================

class BettingRound:
    """Gestiona una ronda completa de apuestas"""
    
    def __init__(
        self,
        players: List[PlayerState],
        bb_amount: int,
        start_idx: int,
        is_preflop: bool = False
    ):
        self.players = players
        self.bb_amount = bb_amount
        self.current_bet = 0
        self.min_raise = bb_amount
        self.last_raiser_idx: Optional[int] = None
        self.active_idx = start_idx
        self.is_preflop = is_preflop
        self.actions: List[Action] = []
        
        # Reset committed amounts para nueva ronda
        for p in players:
            p.reset_for_betting_round()
    
    def run(self, decision_maker: Callable) -> bool:
        """
        Ejecuta la ronda de apuestas completa.
        
        Args:
            decision_maker: función que decide acción dado (player, game_state)
        
        Returns:
            True si la mano continúa, False si termina (todos fold menos uno)
        """
        action_count = 0
        max_actions = len(self.players) * 100  # safety limit
        
        while not self._is_betting_complete() and action_count < max_actions:
            player = self.players[self.active_idx]
            
            if player.is_active():
                # Obtener acción del decision maker
                action = decision_maker(self._get_game_state(), player)
                
                # Procesar acción
                self._process_action(player, action)
                self.actions.append(action)
                player.acted_this_round = True
            
            # Siguiente jugador
            self.active_idx = (self.active_idx + 1) % len(self.players)
            action_count += 1
        
        # Retorna True si hay más de un jugador activo
        active_count = sum(1 for p in self.players if not p.folded)
        return active_count > 1
    
    def _is_betting_complete(self) -> bool:
        """Determina si la ronda de apuestas ha terminado"""
        active = [p for p in self.players if p.is_active()]
        
        if len(active) == 0:
            # Todos all-in o fold
            return True
        
        if len(active) == 1:
            # Solo queda uno que puede actuar
            # Debe igualar el current_bet o ya lo hizo
            player = active[0]
            if player.committed_this_round >= self.current_bet:
                return True
            if not player.acted_this_round:
                return False
            return True
        
        # Múltiples jugadores activos
        # Todos deben haber actuado y estar igualados
        for p in active:
            if not p.acted_this_round:
                return False
            if p.committed_this_round < self.current_bet:
                return False
        
        return True
    
    def _process_action(self, player: PlayerState, action: Action):
        """Procesa una acción y actualiza el estado"""
        if action.type == ActionType.FOLD:
            player.folded = True
        
        elif action.type == ActionType.CHECK:
            # No hace nada
            pass
        
        elif action.type == ActionType.CALL:
            to_call = self.current_bet - player.committed_this_round
            player.commit(to_call)
        
        elif action.type == ActionType.BET:
            player.commit(action.amount)
            self.current_bet = action.amount
            self.min_raise = action.amount
            self.last_raiser_idx = self.active_idx
            self._reset_other_players_acted(self.active_idx)
        
        elif action.type == ActionType.RAISE:
            # El monto es el total nuevo
            to_call = self.current_bet - player.committed_this_round
            player.commit(to_call + (action.amount - self.current_bet))
            raise_by = action.amount - self.current_bet
            self.min_raise = raise_by
            self.current_bet = action.amount
            self.last_raiser_idx = self.active_idx
            self._reset_other_players_acted(self.active_idx)
        
        elif action.type == ActionType.ALL_IN:
            amount = player.commit(action.amount)
            
            # Determinar si el all-in es un raise válido
            total_committed = player.committed_this_round
            if total_committed > self.current_bet:
                if total_committed >= self.current_bet + self.min_raise:
                    # Es un raise válido
                    self.min_raise = total_committed - self.current_bet
                    self.current_bet = total_committed
                    self.last_raiser_idx = self.active_idx
                    self._reset_other_players_acted(self.active_idx)
                else:
                    # All-in por menos del min_raise, no reabre acción
                    self.current_bet = max(self.current_bet, total_committed)
            else:
                # All-in por menos que current bet (call all-in)
                pass
    
    def _reset_other_players_acted(self, raiser_idx: int):
        """Reset acted flag para jugadores que deben responder al raise"""
        for i, p in enumerate(self.players):
            if i != raiser_idx and p.is_active():
                p.acted_this_round = False
    
    def _get_game_state(self) -> Dict[str, Any]:
        """Construye estado del juego para decision maker"""
        return {
            'current_bet': self.current_bet,
            'min_raise': self.min_raise,
            'bb_amount': self.bb_amount,
            'is_preflop': self.is_preflop,
            'actions': self.actions.copy()
        }
    
    def get_total_pot(self) -> int:
        """Calcula el pot total de esta ronda"""
        return sum(p.committed_this_round for p in self.players)


# ==================== POKER HAND ====================

@dataclass
class SimConfig:
    """Configuración de simulación (compatible con código original)"""
    small_blind: int = 5
    big_blind: int = 10
    starting_stack: int = 1000
    rounds: int = 200
    recursion_depth: int = 1
    mc_samples: int = 150
    bet_sizes: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])


class PokerHand:
    """Representa y ejecuta una mano completa de póker"""
    
    def __init__(
        self,
        players: List[PlayerProfile],
        config: SimConfig,
        button_idx: int = 0
    ):
        self.config = config
        self.button_idx = button_idx
        self.players = [PlayerState(p, i) for i, p in enumerate(players)]
        self.deck: List[str] = []
        self.board: List[str] = []
        self.phase = 'preflop'
        self.hand_id = str(uuid.uuid4())[:8]
        self.history = HandHistory(self.hand_id)
    
    def play(self, decision_maker: Callable) -> Dict[str, Any]:
        """
        Ejecuta la mano completa.
        
        Args:
            decision_maker: función(game_state, player) -> Action
        
        Returns:
            Resultado de la mano con ganadores y montos
        """
        # Setup inicial
        self._reset()
        self._post_blinds()
        self._deal_hole_cards()
        
        # Guardar info inicial
        self.history.set_players([
            {'name': p.profile.name, 'stack': p.profile.stack, 'seat': p.seat}
            for p in self.players
        ])
        self.history.button_idx = self.button_idx
        
        # PREFLOP
        self.phase = 'preflop'
        if not self._run_betting_phase(decision_maker, is_preflop=True):
            return self._award_pot()
        
        # FLOP
        self._deal_flop()
        self.phase = 'flop'
        if not self._run_betting_phase(decision_maker):
            return self._award_pot()
        
        # TURN
        self._deal_turn()
        self.phase = 'turn'
        if not self._run_betting_phase(decision_maker):
            return self._award_pot()
        
        # RIVER
        self._deal_river()
        self.phase = 'river'
        if not self._run_betting_phase(decision_maker):
            return self._award_pot()
        
        # SHOWDOWN
        return self._showdown()
    
    def _reset(self):
        """Inicializa la mano"""
        self.deck = make_deck()
        self.board = []
        for p in self.players:
            p.hole = []
            p.committed_this_round = 0
            p.committed_total = 0
            p.folded = False
            p.all_in = False
            p.acted_this_round = False
    
    def _post_blinds(self):
        """Postea small blind y big blind"""
        n = len(self.players)
        if n < 2:
            return
        
        # Heads-up: button es SB, otro es BB
        # Multi-way: button+1 es SB, button+2 es BB
        if n == 2:
            sb_idx = self.button_idx
            bb_idx = (self.button_idx + 1) % n
        else:
            sb_idx = (self.button_idx + 1) % n
            bb_idx = (self.button_idx + 2) % n
        
        sb_player = self.players[sb_idx]
        bb_player = self.players[bb_idx]
        
        sb_amount = sb_player.commit(self.config.small_blind)
        bb_amount = bb_player.commit(self.config.big_blind)
        
        # Log blinds
        self.history.add_blind(sb_player.profile.name, sb_amount, 'SB')
        self.history.add_blind(bb_player.profile.name, bb_amount, 'BB')
    
    def _deal_hole_cards(self):
        """Reparte cartas hole"""
        for p in self.players:
            p.hole = [self.deck.pop(), self.deck.pop()]
        
        # Log holes (solo para historia)
        for p in self.players:
            self.history.add_hole_cards(p.profile.name, p.hole.copy())
    
    def _deal_flop(self):
        """Reparte el flop"""
        self.deck.pop()  # burn
        self.board = [self.deck.pop(), self.deck.pop(), self.deck.pop()]
        self.history.set_board(self.board.copy())
    
    def _deal_turn(self):
        """Reparte el turn"""
        self.deck.pop()  # burn
        self.board.append(self.deck.pop())
        self.history.set_board(self.board.copy())
    
    def _deal_river(self):
        """Reparte el river"""
        self.deck.pop()  # burn
        self.board.append(self.deck.pop())
        self.history.set_board(self.board.copy())
    
    def _run_betting_phase(
        self,
        decision_maker: Callable,
        is_preflop: bool = False
    ) -> bool:
        """
        Ejecuta una fase de apuestas.
        
        Returns:
            True si continúa, False si termina la mano
        """
        # Determinar quién actúa primero
        n = len(self.players)
        if is_preflop:
            # Preflop: actúa primero el jugador después del BB
            if n == 2:
                start_idx = self.button_idx  # HU: button actúa primero preflop
            else:
                start_idx = (self.button_idx + 3) % n  # UTG
        else:
            # Postflop: actúa primero SB (o siguiente vivo)
            if n == 2:
                start_idx = (self.button_idx + 1) % n
            else:
                start_idx = (self.button_idx + 1) % n
        
        # Encontrar primer jugador activo
        for i in range(n):
            idx = (start_idx + i) % n
            if self.players[idx].is_active():
                start_idx = idx
                break
        
        # Crear y ejecutar ronda de apuestas
        betting = BettingRound(
            self.players,
            self.config.big_blind,
            start_idx,
            is_preflop
        )
        
        # Current bet inicial en preflop es el BB
        if is_preflop:
            betting.current_bet = self.config.big_blind
        
        continues = betting.run(decision_maker)
        
        # Guardar acciones en history
        for action in betting.actions:
            self.history.add_action(self.phase, action)
        
        return continues
    
    def _award_pot(self) -> Dict[str, Any]:
        """
        Otorga el pot cuando todos fold menos uno.
        
        Returns:
            Resultado de la mano
        """
        total_pot = sum(p.committed_total for p in self.players)
        active = [p for p in self.players if not p.folded]
        
        if not active:
            # Todos foldearon (no debería pasar)
            return {'pot': total_pot, 'winners': [], 'result': 'all_folded'}
        
        winner = active[0]
        winner.profile.stack += total_pot
        
        result = {
            'pot': total_pot,
            'winners': [(winner.profile.name, total_pot)],
            'result': 'fold_win',
            'phase': self.phase
        }
        
        self.history.set_result(result)
        return result
    
    def _showdown(self) -> Dict[str, Any]:
        """
        Ejecuta el showdown y otorga pots.
        
        Returns:
            Resultado de la mano
        """
        # Calcular side pots
        pots = SidePotManager.calculate_pots(self.players)
        
        total_pot = sum(p['amount'] for p in pots)
        all_winners = []
        
        # Evaluar cada pot
        for pot_info in pots:
            eligible = [p for p in self.players if p.profile.name in pot_info['eligible']]
            winners = self._determine_winners(eligible)
            
            # Dividir pot entre ganadores
            split = pot_info['amount'] // len(winners)
            for w in winners:
                w.profile.stack += split
                all_winners.append((w.profile.name, split))
        
        result = {
            'pot': total_pot,
            'winners': all_winners,
            'result': 'showdown',
            'pots': pots,
            'phase': 'showdown'
        }
        
        self.history.set_result(result)
        return result
    
    def _determine_winners(self, eligible: List[PlayerState]) -> List[PlayerState]:
        """
        Determina ganadores entre jugadores elegibles.
        
        Usa eval7 si está disponible, sino ranking simple.
        """
        try:
            import eval7
            from poker_gui import eval7_evaluate
            
            scores = []
            for p in eligible:
                hand = [eval7.Card(c) for c in p.hole]
                board = [eval7.Card(c) for c in self.board]
                score = eval7_evaluate(hand, board)
                scores.append((score, p))
            
            scores.sort(key=lambda x: x[0], reverse=True)
            best_score = scores[0][0]
            return [p for s, p in scores if s == best_score]
        
        except Exception:
            # Fallback: ranking simple por valor de cartas
            scores = []
            for p in eligible:
                score = sum(RANKS.index(c[0]) for c in p.hole)
                # Bonus por pareja
                if p.hole[0][0] == p.hole[1][0]:
                    score += 20
                scores.append((score, p))
            
            scores.sort(key=lambda x: x[0], reverse=True)
            best_score = scores[0][0]
            return [p for s, p in scores if s == best_score]


# ==================== HAND HISTORY ====================

class HandHistory:
    """Logging estructurado de manos"""
    
    def __init__(self, hand_id: str):
        self.hand_id = hand_id
        self.timestamp = time.time()
        self.players: List[Dict] = []
        self.button_idx = 0
        self.phases = {
            'preflop': [],
            'flop': [],
            'turn': [],
            'river': []
        }
        self.board: List[str] = []
        self.hole_cards: Dict[str, List[str]] = {}
        self.result: Optional[Dict] = None
        self.blinds: List[Dict] = []
    
    def set_players(self, players: List[Dict]):
        self.players = players
    
    def add_blind(self, player: str, amount: int, blind_type: str):
        self.blinds.append({'player': player, 'amount': amount, 'type': blind_type})
    
    def add_hole_cards(self, player: str, cards: List[str]):
        self.hole_cards[player] = cards
    
    def set_board(self, board: List[str]):
        self.board = board
    
    def add_action(self, phase: str, action: Action):
        if phase in self.phases:
            self.phases[phase].append(action)
    
    def set_result(self, result: Dict):
        self.result = result
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a dict para JSON"""
        return {
            'hand_id': self.hand_id,
            'timestamp': self.timestamp,
            'players': self.players,
            'button': self.button_idx,
            'blinds': self.blinds,
            'hole_cards': self.hole_cards,
            'board': self.board,
            'phases': {
                phase: [a.to_dict() for a in actions]
                for phase, actions in self.phases.items()
            },
            'result': self.result
        }
    
    def to_readable_string(self) -> str:
        """Genera texto legible tipo PokerStars"""
        lines = []
        lines.append(f"Hand #{self.hand_id}")
        lines.append(f"Button: Seat {self.button_idx + 1}")
        lines.append("")
        
        # Players
        for p in self.players:
            lines.append(f"Seat {p['seat'] + 1}: {p['name']} ({p['stack']} in chips)")
        lines.append("")
        
        # Blinds
        for blind in self.blinds:
            lines.append(f"{blind['player']}: posts {blind['type']} {blind['amount']}")
        lines.append("")
        
        # Hole cards
        lines.append("*** HOLE CARDS ***")
        for player, cards in self.hole_cards.items():
            lines.append(f"Dealt to {player} [{' '.join(cards)}]")
        lines.append("")
        
        # Actions por fase
        if self.phases['preflop']:
            lines.append("*** PREFLOP ***")
            for action in self.phases['preflop']:
                lines.append(str(action))
            lines.append("")
        
        if self.board and len(self.board) >= 3:
            lines.append(f"*** FLOP *** [{' '.join(self.board[:3])}]")
            for action in self.phases['flop']:
                lines.append(str(action))
            lines.append("")
        
        if len(self.board) >= 4:
            lines.append(f"*** TURN *** [{' '.join(self.board[:3])}] [{self.board[3]}]")
            for action in self.phases['turn']:
                lines.append(str(action))
            lines.append("")
        
        if len(self.board) >= 5:
            lines.append(f"*** RIVER *** [{' '.join(self.board[:4])}] [{self.board[4]}]")
            for action in self.phases['river']:
                lines.append(str(action))
            lines.append("")
        
        # Result
        if self.result:
            lines.append("*** SUMMARY ***")
            lines.append(f"Total pot: {self.result['pot']}")
            for winner, amount in self.result.get('winners', []):
                lines.append(f"{winner} wins {amount}")
        
        return "\n".join(lines)


# ==================== POKER ENGINE ====================

class PokerEngine:
    """Motor principal de póker - reemplazo de SimplePokerEngine"""
    
    def __init__(
        self,
        profiles: List[PlayerProfile],
        config: SimConfig,
        decision_maker: Optional[Callable] = None
    ):
        self.profiles = profiles
        self.config = config
        self.hand_history_log: List[HandHistory] = []
        self.button_idx = 0
        self._decision_maker = decision_maker or self._default_decision_maker
    
    def play_hand(self) -> Dict[str, Any]:
        """Juega una mano completa"""
        hand = PokerHand(
            players=self.profiles,
            config=self.config,
            button_idx=self.button_idx
        )
        
        result = hand.play(self._decision_maker)
        
        # Guardar history
        self.hand_history_log.append(hand.history)
        
        # Avanzar button
        self.button_idx = (self.button_idx + 1) % len(self.profiles)
        
        return result
    
    def _default_decision_maker(
        self,
        game_state: Dict[str, Any],
        player: PlayerState
    ) -> Action:
        """
        Decision maker por defecto usando heurística.
        Compatible con heuristic_policy_probabilities del código original.
        """
        # Importar solo si es necesario
        from poker_gui import heuristic_policy_probabilities, estimate_equity
        
        # Construir feature vector
        equity = 0.5
        if player.hole and hasattr(self, '_current_board'):
            try:
                # Calcular equity si tenemos board
                board = getattr(self, '_current_board', [])
                equity = estimate_equity(player.hole, board, n=50)
            except Exception:
                equity = 0.5
        
        fv = {
            'equity': equity,
            'pot': sum(p.committed_total for p in self.profiles if hasattr(self, 'profiles')),
            'to_call': game_state['current_bet'] - player.committed_this_round
        }
        
        # Obtener probabilidades de acciones
        probs = heuristic_policy_probabilities(fv, player.profile)
        
        # Seleccionar acción
        action_str = random.choices(
            list(probs.keys()),
            weights=list(probs.values()),
            k=1
        )[0]
        
        # Convertir a Action válida
        current_bet = game_state['current_bet']
        to_call = current_bet - player.committed_this_round
        
        if action_str == 'fold':
            if to_call == 0:
                # No puede fold sin apuesta, check
                return Action(ActionType.CHECK, 0, player.profile.name, 'unknown', to_call)
            return Action(ActionType.FOLD, 0, player.profile.name, 'unknown', to_call)
        
        elif action_str == 'call':
            if to_call == 0:
                return Action(ActionType.CHECK, 0, player.profile.name, 'unknown', 0)
            amount = min(to_call, player.profile.stack)
            if amount == player.profile.stack:
                return Action(ActionType.ALL_IN, amount, player.profile.name, 'unknown', to_call)
            return Action(ActionType.CALL, amount, player.profile.name, 'unknown', to_call)
        
        elif action_str == 'bet':
            pot = fv['pot']
            bet_size = random.choice(self.config.bet_sizes)
            amount = int(max(game_state['bb_amount'], pot * bet_size))
            amount = min(amount, player.profile.stack)
            
            if amount == player.profile.stack:
                return Action(ActionType.ALL_IN, amount, player.profile.name, 'unknown', to_call)
            
            if current_bet > 0:
                # Debe raise, no bet
                raise_to = current_bet + amount
                if raise_to <= current_bet + game_state['min_raise']:
                    raise_to = current_bet + game_state['min_raise']
                raise_to = min(raise_to, current_bet + player.profile.stack)
                return Action(ActionType.RAISE, raise_to, player.profile.name, 'unknown', to_call)
            else:
                return Action(ActionType.BET, amount, player.profile.name, 'unknown', 0)
        
        # Default: check/fold
        if to_call == 0:
            return Action(ActionType.CHECK, 0, player.profile.name, 'unknown', 0)
        return Action(ActionType.FOLD, 0, player.profile.name, 'unknown', to_call)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calcula estadísticas de las manos jugadas"""
        if not self.hand_history_log:
            return {}
        
        stats = {
            'total_hands': len(self.hand_history_log),
            'by_player': defaultdict(lambda: {
                'hands_played': 0,
                'hands_won': 0,
                'total_winnings': 0,
                'vpip_count': 0,
                'pfr_count': 0
            })
        }
        
        for history in self.hand_history_log:
            # Contar por jugador
            for player_info in history.players:
                name = player_info['name']
                stats['by_player'][name]['hands_played'] += 1
            
            # Ganadores
            if history.result:
                for winner_name, amount in history.result.get('winners', []):
                    stats['by_player'][winner_name]['hands_won'] += 1
                    stats['by_player'][winner_name]['total_winnings'] += amount
            
            # VPIP/PFR
            for action in history.phases['preflop']:
                if action.type in [ActionType.CALL, ActionType.RAISE, ActionType.BET]:
                    stats['by_player'][action.player_name]['vpip_count'] += 1
                if action.type in [ActionType.RAISE, ActionType.BET]:
                    stats['by_player'][action.player_name]['pfr_count'] += 1
        
        # Calcular %
        for name, data in stats['by_player'].items():
            hands = data['hands_played']
            if hands > 0:
                data['vpip'] = data['vpip_count'] / hands
                data['pfr'] = data['pfr_count'] / hands
                data['win_rate'] = data['hands_won'] / hands
        
        return dict(stats)


# ==================== EXPORTS ====================

__all__ = [
    'PokerEngine',
    'PokerHand',
    'PlayerProfile',
    'PlayerState',
    'SimConfig',
    'Action',
    'ActionType',
    'ActionValidator',
    'BettingRound',
    'SidePotManager',
    'HandHistory'
]