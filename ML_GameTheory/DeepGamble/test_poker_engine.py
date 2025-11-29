"""
test_poker_engine_v2.py

Test suite completo para el motor de póker - FASE 1

Ejecutar con:
    pytest test_poker_engine_v2.py -v
    
O con coverage:
    pytest test_poker_engine_v2.py --cov=poker_engine_v2 --cov-report=html
"""

import pytest
import random
from poker_engine_v2 import (
    Action, ActionType, ActionValidator,
    PlayerProfile, PlayerState, SimConfig,
    BettingRound, SidePotManager, PokerHand, PokerEngine,
    HandHistory
)


# ==================== FIXTURES ====================

@pytest.fixture
def two_players():
    """Dos jugadores con stacks iguales"""
    return [
        PlayerProfile(name='Hero', stack=1000),
        PlayerProfile(name='Villain', stack=1000)
    ]

@pytest.fixture
def three_players():
    """Tres jugadores con stacks iguales"""
    return [
        PlayerProfile(name='P1', stack=1000),
        PlayerProfile(name='P2', stack=1000),
        PlayerProfile(name='P3', stack=1000)
    ]

@pytest.fixture
def player_states_two():
    """Estados de dos jugadores"""
    p1 = PlayerProfile(name='P1', stack=1000)
    p2 = PlayerProfile(name='P2', stack=1000)
    return [PlayerState(p1, 0), PlayerState(p2, 1)]

@pytest.fixture
def config():
    """Configuración estándar"""
    return SimConfig(small_blind=5, big_blind=10, starting_stack=1000)


# ==================== TEST ACTION ====================

class TestAction:
    """Tests para Action y ActionType"""
    
    def test_action_creation(self):
        """Test crear acción válida"""
        action = Action(
            type=ActionType.RAISE,
            amount=50,
            player_name='Hero',
            phase='flop',
            to_call=20
        )
        assert action.type == ActionType.RAISE
        assert action.amount == 50
        assert action.player_name == 'Hero'
        assert action.phase == 'flop'
        assert action.to_call == 20
    
    def test_action_str(self):
        """Test representación string de acciones"""
        fold = Action(ActionType.FOLD, 0, 'P1', 'preflop')
        assert 'folds' in str(fold)
        
        check = Action(ActionType.CHECK, 0, 'P1', 'flop')
        assert 'checks' in str(check)
        
        call = Action(ActionType.CALL, 50, 'P1', 'turn')
        assert 'calls 50' in str(call)
        
        bet = Action(ActionType.BET, 100, 'P1', 'river')
        assert 'bets 100' in str(bet)
        
        raise_action = Action(ActionType.RAISE, 200, 'P1', 'flop')
        assert 'raises to 200' in str(raise_action)
    
    def test_action_to_dict(self):
        """Test serialización a dict"""
        action = Action(ActionType.BET, 50, 'Hero', 'flop', to_call=0)
        d = action.to_dict()
        
        assert d['type'] == 'bet'
        assert d['amount'] == 50
        assert d['player'] == 'Hero'
        assert d['phase'] == 'flop'
        assert 'timestamp' in d


# ==================== TEST ACTION VALIDATOR ====================

class TestActionValidator:
    """Tests para validación de acciones"""
    
    def test_legal_actions_no_bet(self):
        """Test acciones legales sin apuesta activa"""
        legal = ActionValidator.get_legal_actions(
            player_stack=1000,
            player_committed=0,
            current_bet=0,
            min_raise=10,
            can_check=True
        )
        
        assert ActionType.FOLD in legal
        assert ActionType.CHECK in legal
        assert ActionType.BET in legal
        assert ActionType.ALL_IN in legal
        assert ActionType.CALL not in legal  # No hay nada que igualar
    
    def test_legal_actions_with_bet(self):
        """Test acciones legales con apuesta activa"""
        legal = ActionValidator.get_legal_actions(
            player_stack=1000,
            player_committed=0,
            current_bet=50,
            min_raise=50,
            can_check=False
        )
        
        assert ActionType.FOLD in legal
        assert ActionType.CALL in legal
        assert ActionType.RAISE in legal
        assert ActionType.ALL_IN in legal
        assert ActionType.CHECK not in legal
        assert ActionType.BET not in legal
    
    def test_legal_actions_insufficient_stack(self):
        """Test acciones con stack insuficiente para raise"""
        legal = ActionValidator.get_legal_actions(
            player_stack=30,
            player_committed=0,
            current_bet=50,
            min_raise=50,
            can_check=False
        )
        
        assert ActionType.FOLD in legal
        assert ActionType.ALL_IN in legal
        assert ActionType.CALL not in legal  # No alcanza para call
        assert ActionType.RAISE not in legal  # No alcanza para raise
    
    def test_validate_fold(self):
        """Test validar fold (siempre válido)"""
        action = Action(ActionType.FOLD, 0, 'P1', 'flop')
        valid, reason = ActionValidator.validate_action(action, 1000, 0, 50, 50)
        assert valid
        assert reason == "OK"
    
    def test_validate_check_invalid(self):
        """Test check inválido cuando hay apuesta"""
        action = Action(ActionType.CHECK, 0, 'P1', 'flop')
        valid, reason = ActionValidator.validate_action(action, 1000, 0, 50, 50)
        assert not valid
        assert 'must call' in reason.lower()
    
    def test_validate_check_valid(self):
        """Test check válido sin apuesta"""
        action = Action(ActionType.CHECK, 0, 'P1', 'flop')
        valid, reason = ActionValidator.validate_action(action, 1000, 0, 0, 10)
        assert valid
    
    def test_validate_call_correct_amount(self):
        """Test call con monto correcto"""
        action = Action(ActionType.CALL, 50, 'P1', 'flop', to_call=50)
        valid, reason = ActionValidator.validate_action(action, 1000, 0, 50, 50)
        assert valid
    
    def test_validate_call_wrong_amount(self):
        """Test call con monto incorrecto"""
        action = Action(ActionType.CALL, 30, 'P1', 'flop', to_call=50)
        valid, reason = ActionValidator.validate_action(action, 1000, 0, 50, 50)
        assert not valid
    
    def test_validate_bet_with_existing_bet(self):
        """Test bet cuando ya hay apuesta (debe usar raise)"""
        action = Action(ActionType.BET, 100, 'P1', 'flop')
        valid, reason = ActionValidator.validate_action(action, 1000, 0, 50, 50)
        assert not valid
        assert 'raise' in reason.lower()
    
    def test_validate_raise_minimum(self):
        """Test raise debe cumplir mínimo"""
        # Current bet = 50, min_raise = 50, entonces min raise to = 100
        action = Action(ActionType.RAISE, 80, 'P1', 'flop')
        valid, reason = ActionValidator.validate_action(action, 1000, 0, 50, 50)
        assert not valid
        
        action_valid = Action(ActionType.RAISE, 100, 'P1', 'flop')
        valid, reason = ActionValidator.validate_action(action_valid, 1000, 0, 50, 50)
        assert valid
    
    def test_validate_all_in(self):
        """Test all-in debe ser por todo el stack"""
        action = Action(ActionType.ALL_IN, 1000, 'P1', 'flop')
        valid, reason = ActionValidator.validate_action(action, 1000, 0, 50, 50)
        assert valid
        
        action_wrong = Action(ActionType.ALL_IN, 500, 'P1', 'flop')
        valid, reason = ActionValidator.validate_action(action_wrong, 1000, 0, 50, 50)
        assert not valid


# ==================== TEST PLAYER STATE ====================

class TestPlayerState:
    """Tests para estado de jugador"""
    
    def test_player_state_creation(self):
        """Test crear estado de jugador"""
        profile = PlayerProfile(name='Hero', stack=1000)
        state = PlayerState(profile, seat=0)
        
        assert state.profile.name == 'Hero'
        assert state.seat == 0
        assert state.profile.stack == 1000
        assert not state.folded
        assert not state.all_in
    
    def test_commit_chips(self):
        """Test commitear fichas"""
        profile = PlayerProfile(name='Hero', stack=1000)
        state = PlayerState(profile, 0)
        
        committed = state.commit(100)
        
        assert committed == 100
        assert state.profile.stack == 900
        assert state.committed_this_round == 100
        assert state.committed_total == 100
        assert not state.all_in
    
    def test_commit_all_in(self):
        """Test commitear todo el stack"""
        profile = PlayerProfile(name='Hero', stack=100)
        state = PlayerState(profile, 0)
        
        committed = state.commit(100)
        
        assert committed == 100
        assert state.profile.stack == 0
        assert state.all_in
    
    def test_commit_more_than_stack(self):
        """Test commitear más del stack disponible"""
        profile = PlayerProfile(name='Hero', stack=50)
        state = PlayerState(profile, 0)
        
        committed = state.commit(100)
        
        assert committed == 50  # Solo commitea lo que tiene
        assert state.profile.stack == 0
        assert state.all_in
    
    def test_reset_for_betting_round(self):
        """Test reset entre rondas"""
        profile = PlayerProfile(name='Hero', stack=1000)
        state = PlayerState(profile, 0)
        
        state.commit(100)
        state.acted_this_round = True
        state.reset_for_betting_round()
        
        assert state.committed_this_round == 0
        assert not state.acted_this_round
        assert state.committed_total == 100  # Se mantiene el total
    
    def test_is_active(self):
        """Test si jugador está activo"""
        profile = PlayerProfile(name='Hero', stack=1000)
        state = PlayerState(profile, 0)
        
        assert state.is_active()
        
        state.folded = True
        assert not state.is_active()
        
        state.folded = False
        state.all_in = True
        assert not state.is_active()


# ==================== TEST SIDE POT MANAGER ====================

class TestSidePotManager:
    """Tests para cálculo de side pots"""
    
    def test_no_side_pot_equal_commits(self):
        """Test sin side pot cuando todos comitean igual"""
        p1 = PlayerState(PlayerProfile('P1', 1000), 0)
        p2 = PlayerState(PlayerProfile('P2', 1000), 1)
        
        p1.commit(100)
        p2.commit(100)
        
        pots = SidePotManager.calculate_pots([p1, p2])
        
        assert len(pots) == 1
        assert pots[0]['amount'] == 200
        assert set(pots[0]['eligible']) == {'P1', 'P2'}
    
    def test_side_pot_one_all_in(self):
        """Test side pot con un all-in"""
        p1 = PlayerState(PlayerProfile('P1', 100), 0)
        p2 = PlayerState(PlayerProfile('P2', 1000), 1)
        
        p1.commit(100)  # all-in
        p1.all_in = True
        p2.commit(200)
        
        pots = SidePotManager.calculate_pots([p1, p2])
        
        # Main pot: 100 * 2 = 200 (ambos elegibles)
        # Side pot: 100 * 1 = 100 (solo P2)
        assert len(pots) == 2
        assert pots[0]['amount'] == 200
        assert set(pots[0]['eligible']) == {'P1', 'P2'}
        assert pots[1]['amount'] == 100
        assert pots[1]['eligible'] == ['P2']
    
    def test_side_pot_multiple_all_ins(self):
        """Test múltiples side pots"""
        p1 = PlayerState(PlayerProfile('P1', 50), 0)
        p2 = PlayerState(PlayerProfile('P2', 100), 1)
        p3 = PlayerState(PlayerProfile('P3', 200), 2)
        
        p1.commit(50)
        p1.all_in = True
        p2.commit(100)
        p2.all_in = True
        p3.commit(200)
        
        pots = SidePotManager.calculate_pots([p1, p2, p3])
        
        # Main pot: 50 * 3 = 150 (todos)
        # Side pot 1: 50 * 2 = 100 (P2, P3)
        # Side pot 2: 100 * 1 = 100 (P3)
        assert len(pots) == 3
        assert pots[0]['amount'] == 150
        assert set(pots[0]['eligible']) == {'P1', 'P2', 'P3'}
        assert pots[1]['amount'] == 100
        assert set(pots[1]['eligible']) == {'P2', 'P3'}
        assert pots[2]['amount'] == 100
        assert pots[2]['eligible'] == ['P3']
    
    def test_side_pot_with_fold(self):
        """Test side pot cuando alguien foldea"""
        p1 = PlayerState(PlayerProfile('P1', 100), 0)
        p2 = PlayerState(PlayerProfile('P2', 1000), 1)
        p3 = PlayerState(PlayerProfile('P3', 1000), 2)
        
        p1.commit(50)
        p1.folded = True  # Foldea tras commitear
        p2.commit(100)
        p2.all_in = True
        p3.commit(200)
        
        pots = SidePotManager.calculate_pots([p1, p2, p3])
        
        # P1 foldeó, no es elegible para ganar
        # Main pot incluye contribución de P1
        assert all('P1' not in pot['eligible'] for pot in pots)


# ==================== TEST BETTING ROUND ====================

class TestBettingRound:
    """Tests para rondas de apuestas"""
    
    def test_simple_check_check(self, player_states_two):
        """Test ronda simple: todos check"""
        def decision_maker(game_state, player):
            return Action(ActionType.CHECK, 0, player.profile.name, 'flop')
        
        betting = BettingRound(player_states_two, bb_amount=10, start_idx=0)
        continues = betting.run(decision_maker)
        
        assert continues
        assert betting.get_total_pot() == 0
        assert all(not p.folded for p in player_states_two)
    
    def test_bet_and_call(self, player_states_two):
        """Test bet y call"""
        actions_to_take = [
            Action(ActionType.BET, 50, 'P1', 'flop'),
            Action(ActionType.CALL, 50, 'P2', 'flop')
        ]
        action_idx = [0]
        
        def decision_maker(game_state, player):
            action = actions_to_take[action_idx[0]]
            action_idx[0] += 1
            return action
        
        betting = BettingRound(player_states_two, bb_amount=10, start_idx=0)
        continues = betting.run(decision_maker)
        
        assert continues
        assert betting.get_total_pot() == 100
        assert player_states_two[0].committed_this_round == 50
        assert player_states_two[1].committed_this_round == 50
    
    def test_bet_and_fold(self, player_states_two):
        """Test bet y fold"""
        actions_to_take = [
            Action(ActionType.BET, 50, 'P1', 'flop'),
            Action(ActionType.FOLD, 0, 'P2', 'flop')
        ]
        action_idx = [0]
        
        def decision_maker(game_state, player):
            action = actions_to_take[action_idx[0]]
            action_idx[0] += 1
            return action
        
        betting = BettingRound(player_states_two, bb_amount=10, start_idx=0)
        continues = betting.run(decision_maker)
        
        assert not continues  # No continúa porque solo queda uno
        assert player_states_two[1].folded
    
    def test_bet_raise_call(self, player_states_two):
        """Test bet, raise y call"""
        actions_to_take = [
            Action(ActionType.BET, 50, 'P1', 'flop'),
            Action(ActionType.RAISE, 150, 'P2', 'flop'),
            Action(ActionType.CALL, 100, 'P1', 'flop')  # Debe igualar 100 más
        ]
        action_idx = [0]
        
        def decision_maker(game_state, player):
            action = actions_to_take[action_idx[0]]
            action_idx[0] += 1
            return action
        
        betting = BettingRound(player_states_two, bb_amount=10, start_idx=0)
        continues = betting.run(decision_maker)
        
        assert continues
        assert betting.current_bet == 150
        assert player_states_two[0].committed_this_round == 150
        assert player_states_two[1].committed_this_round == 150
    
    def test_preflop_with_bb(self, player_states_two):
        """Test preflop con BB ya posteado"""
        # Simular que BB ya comprometió 10
        player_states_two[1].commit(10)
        player_states_two[1].reset_for_betting_round()
        player_states_two[1].commit(10)  # BB en committed_this_round
        
        actions_to_take = [
            Action(ActionType.CALL, 10, 'P1', 'preflop'),
            Action(ActionType.CHECK, 0, 'P2', 'preflop')
        ]
        action_idx = [0]
        
        def decision_maker(game_state, player):
            action = actions_to_take[action_idx[0]]
            action_idx[0] += 1
            return action
        
        betting = BettingRound(player_states_two, bb_amount=10, start_idx=0, is_preflop=True)
        betting.current_bet = 10  # BB define la apuesta inicial
        continues = betting.run(decision_maker)
        
        assert continues
        assert player_states_two[0].committed_this_round == 10
        assert player_states_two[1].committed_this_round == 10


# ==================== TEST POKER HAND ====================

class TestPokerHand:
    """Tests para ejecución de manos completas"""
    
    def test_hand_initialization(self, two_players, config):
        """Test inicialización de mano"""
        hand = PokerHand(two_players, config, button_idx=0)
        
        assert hand.button_idx == 0
        assert len(hand.players) == 2
        assert hand.phase == 'preflop'
        assert len(hand.board) == 0
    
    def test_post_blinds_heads_up(self, two_players, config):
        """Test postear blinds en heads-up"""
        hand = PokerHand(two_players, config, button_idx=0)
        hand._reset()
        hand._post_blinds()
        
        # HU: button es SB
        assert hand.players[0].committed_total == 5  # SB
        assert hand.players[1].committed_total == 10  # BB
    
    def test_deal_hole_cards(self, two_players, config):
        """Test repartir hole cards"""
        hand = PokerHand(two_players, config)
        hand._reset()
        hand._deal_hole_cards()
        
        assert len(hand.players[0].hole) == 2
        assert len(hand.players[1].hole) == 2
        assert hand.players[0].hole[0] != hand.players[1].hole[0]
    
    def test_deal_flop(self, two_players, config):
        """Test repartir flop"""
        hand = PokerHand(two_players, config)
        hand._reset()
        hand._deal_flop()
        
        assert len(hand.board) == 3
        assert all(card in DECK for card in hand.board)
    
    def test_deal_turn(self, two_players, config):
        """Test repartir turn"""
        hand = PokerHand(two_players, config)
        hand._reset()
        hand._deal_flop()
        hand._deal_turn()
        
        assert len(hand.board) == 4
    
    def test_deal_river(self, two_players, config):
        """Test repartir river"""
        hand = PokerHand(two_players, config)
        hand._reset()
        hand._deal_flop()
        hand._deal_turn()
        hand._deal_river()
        
        assert len(hand.board) == 5
    
    def test_complete_hand_all_fold_preflop(self, two_players, config):
        """Test mano que termina en preflop por fold"""
        def decision_maker(game_state, player):
            if player.profile.name == 'Hero':
                return Action(ActionType.RAISE, 30, 'Hero', 'preflop')
            else:
                return Action(ActionType.FOLD, 0, 'Villain', 'preflop')
        
        hand = PokerHand(two_players, config, button_idx=0)
        result = hand.play(decision_maker)
        
        assert result['result'] == 'fold_win'
        assert result['phase'] == 'preflop'
        assert len(result['winners']) == 1
    
    def test_complete_hand_to_showdown(self, two_players, config):
        """Test mano completa hasta showdown"""
        def decision_maker(game_state, player):
            # Todos check/call siempre
            to_call = game_state['current_bet'] - player.committed_this_round
            if to_call > 0:
                amount = min(to_call, player.profile.stack)
                return Action(ActionType.CALL, amount, player.profile.name, game_state.get('phase', 'unknown'))
            else:
                return Action(ActionType.CHECK, 0, player.profile.name, game_state.get('phase', 'unknown'))
        
        hand = PokerHand(two_players, config, button_idx=0)
        result = hand.play(decision_maker)
        
        assert result['result'] == 'showdown'
        assert len(hand.board) == 5
        assert result['pot'] > 0
        assert len(result['winners']) >= 1


# ==================== TEST HAND HISTORY ====================

class TestHandHistory:
    """Tests para logging de manos"""
    
    def test_history_creation(self):
        """Test crear history"""
        history = HandHistory('test-123')
        
        assert history.hand_id == 'test-123'
        assert 'preflop' in history.phases
        assert len(history.board) == 0
    
    def test_add_action(self):
        """Test añadir acción"""
        history = HandHistory('test-123')
        action = Action(ActionType.BET, 50, 'Hero', 'flop')
        
        history.add_action('flop', action)
        
        assert len(history.phases['flop']) == 1
        assert history.phases['flop'][0] == action
    
    def test_to_dict(self):
        """Test serialización"""
        history = HandHistory('test-123')
        history.set_players([{'name': 'Hero', 'stack': 1000, 'seat': 0}])
        history.set_board(['Ah', 'Kd', 'Qc'])
        
        d = history.to_dict()
        
        assert d['hand_id'] == 'test-123'
        assert len(d['players']) == 1
        assert d['board'] == ['Ah', 'Kd', 'Qc']
    
    def test_readable_string(self):
        """Test formato legible"""
        history = HandHistory('abc123')
        history.set_players([
            {'name': 'Hero', 'stack': 1000, 'seat': 0},
            {'name': 'Villain', 'stack': 1000, 'seat': 1}
        ])
        history.add_blind('Hero', 5, 'SB')
        history.add_blind('Villain', 10, 'BB')
        history.add_hole_cards('Hero', ['Ah', 'Kd'])
        
        readable = history.to_readable_string()
        
        assert 'Hand #abc123' in readable
        assert 'Hero' in readable
        assert 'Villain' in readable
        assert 'posts SB 5' in readable


# ==================== TEST POKER ENGINE ====================

class TestPokerEngine:
    """Tests para el motor completo"""
    
    def test_engine_initialization(self, two_players, config):
        """Test inicializar engine"""
        engine = PokerEngine(two_players, config)
        
        assert len(engine.profiles) == 2
        assert engine.button_idx == 0
        assert len(engine.hand_history_log) == 0
    
    def test_play_single_hand(self, two_players, config):
        """Test jugar una mano"""
        engine = PokerEngine(two_players, config)
        result = engine.play_hand()
        
        assert 'pot' in result
        assert 'winners' in result
        assert len(engine.hand_history_log) == 1
        assert engine.button_idx == 1
    
    def test_play_multiple_hands(self, two_players, config):
        """Test jugar múltiples manos"""
        engine = PokerEngine(two_players, config)
        
        for i in range(10):
            result = engine.play_hand()
            assert result['pot'] >= 0
        
        assert len(engine.hand_history_log) == 10
        assert engine.button_idx == 10 % 2
    
    def test_stack_conservation(self, two_players, config):
        """Test que las fichas se conservan"""
        initial_total = sum(p.stack for p in two_players)
        engine = PokerEngine(two_players, config)
        
        for i in range(20):
            engine.play_hand()
            current_total = sum(p.stack for p in engine.profiles)
            assert current_total == initial_total, f"Hand {i}: stacks don't add up"
    
    def test_get_statistics(self, two_players, config):
        """Test obtener estadísticas"""
        engine = PokerEngine(two_players, config)
        
        for i in range(50):
            engine.play_hand()
        
        stats = engine.get_statistics()
        
        assert stats['total_hands'] == 50
        assert 'by_player' in stats
        assert 'Hero' in stats['by_player']
        assert 'Villain' in stats['by_player']
        assert stats['by_player']['Hero']['hands_played'] == 50


# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Tests de integración end-to-end"""
    
    def test_full_simulation_no_crashes(self, three_players, config):
        """Test simulación completa sin crashes"""
        random.seed(42)
        engine = PokerEngine(three_players, config)
        
        for i in range(100):
            try:
                result = engine.play_hand()
                assert result is not None
            except Exception as e:
                pytest.fail(f"Hand {i} crashed: {e}")
    
    def test_stack_sizes_remain_positive(self, two_players, config):
        """Test que los stacks no se vuelven negativos"""
        engine = PokerEngine(two_players, config)
        
        for i in range(100):
            engine.play_hand()
            for player in engine.profiles:
                assert player.stack >= 0, f"Negative stack for {player.name}"
    
    def test_hand_history_complete(self, two_players, config):
        """Test que hand history está completa"""
        engine = PokerEngine(two_players, config)
        engine.play_hand()
        
        history = engine.hand_history_log[0]
        
        assert len(history.players) == 2
        assert history.hand_id is not None
        assert len(history.blinds) == 2
        assert history.result is not None
    
    def test_all_in_scenarios(self):
        """Test escenarios con all-in"""
        # P1 tiene stack corto
        p1 = PlayerProfile(name='Short', stack=50)
        p2 = PlayerProfile(name='Big', stack=1000)
        config = SimConfig(small_blind=5, big_blind=10)
        
        engine = PokerEngine([p1, p2], config)
        
        # Jugar hasta que P1 quede sin fichas o gane mucho
        for i in range(20):
            if p1.stack == 0:
                break
            engine.play_hand()
        
        # Verificar que P1 llegó a 0 o ganó fichas
        assert p1.stack >= 0


# ==================== PERFORMANCE TESTS ====================

class TestPerformance:
    """Tests de rendimiento (opcional)"""
    
    @pytest.mark.slow
    def test_1000_hands_performance(self, two_players, config):
        """Test jugar 1000 manos en tiempo razonable"""
        import time
        
        engine = PokerEngine(two_players, config)
        
        start = time.time()
        for i in range(1000):
            engine.play_hand()
        elapsed = time.time() - start
        
        # Debe completarse en menos de 30 segundos
        assert elapsed < 30, f"1000 hands took {elapsed:.2f}s"
        print(f"\n1000 hands in {elapsed:.2f}s ({1000/elapsed:.1f} hands/sec)")


# ==================== PARAMETRIZED TESTS ====================

@pytest.mark.parametrize("num_players", [2, 3, 4, 6, 9])
def test_different_player_counts(num_players):
    """Test con diferentes números de jugadores"""
    players = [PlayerProfile(name=f'P{i}', stack=1000) for i in range(num_players)]
    config = SimConfig()
    engine = PokerEngine(players, config)
    
    result = engine.play_hand()
    assert result is not None
    assert result['pot'] >= 0


@pytest.mark.parametrize("sb,bb", [(1, 2), (5, 10), (25, 50), (100, 200)])
def test_different_blind_levels(sb, bb):
    """Test con diferentes niveles de ciegas"""
    players = [PlayerProfile(name='P1', stack=10000), PlayerProfile(name='P2', stack=10000)]
    config = SimConfig(small_blind=sb, big_blind=bb)
    engine = PokerEngine(players, config)
    
    result = engine.play_hand()
    assert result['pot'] >= bb  # Al menos el BB debe estar en el pot


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
