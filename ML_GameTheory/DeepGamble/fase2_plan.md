# FASE 2: SISTEMA DE FEATURES RICO Y GAME STATE
## Estado: 🟡 PENDIENTE | Depende de: ✅ Fase 1 COMPLETA

---

## 🎯 OBJETIVOS DE LA FASE 2

**Meta**: Construir un sistema completo de extracción de features y representación del estado del juego para ML avanzado.

**Entregables**:
1. ✅ GameState completo con toda la información relevante
2. ✅ FeatureExtractor que convierte GameState a vectores numéricos
3. ✅ Calculadores de métricas avanzadas (outs, pot odds, implied odds)
4. ✅ Position awareness y encoding
5. ✅ Action sequence encoding
6. ✅ Board texture analysis
7. ✅ Tests unitarios completos

---

## 📐 ARQUITECTURA

```
poker_features/
├── __init__.py
├── game_state.py       # GameState dataclass completo
├── feature_extractor.py # FeatureExtractor principal
├── calculators.py      # Pot odds, outs, equity, etc.
├── encoders.py         # Card encoding, position, etc.
├── board_texture.py    # Análisis de textura del board
└── tests/
    ├── test_game_state.py
    ├── test_feature_extractor.py
    ├── test_calculators.py
    └── test_encoders.py
```

---

## 🔧 COMPONENTES A IMPLEMENTAR

### 1. GameState Completo

```python
@dataclass
class GameState:
    """Estado completo del juego para decision making"""
    
    # Identificación
    hand_id: str
    phase: str  # preflop, flop, turn, river
    
    # Contexto financiero
    pot: int
    current_bet: int
    min_raise: int
    
    # Jugador actual
    player_name: str
    hole_cards: List[str]
    stack: int
    position: str  # BTN, SB, BB, UTG, MP, CO
    committed_this_round: int
    committed_total: int
    
    # Board
    board: List[str]  # 0-5 cartas
    
    # Oponentes
    num_players: int
    num_active: int
    num_all_in: int
    opponents_stacks: List[int]
    
    # Histórico de acciones
    actions_preflop: List[Action]
    actions_flop: List[Action]
    actions_turn: List[Action]
    actions_river: List[Action]
    
    # Métricas calculadas (lazy evaluation)
    _equity: Optional[float] = None
    _pot_odds: Optional[float] = None
    _outs: Optional[int] = None
    _implied_odds: Optional[float] = None
    
    @property
    def to_call(self) -> int:
        return self.current_bet - self.committed_this_round
    
    @property
    def pot_odds(self) -> float:
        if self._pot_odds is None:
            self._pot_odds = calculate_pot_odds(self.to_call, self.pot)
        return self._pot_odds
    
    @property
    def equity(self) -> float:
        if self._equity is None:
            self._equity = estimate_equity(self.hole_cards, self.board)
        return self._equity
```

**Tests necesarios**:
- ✅ Crear GameState válido
- ✅ Calcular to_call correctamente
- ✅ Lazy evaluation de equity
- ✅ Serialización/deserialización

### 2. Calculadores de Métricas

```python
class PokerCalculators:
    """Calculadores de métricas de póker"""
    
    @staticmethod
    def calculate_pot_odds(to_call: int, pot: int) -> float:
        """
        Pot odds = to_call / (pot + to_call)
        
        Ejemplo: 50 to call, pot de 100 = 50 / 150 = 0.33 (33%)
        """
        if to_call == 0:
            return 0.0
        return to_call / (pot + to_call)
    
    @staticmethod
    def calculate_outs(hole: List[str], board: List[str], target: str) -> int:
        """
        Cuenta outs para mejorar la mano.
        
        Args:
            target: 'flush', 'straight', 'pair', 'two_pair', 'set', etc.
        """
        # Implementación según target
        pass
    
    @staticmethod
    def calculate_implied_odds(
        pot: int,
        to_call: int,
        opponent_stack: int,
        win_probability: float
    ) -> float:
        """
        Implied odds consideran fichas futuras que puedes ganar.
        
        IO = (pot + expected_future_bets) / to_call
        """
        expected_future = opponent_stack * win_probability * 0.3  # heurística
        return (pot + expected_future) / max(1, to_call)
    
    @staticmethod
    def count_outs_flush_draw(hole: List[str], board: List[str]) -> int:
        """Cuenta outs para flush draw"""
        from collections import Counter
        
        all_cards = hole + board
        suits = Counter(c[1] for c in all_cards)
        
        # Si tenemos 4 de un palo, tenemos flush draw
        for suit, count in suits.items():
            if count == 4:
                # 13 cartas del palo - 4 que vemos = 9 outs
                return 9
        return 0
    
    @staticmethod
    def count_outs_straight_draw(hole: List[str], board: List[str]) -> int:
        """Cuenta outs para straight draw (OESD = 8, gutshot = 4)"""
        # Implementación compleja, ver algoritmo de detección de escaleras
        pass
```

**Tests necesarios**:
- ✅ Pot odds con varios escenarios
- ✅ Detectar flush draw (9 outs)
- ✅ Detectar OESD (8 outs)
- ✅ Detectar gutshot (4 outs)
- ✅ Implied odds calculation

### 3. Encoders

```python
class CardEncoder:
    """Encode cartas a representación numérica"""
    
    @staticmethod
    def one_hot_card(card: str) -> np.ndarray:
        """
        One-hot encoding de una carta (52 dimensiones).
        
        Ejemplo: 'Ah' -> [0,0,...,1,...,0] (posición 51)
        """
        from poker_engine_v2 import RANKS, SUITS, DECK
        
        if card not in DECK:
            return np.zeros(52)
        
        idx = DECK.index(card)
        vec = np.zeros(52)
        vec[idx] = 1
        return vec
    
    @staticmethod
    def embed_card(card: str, embedding_dim: int = 8) -> np.ndarray:
        """
        Embedding denso de carta (más eficiente que one-hot).
        
        Usa rank + suit como features separadas.
        """
        from poker_engine_v2 import RANKS, SUITS
        
        rank = RANKS.index(card[0]) / len(RANKS)  # Normalizado 0-1
        suit = SUITS.index(card[1]) / len(SUITS)
        
        # Características derivadas
        is_high_card = 1.0 if card[0] in 'TJQKA' else 0.0
        is_suited_connector = 0.0  # Requiere contexto
        
        return np.array([rank, suit, is_high_card, is_suited_connector])
    
    @staticmethod
    def encode_hole_cards(hole: List[str]) -> np.ndarray:
        """Encode hole cards (2 cartas)"""
        if len(hole) != 2:
            return np.zeros(104)  # 52 * 2
        
        c1 = CardEncoder.one_hot_card(hole[0])
        c2 = CardEncoder.one_hot_card(hole[1])
        return np.concatenate([c1, c2])
    
    @staticmethod
    def encode_board(board: List[str], phase: str) -> np.ndarray:
        """
        Encode board según la fase.
        
        Preflop: zeros
        Flop: 3 cartas
        Turn: 4 cartas
        River: 5 cartas
        """
        max_board = 5
        encoded = []
        
        for i in range(max_board):
            if i < len(board):
                encoded.append(CardEncoder.one_hot_card(board[i]))
            else:
                encoded.append(np.zeros(52))
        
        return np.concatenate(encoded)


class PositionEncoder:
    """Encode posición en la mesa"""
    
    POSITIONS = ['BTN', 'SB', 'BB', 'UTG', 'UTG+1', 'MP', 'MP+1', 'CO']
    
    @staticmethod
    def one_hot_position(position: str) -> np.ndarray:
        """One-hot encoding de posición"""
        vec = np.zeros(len(PositionEncoder.POSITIONS))
        try:
            idx = PositionEncoder.POSITIONS.index(position)
            vec[idx] = 1
        except ValueError:
            pass
        return vec
    
    @staticmethod
    def position_score(position: str) -> float:
        """
        Score de posición (más alto = mejor posición).
        
        BTN = 1.0, SB = 0.0
        """
        try:
            idx = PositionEncoder.POSITIONS.index(position)
            return 1.0 - (idx / len(PositionEncoder.POSITIONS))
        except ValueError:
            return 0.5


class ActionSequenceEncoder:
    """Encode secuencia de acciones (para RNN/LSTM)"""
    
    @staticmethod
    def encode_action_sequence(
        actions: List[Action],
        max_length: int = 20
    ) -> np.ndarray:
        """
        Encode secuencia de acciones.
        
        Cada acción: [type_onehot (6), amount_norm (1), player_id (1)]
        """
        action_types = ['fold', 'check', 'call', 'bet', 'raise', 'all_in']
        action_dim = len(action_types) + 2  # +2 para amount y player
        
        encoded = np.zeros((max_length, action_dim))
        
        for i, action in enumerate(actions[:max_length]):
            # One-hot del tipo
            try:
                type_idx = action_types.index(action.type.value)
                encoded[i, type_idx] = 1
            except (ValueError, AttributeError):
                pass
            
            # Amount normalizado
            encoded[i, len(action_types)] = action.amount / 1000.0
            
            # Player ID (hash simple)
            encoded[i, len(action_types) + 1] = hash(action.player_name) % 10 / 10.0
        
        return encoded.flatten()
```

**Tests necesarios**:
- ✅ One-hot encoding de cartas correctas
- ✅ Embedding de cartas
- ✅ Encode board en diferentes fases
- ✅ Position encoding
- ✅ Action sequence encoding

### 4. Board Texture Analysis

```python
class BoardTextureAnalyzer:
    """Analiza la textura del board (wet/dry, conectado, etc.)"""
    
    @staticmethod
    def is_paired(board: List[str]) -> bool:
        """Board tiene pareja"""
        ranks = [c[0] for c in board]
        return len(ranks) != len(set(ranks))
    
    @staticmethod
    def is_monotone(board: List[str]) -> bool:
        """Todas las cartas del mismo palo"""
        if len(board) < 3:
            return False
        suits = [c[1] for c in board]
        return len(set(suits)) == 1
    
    @staticmethod
    def is_two_tone(board: List[str]) -> bool:
        """Dos cartas del mismo palo"""
        if len(board) < 3:
            return False
        suits = [c[1] for c in board]
        suit_counts = {s: suits.count(s) for s in set(suits)}
        return any(count >= 2 for count in suit_counts.values())
    
    @staticmethod
    def connectivity_score(board: List[str]) -> float:
        """
        Score de conectividad (0-1).
        
        1.0 = muy conectado (ej: 789)
        0.0 = desconectado (ej: 2 7 K)
        """
        if len(board) < 3:
            return 0.0
        
        from poker_engine_v2 import RANKS
        ranks_idx = sorted([RANKS.index(c[0]) for c in board])
        
        # Calcular gaps
        gaps = [ranks_idx[i+1] - ranks_idx[i] for i in range(len(ranks_idx)-1)]
        avg_gap = sum(gaps) / len(gaps)
        
        # Score inversamente proporcional al gap promedio
        return max(0.0, 1.0 - (avg_gap - 1) / 5.0)
    
    @staticmethod
    def high_card_score(board: List[str]) -> float:
        """Score de cartas altas en el board"""
        from poker_engine_v2 import RANKS
        
        high_cards = 'TJQKA'
        count = sum(1 for c in board if c[0] in high_cards)
        return count / len(board) if board else 0.0
    
    @staticmethod
    def board_features(board: List[str]) -> Dict[str, float]:
        """Retorna todas las features de textura"""
        return {
            'paired': float(BoardTextureAnalyzer.is_paired(board)),
            'monotone': float(BoardTextureAnalyzer.is_monotone(board)),
            'two_tone': float(BoardTextureAnalyzer.is_two_tone(board)),
            'connectivity': BoardTextureAnalyzer.connectivity_score(board),
            'high_cards': BoardTextureAnalyzer.high_card_score(board),
            'num_cards': len(board) / 5.0  # Normalizado
        }
```

**Tests necesarios**:
- ✅ Detectar board paired
- ✅ Detectar monotone (flush possible)
- ✅ Connectivity score correcto
- ✅ High card score

### 5. FeatureExtractor Principal

```python
class FeatureExtractor:
    """
    Extractor principal que combina todos los encoders.
    
    Genera vector de features completo para ML.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'use_one_hot_cards': True,
            'use_board_texture': True,
            'use_action_sequence': True,
            'max_action_sequence': 20
        }
    
    def extract(self, game_state: GameState) -> np.ndarray:
        """
        Extrae features completos de un GameState.
        
        Returns:
            Vector numpy con todas las features
        """
        features = []
        
        # 1. Hole cards (104 dims si one-hot)
        if self.config['use_one_hot_cards']:
            features.append(CardEncoder.encode_hole_cards(game_state.hole_cards))
        else:
            # Embedding denso (8 dims)
            for card in game_state.hole_cards:
                features.append(CardEncoder.embed_card(card))
        
        # 2. Board cards (260 dims si one-hot)
        if self.config['use_one_hot_cards']:
            features.append(CardEncoder.encode_board(game_state.board, game_state.phase))
        
        # 3. Position (8 dims one-hot)
        features.append(PositionEncoder.one_hot_position(game_state.position))
        
        # 4. Stack/Pot features (escalares normalizados)
        stack_features = np.array([
            game_state.stack / 1000.0,
            game_state.pot / 1000.0,
            game_state.to_call / 1000.0,
            game_state.current_bet / 1000.0,
            game_state.committed_this_round / 1000.0,
            game_state.committed_total / 1000.0,
        ])
        features.append(stack_features)
        
        # 5. Pot odds y equity
        odds_features = np.array([
            game_state.pot_odds,
            game_state.equity,
            game_state.pot_odds - game_state.equity,  # Diferencia (importante!)
        ])
        features.append(odds_features)
        
        # 6. Phase encoding (4 dims one-hot)
        phases = ['preflop', 'flop', 'turn', 'river']
        phase_vec = np.zeros(4)
        try:
            phase_vec[phases.index(game_state.phase)] = 1
        except ValueError:
            pass
        features.append(phase_vec)
        
        # 7. Opponent info
        opp_features = np.array([
            game_state.num_players / 10.0,
            game_state.num_active / 10.0,
            game_state.num_all_in / 10.0,
        ])
        features.append(opp_features)
        
        # 8. Board texture
        if self.config['use_board_texture']:
            texture = BoardTextureAnalyzer.board_features(game_state.board)
            texture_vec = np.array(list(texture.values()))
            features.append(texture_vec)
        
        # 9. Action sequence (historico)
        if self.config['use_action_sequence']:
            all_actions = (
                game_state.actions_preflop +
                game_state.actions_flop +
                game_state.actions_turn +
                game_state.actions_river
            )
            action_seq = ActionSequenceEncoder.encode_action_sequence(
                all_actions,
                self.config['max_action_sequence']
            )
            features.append(action_seq)
        
        # Concatenar todo
        return np.concatenate(features)
    
    def get_feature_names(self) -> List[str]:
        """Retorna nombres de features para interpretabilidad"""
        names = []
        
        if self.config['use_one_hot_cards']:
            names += [f'hole_{i}' for i in range(104)]
            names += [f'board_{i}' for i in range(260)]
        
        names += [f'pos_{p}' for p in PositionEncoder.POSITIONS]
        names += ['stack_norm', 'pot_norm', 'to_call_norm', 'current_bet_norm', 
                  'committed_round', 'committed_total']
        names += ['pot_odds', 'equity', 'odds_equity_diff']
        names += ['phase_preflop', 'phase_flop', 'phase_turn', 'phase_river']
        names += ['num_players', 'num_active', 'num_all_in']
        
        if self.config['use_board_texture']:
            names += ['paired', 'monotone', 'two_tone', 'connectivity', 
                      'high_cards', 'num_board_cards']
        
        if self.config['use_action_sequence']:
            names += [f'action_seq_{i}' for i in range(self.config['max_action_sequence'] * 8)]
        
        return names
    
    def get_feature_dim(self) -> int:
        """Retorna dimensión total del vector de features"""
        dummy_state = self._create_dummy_state()
        return len(self.extract(dummy_state))
    
    def _create_dummy_state(self) -> GameState:
        """Crea GameState dummy para calcular dimensiones"""
        return GameState(
            hand_id='dummy',
            phase='flop',
            pot=100,
            current_bet=50,
            min_raise=50,
            player_name='Dummy',
            hole_cards=['Ah', 'Kd'],
            stack=1000,
            position='BTN',
            committed_this_round=0,
            committed_total=0,
            board=['Jh', 'Tc', '9s'],
            num_players=2,
            num_active=2,
            num_all_in=0,
            opponents_stacks=[1000],
            actions_preflop=[],
            actions_flop=[],
            actions_turn=[],
            actions_river=[]
        )
```

**Tests necesarios**:
- ✅ Extract features de GameState completo
- ✅ Dimensión correcta del vector
- ✅ Nombres de features coinciden con dimensión
- ✅ Features normalizadas (0-1 range)

---

## 📊 MÉTRICAS DE ÉXITO

### Tests de Cobertura
- ✅ 100% de GameState y calculators
- ✅ 95%+ de encoders
- ✅ 90%+ de FeatureExtractor
- ✅ 85%+ de BoardTextureAnalyzer

### Tests de Integración
- ✅ Extract features de 1000 GameStates sin errores
- ✅ Features son determinísticas (mismo input = mismo output)
- ✅ Todos los valores están en rango esperado

### Validación
- ✅ Feature vector puede alimentar modelo de ML
- ✅ Dimensiones compatibles con PyTorch/TensorFlow
- ✅ Performance aceptable (< 1ms por extracción)

---

## 📝 CHECKLIST DE IMPLEMENTACIÓN

### Día 1-2: GameState y Calculators
- [ ] Implementar GameState completo
- [ ] Implementar PokerCalculators
- [ ] Tests para GameState (10+ tests)
- [ ] Tests para Calculators (15+ tests)

### Día 3-4: Encoders
- [ ] Implementar CardEncoder
- [ ] Implementar PositionEncoder
- [ ] Implementar ActionSequenceEncoder
- [ ] Tests para encoders (20+ tests)

### Día 5: Board Texture
- [ ] Implementar BoardTextureAnalyzer
- [ ] Tests para board texture (10+ tests)

### Día 6-7: FeatureExtractor
- [ ] Implementar FeatureExtractor principal
- [ ] Configuración flexible
- [ ] Tests de integración (15+ tests)
- [ ] Benchmark de performance

### Día 8: Integration
- [ ] Integrar con PokerEngine de Fase 1
- [ ] Generar dataset de ejemplo
- [ ] Documentación completa
- [ ] Ejemplos de uso

---

## 🔗 INTEGRACIÓN CON FASE 1

### Modificaciones en poker_engine_v2.py

```python
# Añadir método a PokerHand para generar GameState

class PokerHand:
    # ... código existente ...
    
    def get_current_game_state(self, player: PlayerState) -> GameState:
        """Genera GameState para el jugador actual"""
        from poker_features import GameState
        
        # Determinar posición
        position = self._get_player_position(player.seat)
        
        # Recolectar acciones por fase
        # (asumiendo que tenemos tracking interno)
        
        return GameState(
            hand_id=self.hand_id,
            phase=self.phase,
            pot=sum(p.committed_total for p in self.players),
            current_bet=self._current_bet,  # Necesita tracking
            min_raise=self._min_raise,
            player_name=player.profile.name,
            hole_cards=player.hole,
            stack=player.profile.stack,
            position=position,
            committed_this_round=player.committed_this_round,
            committed_total=player.committed_total,
            board=self.board,
            num_players=len(self.players),
            num_active=sum(1 for p in self.players if p.is_active()),
            num_all_in=sum(1 for p in self.players if p.all_in),
            opponents_stacks=[p.profile.stack for p in self.players if p != player],
            actions_preflop=self.history.phases['preflop'],
            actions_flop=self.history.phases['flop'],
            actions_turn=self.history.phases['turn'],
            actions_river=self.history.phases['river']
        )
```

---

## 🎨 EJEMPLO DE USO

```python
from poker_engine_v2 import PokerEngine, PlayerProfile, SimConfig
from poker_features import FeatureExtractor, GameState

# Setup
profiles = [PlayerProfile(name='Hero'), PlayerProfile(name='Villain')]
config = SimConfig()
engine = PokerEngine(profiles, config)

# Feature extractor
extractor = FeatureExtractor(config={
    'use_one_hot_cards': True,
    'use_board_texture': True,
    'use_action_sequence': True,
    'max_action_sequence': 20
})

# Jugar mano y extraer features
result = engine.play_hand()

# Obtener GameState de última acción
# (esto requiere modificar PokerHand para exponer state)
game_state = engine.get_last_game_state()

# Extract features
features = extractor.extract(game_state)

print(f"Feature vector dimension: {len(features)}")
print(f"Feature names: {extractor.get_feature_names()[:10]}...")  # Primeras 10

# Usar en modelo de ML
import torch
features_tensor = torch.tensor(features, dtype=torch.float32)
# prediction = model(features_tensor)
```

---

## 💾 SAVE POINTS

Guardar progreso después de:
1. ✅ GameState y Calculators funcionando
2. ✅ Encoders completos con tests
3. ✅ FeatureExtractor funcionando
4. ✅ Integración con Fase 1 completa

---

## ⏱️ ESTIMACIÓN DE TIEMPO

**Optimista**: 5-6 días
**Realista**: 8-10 días
**Pesimista**: 12-14 días

---

## 📋 CRITERIOS DE FINALIZACIÓN

✅ Fase 2 está COMPLETA cuando:
1. Todos los tests pasan (95%+ coverage)
2. Feature extraction funciona en 1000+ game states
3. Features son determinísticas y reproducibles
4. Integración con Fase 1 sin romper funcionalidad
5. Performance < 1ms por extracción
6. Documentación y ejemplos completos

---

**Estado**: 🟡 PENDIENTE
**Prioridad**: 🔥 ALTA
**Bloqueante para**: Fase 3 (Visualización), Fase 4 (ML Avanzado)