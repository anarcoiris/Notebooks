# Plan de Mejora y Completitud: Poker Recursive Trainer

## 📊 ANÁLISIS DEL ESTADO ACTUAL

### Fortalezas Identificadas
✅ **Arquitectura modular** - Separación clara entre GUI, engine y lógica
✅ **Múltiples funcionalidades** - Simulación, análisis, entrenamiento, optimización
✅ **Persistencia** - Carga/guardado de configuraciones y logs
✅ **Integración opcional** - eval7, PyTorch, matplotlib
✅ **GUI organizada** - Sistema de pestañas intuitivo

### Problemas Críticos Detectados

#### 🔴 **1. Motor de Póker Incompleto e Incorrecto**
```python
# PROBLEMA: simulate_betting_simple() es extremadamente simplista
def simulate_betting_simple(self):
    for p in self.profiles:
        if p.folded or p.profile.stack <= 0:
            continue
        # ❌ No hay concepto de "acción mínima" (to_call siempre es 0)
        # ❌ Todos actúan simultáneamente sin turnos
        # ❌ No hay raises progresivos
        # ❌ No se respeta el orden de actuación
```

**Impacto**: El juego no simula póker real, invalidando todo el análisis posterior.

#### 🟠 **2. Sistema de Apuestas Roto**
- `to_call` siempre es 0 en el feature vector
- No hay tracking de la apuesta actual (current_bet)
- Las pot odds nunca se calculan correctamente
- El concepto de "call" comitea 0 chips

#### 🟡 **3. Falta de Fases Completas del Juego**
- **Preflop**: No existe como fase separada
- **Flop/Turn/River**: Se llama a `simulate_betting_simple()` pero sin diferenciar estrategias
- **No hay all-in**: Falta lógica de side pots
- **Showdown simplista**: Ranking incorrecto sin eval7

#### 🟡 **4. Análisis e Historial Deficientes**
```python
# PROBLEMA: No se registran las acciones
record = {
    "board": self.board.copy(),
    "holes": [p.hole.copy() for p in self.profiles],
    "pot": self.pot,
    "result": res
}
# ❌ Falta: acciones por jugador, montos apostados, fold points
```

#### 🟡 **5. Visualización Inexistente**
- No hay representación gráfica de cartas
- No hay gráficos de progreso/estadísticas
- matplotlib importado pero no usado
- No hay replay de manos

#### 🟡 **6. Training/ML Superficial**
- Features mínimos (solo equity + pot)
- No hay state encoding completo
- Dataset sintético poco realista
- No hay validación/testing del modelo

---

## 🎯 PLAN DE MEJORA ESTRUCTURADO

### **FASE 1: CORRECCIÓN DEL MOTOR DE PÓKER** (Prioridad CRÍTICA)

#### 1.1 Sistema de Apuestas Correcto
```python
class BettingRound:
    """Gestiona una ronda completa de apuestas"""
    def __init__(self, players, bb_amount):
        self.players = players
        self.current_bet = 0
        self.min_raise = bb_amount
        self.pot = 0
        self.last_aggressor = None
        
    def run(self):
        """Ejecuta ronda hasta que todos igualen o fold"""
        active = [p for p in self.players if not p.folded]
        while not self._betting_complete(active):
            for player in active:
                action = self._get_player_action(player)
                self._process_action(player, action)
```

**Tareas**:
- [ ] Implementar tracking de `current_bet` y `to_call` por jugador
- [ ] Sistema de turnos con botón/SB/BB correcto
- [ ] Lógica de raises: min_raise, max_raise (all-in)
- [ ] Detección de fin de ronda (todos igualados o folded)
- [ ] Sistema de side pots para all-ins múltiples

#### 1.2 Fases del Juego Completas
```python
class PokerHand:
    """Representa una mano completa con todas sus fases"""
    def __init__(self, players, config):
        self.phases = ['preflop', 'flop', 'turn', 'river']
        self.current_phase = 0
        
    def play(self):
        self.post_blinds()
        self.deal_hole_cards()
        
        # PREFLOP
        if self.run_betting_round('preflop'):
            self.deal_flop()
            
            # FLOP
            if self.run_betting_round('flop'):
                self.deal_turn()
                
                # TURN
                if self.run_betting_round('turn'):
                    self.deal_river()
                    
                    # RIVER
                    if self.run_betting_round('river'):
                        return self.showdown()
        
        return self.award_pot()  # Alguien foldeó
```

**Tareas**:
- [ ] Separar lógica de cada fase
- [ ] Estrategias diferenciadas por fase
- [ ] Tracking de acciones por fase
- [ ] History detallado: `{phase: 'flop', player: 'Hero', action: 'raise', amount: 50}`

#### 1.3 Sistema de Acciones Correcto
```python
@dataclass
class Action:
    type: str  # fold, check, call, bet, raise, all_in
    amount: int
    to_call: int
    player: str
    phase: str
    timestamp: float

class ActionValidator:
    """Valida que las acciones sean legales"""
    @staticmethod
    def is_valid(action, game_state):
        if action.type == 'raise':
            return action.amount >= game_state.min_raise
        # ... más validaciones
```

**Tareas**:
- [ ] Enum de acciones válidas
- [ ] Validador de acciones legales
- [ ] Logger de todas las acciones
- [ ] Replay system desde action log

---

### **FASE 2: SISTEMA DE FEATURES RICO** (Prioridad ALTA)

#### 2.1 Estado del Juego Completo
```python
@dataclass
class GameState:
    # Contexto
    phase: str
    pot: int
    current_bet: int
    to_call: int
    
    # Jugador
    hole_cards: List[str]
    stack: int
    position: str  # BTN, SB, BB, UTG, etc.
    
    # Board
    board: List[str]
    
    # Oponentes
    num_active: int
    num_folded: int
    
    # Histórico
    actions_preflop: List[Action]
    actions_flop: List[Action]
    aggression_history: Dict[str, float]
    
    # Calculados
    equity: float
    pot_odds: float
    implied_odds: float
    outs: int
```

#### 2.2 Feature Engineering
```python
class FeatureExtractor:
    """Convierte GameState en vector numérico para ML"""
    
    def extract(self, state: GameState) -> np.ndarray:
        features = []
        
        # Mano (52 bits one-hot)
        features += self._encode_cards(state.hole_cards)
        
        # Board (260 bits: 5 cartas * 52)
        features += self._encode_board(state.board, state.phase)
        
        # Posición (one-hot 9 posiciones)
        features += self._encode_position(state.position)
        
        # Stack/Pot (normalizados)
        features += [
            state.stack / 1000.0,
            state.pot / 1000.0,
            state.to_call / 1000.0,
            state.pot_odds,
            state.equity
        ]
        
        # Fase (one-hot)
        features += self._encode_phase(state.phase)
        
        # Histórico de acciones (RNN embeddings)
        features += self._encode_action_sequence(state.actions_preflop)
        
        return np.array(features, dtype=np.float32)
```

**Tareas**:
- [ ] Implementar FeatureExtractor completo
- [ ] Calcular outs correctamente
- [ ] Implied odds estimator
- [ ] Position awareness
- [ ] Action sequence encoding

---

### **FASE 3: VISUALIZACIÓN Y UX** (Prioridad MEDIA-ALTA)

#### 3.1 Canvas de Mesa de Póker
```python
class PokerTableCanvas(tk.Canvas):
    """Visualización gráfica de la mesa"""
    
    def __init__(self, parent):
        super().__init__(parent, width=800, height=600, bg='#0a5f0a')
        self.draw_table()
        
    def draw_table(self):
        # Mesa ovalada
        self.create_oval(100, 100, 700, 500, fill='#1a7f1a', width=5)
        
        # Posiciones de jugadores (círculos)
        positions = self._calculate_positions(num_players=6)
        for i, pos in enumerate(positions):
            self._draw_player_seat(pos, i)
    
    def update_hand(self, game_state):
        """Actualiza visualización durante la mano"""
        self._draw_community_cards(game_state.board)
        self._draw_pot(game_state.pot)
        self._draw_player_actions(game_state.last_actions)
        self._highlight_active_player(game_state.active_player)
```

#### 3.2 Gráficos de Análisis
```python
class StatsVisualizer:
    """Genera gráficos con matplotlib"""
    
    def plot_session_results(self, history):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Stack over time
        axes[0, 0].plot(stack_progression)
        axes[0, 0].set_title('Stack Progress')
        
        # VPIP / PFR by position
        axes[0, 1].bar(positions, vpip_values)
        axes[0, 1].set_title('VPIP by Position')
        
        # Win rate by hand strength
        axes[1, 0].scatter(hand_strength, win_rates)
        
        # Action frequency (pie chart)
        axes[1, 1].pie(action_counts, labels=['Fold', 'Call', 'Raise'])
        
        plt.tight_layout()
        return fig
```

#### 3.3 Hand Replayer
```python
class HandReplayer:
    """Reproduce manos paso a paso"""
    
    def __init__(self, hand_history):
        self.history = hand_history
        self.current_step = 0
        
    def next_action(self):
        """Avanza un paso y retorna el estado"""
        if self.current_step < len(self.history['actions']):
            action = self.history['actions'][self.current_step]
            self.current_step += 1
            return self._reconstruct_state(action)
    
    def prev_action(self):
        """Retrocede un paso"""
        if self.current_step > 0:
            self.current_step -= 1
            return self._reconstruct_state(self.history['actions'][self.current_step])
```

**Tareas**:
- [ ] Implementar PokerTableCanvas con cartas gráficas
- [ ] Sistema de animaciones (deal, fold, chip movements)
- [ ] Gráficos estadísticos interactivos
- [ ] Hand replayer con controles play/pause/step
- [ ] Export de gráficos a PNG/PDF

---

### **FASE 4: ANÁLISIS Y ESTADÍSTICAS** (Prioridad MEDIA)

#### 4.1 Stats Calculator
```python
class PokerStats:
    """Calcula estadísticas de póker estándar"""
    
    def __init__(self, hand_history: List[Dict]):
        self.history = hand_history
        
    def calculate_vpip(self, player: str) -> float:
        """Voluntarily Put $ In Pot"""
        total = 0
        voluntary = 0
        for hand in self.history:
            if player in hand['players']:
                total += 1
                preflop = hand['phases']['preflop']
                if any(a['player'] == player and a['type'] in ['call', 'raise'] 
                       for a in preflop):
                    voluntary += 1
        return voluntary / total if total > 0 else 0.0
    
    def calculate_pfr(self, player: str) -> float:
        """Preflop Raise %"""
        # Similar implementación
        
    def calculate_aggression_factor(self, player: str) -> float:
        """(Bets + Raises) / Calls"""
        bets_raises = 0
        calls = 0
        for hand in self.history:
            for action in hand['all_actions']:
                if action['player'] == player:
                    if action['type'] in ['bet', 'raise']:
                        bets_raises += 1
                    elif action['type'] == 'call':
                        calls += 1
        return bets_raises / calls if calls > 0 else float('inf')
    
    def calculate_cbet(self, player: str) -> float:
        """Continuation Bet %"""
        # % de veces que apuesta flop tras raise preflop
        
    def calculate_3bet(self, player: str) -> float:
        """3-Bet %"""
```

#### 4.2 Range Analysis
```python
class RangeAnalyzer:
    """Analiza rangos de manos de oponentes"""
    
    def __init__(self):
        self.hand_matrix = self._build_hand_matrix()
        
    def estimate_range(self, actions: List[Action], position: str) -> np.ndarray:
        """Estima rango dado historial de acciones"""
        # Comienza con rango completo
        range_matrix = np.ones((13, 13))
        
        for action in actions:
            if action.type == 'raise' and action.phase == 'preflop':
                # Estrecha a top 15% aprox
                range_matrix = self._apply_filter(range_matrix, 'tight_raise')
            elif action.type == 'call':
                # Más amplio
                range_matrix = self._apply_filter(range_matrix, 'loose_call')
        
        return range_matrix
    
    def visualize_range(self, range_matrix):
        """Dibuja heat map de rango"""
        plt.imshow(range_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar()
        # Etiquetas: AA, AKs, AKo, etc.
```

**Tareas**:
- [ ] Implementar calculadores de stats estándar
- [ ] System de badges/tags de jugadores (TAG, LAG, Maniac)
- [ ] Análisis por posición
- [ ] Range visualizer
- [ ] Export a HUD overlay format

---

### **FASE 5: MACHINE LEARNING AVANZADO** (Prioridad MEDIA)

#### 5.1 Modelo Mejorado
```python
class PokerNetV2(nn.Module):
    """Red neuronal para decisiones de póker"""
    
    def __init__(self, input_dim=512):
        super().__init__()
        
        # Card embeddings
        self.card_embed = nn.Embedding(52, 32)
        
        # Action sequence (LSTM)
        self.action_lstm = nn.LSTM(16, 64, batch_first=True)
        
        # Main network
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Output heads
        self.action_head = nn.Linear(64, 5)  # fold, check, call, bet, raise
        self.amount_head = nn.Linear(64, 1)  # cuánto apostar (regresión)
        
    def forward(self, features):
        # ... encoding completo
        action_logits = self.action_head(x)
        bet_amount = self.amount_head(x)
        return action_logits, bet_amount
```

#### 5.2 Reinforcement Learning
```python
class PokerRLTrainer:
    """Entrena via self-play con PPO"""
    
    def __init__(self, model, env):
        self.model = model
        self.env = env  # PokerEnvironment
        self.optimizer = torch.optim.Adam(model.parameters())
        
    def train_episode(self):
        """Un episodio de self-play"""
        state = self.env.reset()
        states, actions, rewards = [], [], []
        
        while not self.env.done:
            action, log_prob = self.model.select_action(state)
            next_state, reward, done = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        # Calcular returns y actualizar
        returns = self._compute_returns(rewards)
        loss = self._ppo_loss(states, actions, returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

**Tareas**:
- [ ] Implementar arquitectura de red mejorada
- [ ] Dataset generation pipeline robusto
- [ ] Train/validation split
- [ ] Metrics: accuracy, profit expectation
- [ ] RL environment (OpenAI Gym style)
- [ ] Self-play training loop

---

### **FASE 6: DEBUGGING Y TESTING** (Prioridad ALTA)

#### 6.1 Sistema de Logging
```python
import logging
from datetime import datetime

class PokerLogger:
    """Sistema de logging estructurado"""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Handler para archivo
        fh = logging.FileHandler(
            self.log_dir / f'poker_{datetime.now():%Y%m%d_%H%M%S}.log'
        )
        fh.setLevel(logging.DEBUG)
        
        # Handler para consola
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formato
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger = logging.getLogger('PokerApp')
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.DEBUG)
    
    def log_action(self, player, action, game_state):
        """Log estructurado de acción"""
        self.logger.info(
            f"Player={player} Action={action.type} "
            f"Amount={action.amount} Phase={game_state.phase} "
            f"Pot={game_state.pot} Stack={player.stack}"
        )
```

#### 6.2 Unit Tests
```python
import unittest

class TestBettingRound(unittest.TestCase):
    def setUp(self):
        self.players = [
            PlayerProfile(name='P1', stack=1000),
            PlayerProfile(name='P2', stack=1000)
        ]
        
    def test_simple_call(self):
        """Test que call funciona correctamente"""
        round = BettingRound(self.players, bb=10)
        # P1 apuesta 10
        round.process_action(self.players[0], Action('bet', 10))
        # P2 iguala
        round.process_action(self.players[1], Action('call', 10))
        
        self.assertEqual(round.pot, 20)
        self.assertEqual(self.players[0].stack, 990)
        self.assertEqual(self.players[1].stack, 990)
        
    def test_side_pot_all_in(self):
        """Test de side pots con all-in"""
        self.players[0].stack = 100
        round = BettingRound(self.players, bb=10)
        # P1 all-in 100
        round.process_action(self.players[0], Action('all_in', 100))
        # P2 call (solo 100, no todo su stack)
        round.process_action(self.players[1], Action('call', 100))
        
        self.assertEqual(round.pot, 200)
        self.assertEqual(round.side_pots, [])  # No hay side pot (solo 2 jugadores)
```

**Tareas**:
- [ ] Implementar logger estructurado
- [ ] Tests unitarios para betting logic
- [ ] Tests para equity calculator
- [ ] Tests para feature extraction
- [ ] Integration tests (manos completas)
- [ ] Debug UI panel en GUI

---

### **FASE 7: OPTIMIZACIÓN Y RENDIMIENTO** (Prioridad BAJA)

#### 7.1 Caching y Memoization
```python
from functools import lru_cache

class EquityCache:
    """Cache de cálculos de equity"""
    
    def __init__(self):
        self.cache = {}
        
    @lru_cache(maxsize=10000)
    def get_equity(self, hero_tuple, board_tuple, num_opponents=1):
        """Memoiza cálculos costosos"""
        return estimate_equity(list(hero_tuple), list(board_tuple))
```

#### 7.2 Paralelización
```python
from multiprocessing import Pool

class ParallelSimulator:
    """Ejecuta simulaciones en paralelo"""
    
    def run_parallel(self, num_hands, num_processes=4):
        with Pool(num_processes) as pool:
            results = pool.map(self._simulate_hand, range(num_hands))
        return results
```

**Tareas**:
- [ ] Profilear código (cProfile)
- [ ] Optimizar hot paths
- [ ] Paralelizar simulaciones
- [ ] Cache de equity lookups
- [ ] Progress bars para operaciones largas

---

## 📋 ROADMAP DE IMPLEMENTACIÓN

### Sprint 1 (1-2 semanas): CORE FIXES
- ✅ Implementar BettingRound correcto
- ✅ Sistema de acciones completo
- ✅ Fases del juego separadas
- ✅ Action logging

### Sprint 2 (1 semana): FEATURES
- ✅ FeatureExtractor completo
- ✅ GameState rico
- ✅ Stats calculator básico

### Sprint 3 (2 semanas): VISUALIZACIÓN
- ✅ PokerTableCanvas
- ✅ Gráficos de análisis
- ✅ Hand replayer

### Sprint 4 (1-2 semanas): ML
- ✅ Modelo mejorado
- ✅ Dataset pipeline
- ✅ Training loop robusto

### Sprint 5 (1 semana): TESTING
- ✅ Unit tests
- ✅ Logging system
- ✅ Debug tools

### Sprint 6 (continuo): POLISH
- ✅ Optimización
- ✅ Documentación
- ✅ User testing

---

## 🎨 MEJORAS DE UX/UI ADICIONALES

### Configuración Avanzada
```python
# Panel de configuración expandido
- [ ] Presets de torneos (stack sizes, blind structure)
- [ ] Importar logs de PokerStars/PartyPoker
- [ ] Export a formato estándar (PHH - Poker Hand History)
```

### Features Adicionales
```python
- [ ] Modo torneo vs cash game
- [ ] ICM calculator
- [ ] HUD overlay (si se juega online)
- [ ] Integration con PokerTracker database
- [ ] Multi-table simulation
- [ ] GTO solver integration (opcional, PioSOLVER format)
```

---

## 📚 DOCUMENTACIÓN NECESARIA

### Code Documentation
- Docstrings en todas las funciones públicas
- Type hints completos
- Architecture diagram (mermaid/graphviz)

### User Documentation
- README.md con instalación y quick start
- Tutorial paso a paso
- Video walkthrough
- FAQ

### Developer Documentation
- Contributing guide
- API documentation (Sphinx)
- Testing guidelines

---

## 🔧 HERRAMIENTAS RECOMENDADAS

```bash
# Code quality
pip install black isort pylint mypy

# Testing
pip install pytest pytest-cov pytest-mock

# Docs
pip install sphinx sphinx-rtd-theme

# Performance
pip install line_profiler memory_profiler

# GUI (alternativas modernas)
pip install PyQt5  # o PySide6 para GUI más profesional
pip install customtkinter  # Tkinter moderno
```

---

## ✅ CHECKLIST FINAL PARA VERSIÓN 1.0

- [ ] Motor de póker 100% correcto (validado con tests)
- [ ] Todas las fases implementadas y diferenciadas
- [ ] Visualización gráfica de mesa
- [ ] Stats completos (VPIP, PFR, AF, WTSD, etc.)
- [ ] Hand replayer funcional
- [ ] ML model entrenado y funcional
- [ ] 80%+ test coverage
- [ ] Documentación completa
- [ ] Logging robusto
- [ ] Performance aceptable (>100 hands/sec)

---

## 💡 IDEAS FUTURAS (v2.0+)

- GTO solver integration
- Multi-agent tournament simulation
- Web interface (FastAPI + React)
- Mobile app
- Online multiplayer
- Blockchain integration para verificación de manos
- AR/VR poker table experience

---

**Nota Final**: Este plan es ambicioso pero estructurado. Recomiendo empezar por **FASE 1** (crítica) y avanzar secuencialmente. Cada fase es independiente pero construye sobre la anterior.

**Esfuerzo Estimado Total**: 6-10 semanas a tiempo completo para desarrollador experimentado.

**Quick Win**: Implementar PokerTableCanvas (FASE 3.1) en paralelo para tener algo visual rápidamente mientras arreglas el core.