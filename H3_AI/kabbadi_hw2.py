#     # AI Assignment- 2, Nishant Kumar (2022326)
# import math, random, itertools, argparse
# from dataclasses import dataclass
# from typing import List, Tuple, Dict, Optional
# from collections import Counter

# @dataclass(frozen=True)
# class Move:
#     dx: int
#     dy: int

# UP = Move(0,-1); DOWN = Move(0,1); LEFT = Move(-1,0); RIGHT = Move(1,0); STAY = Move(0,0)
# ALL_MOVES = (UP, DOWN, LEFT, RIGHT, STAY)

# @dataclass
# class Player:
#     team: int   
#     idx: int    
#     x: int
#     y: int
#     alive: bool = True
#     has_treasure: bool = False

#     @property
#     def pos(self): return (self.x, self.y)

#     def copy(self):
#         return Player(self.team, self.idx, self.x, self.y, self.alive, self.has_treasure)

# @dataclass
# class Treasure:
#     team: int
#     x: int
#     y: int
#     at_base: bool = True

#     def copy(self):
#         return Treasure(self.team, self.x, self.y, self.at_base)

# @dataclass
# class KabbadiState:
#     W: int; H: int
#     players: List[Player]        
#     treasures: List[Treasure]    
#     to_move: int                
#     mode: str                  
#     step: int = 0
#     max_steps: int = 120

#     def copy(self):
#         return KabbadiState(
#             self.W, self.H,
#             [p.copy() for p in self.players],
#             [t.copy() for t in self.treasures],
#             self.to_move, self.mode, self.step, self.max_steps
#         )

#     def half_split_x(self):
#         return self.W // 2

#     def in_enemy_half(self, team: int, x: int) -> bool:
#         mid = self.half_split_x()
#         return (x >= mid) if team == 0 else (x < mid)

#     def in_own_half(self, team: int, x: int) -> bool:
#         return not self.in_enemy_half(team, x)

#     def team_players(self, team: int) -> List[Player]:
#         return [p for p in self.players if p.team == team and p.alive]

#     def enemy_team(self, team: int) -> int: return 1 - team

#     def treasure_of(self, team: int) -> Treasure: return self.treasures[team]
#     def enemy_treasure(self, team: int) -> Treasure: return self.treasures[1-team]

#     def treasure_carrier(self, team: int) -> Optional[Player]:
#         for p in self.players:
#             if p.team == team and p.alive and p.has_treasure:
#                 return p
#         return None

#     def is_terminal(self) -> Optional[int]:
#         # Win if treasure carrier is in own half
#         for team in (0,1):
#             c = self.treasure_carrier(team)
#             if c is not None and self.in_own_half(team, c.x):
#                 return team
#         # Win by elimination
#         alive0 = any(p.alive for p in self.players if p.team==0)
#         alive1 = any(p.alive for p in self.players if p.team==1)
#         if alive0 and not alive1: return 0
#         if alive1 and not alive0: return 1
#         # Draw on step limit
#         if self.step >= self.max_steps: return -1
#         return None

#     def manhattan(self, a: Tuple[int,int], b: Tuple[int,int]) -> int:
#         return abs(a[0]-b[0]) + abs(a[1]-b[1])

#     def eval_static(self, team: int) -> float:
#         # Heuristic: approach enemy treasure, escort carrier home, keep ours safe, value alive players.
#         s = 0.0
#         mid = self.half_split_x()

#         def own_edge_dist(pl: Player):
#             if team==0: return max(0, mid-1 - pl.x)
#             else:       return max(0, pl.x - mid)

#         # our players
#         for p in self.team_players(team):
#             s += 0.5  # alive
#             if p.has_treasure: s += 8.0 - 0.7*own_edge_dist(p)
#             else:
#                 et = self.enemy_treasure(team)
#                 s += 5.0 - 0.4*self.manhattan(p.pos, (et.x, et.y))

#         # opponent gets negative
#         opp = 1-team
#         for p in self.team_players(opp):
#             s -= 0.5  # their alive
#             if p.has_treasure:
#                 # bad for us: they carry OUR treasure
#                 if opp==0: s -= (8.0 - 0.7*max(0, mid-1 - p.x))
#                 else:      s -= (8.0 - 0.7*max(0, p.x - mid))
#             else:
#                 t = self.treasure_of(team)
#                 s -= max(0.0, 4.0 - 0.3*self.manhattan(p.pos, (t.x, t.y)))

#         if self.treasure_of(team).at_base: s += 0.8
#         return s


# # Environment
# class KabbadiEnv:
#     def __init__(self, W=9, H=5, mode='turn', max_steps=120, seed=None):
#         assert mode in ('turn', 'simul')
#         self.W, self.H, self.mode, self.max_steps = W, H, mode, max_steps
#         self.rng = random.Random(seed)
#         self.reset()

#     def reset(self):
#         H, W = self.H, self.W
#         t0 = Treasure(team=0, x=1, y=H//2, at_base=True)
#         t1 = Treasure(team=1, x=W-2, y=H//2, at_base=True)
#         p00 = Player(0,0, 1, max(0, H//2-1))
#         p01 = Player(0,1, 1, min(H-1, H//2+1))
#         p10 = Player(1,0, W-2, max(0, H//2-1))
#         p11 = Player(1,1, W-2, min(H-1, H//2+1))
#         self.state = KabbadiState(W, H, [p00,p01,p10,p11], [t0,t1], to_move=0, mode=self.mode, step=0, max_steps=self.max_steps)
#         return self.state

#     def clone(self):
#         env = KabbadiEnv(self.W, self.H, self.mode, self.max_steps)
#         env.state = self.state.copy()
#         return env

#     def in_bounds(self, x,y): return 0 <= x < self.W and 0 <= y < self.H

#     def legal_joint_actions(self, team: int) -> List[Dict[int, Move]]:
#         pls = [p for p in self.state.players if p.team==team and p.alive]
#         if not pls: return [dict()]
#         team_idxs = [p.idx for p in pls]
#         acts = []
#         for combo in itertools.product(ALL_MOVES, repeat=len(pls)):
#             if not any(m.dx or m.dy for m in combo):  
#                 continue
#             ok = True
#             for p, mv in zip(pls, combo):
#                 nx, ny = p.x + mv.dx, p.y + mv.dy
#                 if not self.in_bounds(nx, ny): ok=False; break
#             if ok:
#                 acts.append({idx: mv for idx, mv in zip(team_idxs, combo)})
#         return acts or [dict()]

#     #internals 
#     def _apply_joint_action(self, state: KabbadiState, team: int, action: Dict[int, Move]):
#         # move only that team's alive players
#         for p in state.players:
#             if p.alive and p.team==team:
#                 mv = action.get(p.idx, STAY)
#                 nx, ny = p.x + mv.dx, p.y + mv.dy
#                 if 0<=nx<state.W and 0<=ny<state.H:
#                     p.x, p.y = nx, ny
#         # treasure pickup
#         enemy_t = state.enemy_treasure(team)
#         if enemy_t.at_base:
#             for p in state.players:
#                 if p.alive and p.team==team and (p.x,p.y)==(enemy_t.x, enemy_t.y):
#                     p.has_treasure = True; enemy_t.at_base = False; break

#     def _resolve_captures(self, state: KabbadiState):
#         # players in enemy half sharing square with opponent get captured
#         pos_map = {}
#         for i,p in enumerate(state.players):
#             if p.alive:
#                 pos_map.setdefault((p.x,p.y), []).append(i)
#         to_cap = set()
#         for xy, idxs in pos_map.items():
#             if len(idxs) < 2: continue
#             teams_present = set(state.players[i].team for i in idxs if state.players[i].alive)
#             if len(teams_present) == 2:
#                 for i in idxs:
#                     p = state.players[i]
#                     if state.in_enemy_half(p.team, p.x):
#                         to_cap.add(i)
#         for i in to_cap:
#             p = state.players[i]; p.alive=False
#             if p.has_treasure:
#                 # drop: enemy treasure returns to base
#                 t = state.treasures[1 - p.team]; t.at_base=True; p.has_treasure=False

#     def step_turn(self, action_team):
#         self._apply_joint_action(self.state, self.state.to_move, action_team)
#         self._resolve_captures(self.state)
#         self.state.step += 1; self.state.to_move = 1 - self.state.to_move

#     def step_simul(self, a0, a1):
#         st = self.state
#         # stage 1: compute intended positions by index (avoid unhashable obj)
#         intents = [(p.x, p.y) for p in st.players]
#         for i,p in enumerate(st.players):
#             if p.alive:
#                 act = a0 if p.team==0 else a1
#                 mv = act.get(p.idx, STAY)
#                 nx, ny = p.x + mv.dx, p.y + mv.dy
#                 if 0<=nx<st.W and 0<=ny<st.H:
#                     intents[i] = (nx, ny)
#         # stage 2: apply moves
#         for i,p in enumerate(st.players):
#             if p.alive:
#                 p.x, p.y = intents[i]
#         # stage 3: treasure pickup (either team may land)
#         for team in (0,1):
#             enemy_t = st.enemy_treasure(team)
#             if enemy_t.at_base:
#                 for p in st.players:
#                     if p.alive and p.team==team and (p.x,p.y)==(enemy_t.x, enemy_t.y):
#                         p.has_treasure=True; enemy_t.at_base=False; break
#         # stage 4: captures
#         self._resolve_captures(st)
#         st.step += 1; st.to_move = 1 - st.to_move

#     def current_team(self): return self.state.to_move
#     def winner(self): return self.state.is_terminal()

#     def step(self, action_team0, action_team1=None):
#         if self.mode=='turn':
#             self.step_turn(action_team0)
#         else:
#             if action_team1 is None: raise ValueError("Simultaneous mode requires two actions")
#             self.step_simul(action_team0, action_team1)

# # Agents
# class Agent:
#     def __init__(self, team: int, seed: Optional[int]=None):
#         self.team = team
#         self.rng = random.Random(seed)
#     def select_action(self, env: 'KabbadiEnv') -> Dict[int, Move]:
#         raise NotImplementedError

# class RandomAgent(Agent):
#     def select_action(self, env: 'KabbadiEnv'):
#         legal = env.legal_joint_actions(self.team)
#         return self.rng.choice(legal) if legal else {}

# def _target_for_player(st: KabbadiState, p: Player) -> Tuple[int,int]:
#     if p.has_treasure:
#         tx = 0 if p.team==0 else st.W-1
#         return (tx, p.y)
#     else:
#         et = st.enemy_treasure(p.team)
#         return (et.x, et.y)

# def _greedy_candidates(env: 'KabbadiEnv', team: int, k_per_player: int=3):
#     st = env.state; pls = [p for p in st.players if p.team==team and p.alive]
#     if not pls: return [dict()]
#     per = []
#     for p in pls:
#         tx, ty = _target_for_player(st,p)
#         scored = []
#         for mv in (UP,DOWN,LEFT,RIGHT,STAY):
#             nx, ny = p.x+mv.dx, p.y+mv.dy
#             if not env.in_bounds(nx, ny): continue
#             d = abs(nx-tx) + abs(ny-ty)
#             scored.append((d, mv))
#         scored.sort(key=lambda x:x[0])
#         keep = [mv for _,mv in scored[:k_per_player]]
#         if STAY not in keep: keep.append(STAY)
#         per.append(keep)
#     acts = []
#     for combo in itertools.product(*per):
#         if any(m.dx or m.dy for m in combo):
#             acts.append({p.idx: mv for p,mv in zip(pls, combo)})
#     return acts or env.legal_joint_actions(team)

# class GreedyAgent(Agent):
#     def __init__(self, team:int, seed:Optional[int]=None, candidate_k:int=3):
#         super().__init__(team, seed); self.k=candidate_k
#     def select_action(self, env: 'KabbadiEnv'):
#         best, bestv = None, -1e18
#         actions = _greedy_candidates(env, self.team, self.k)
#         for a in actions:
#             sim = env.clone()
#             if sim.mode=='turn':
#                 sim.step(a, None)
#                 v = sim.state.eval_static(self.team)
#             else:
#                 opp_actions = sim.legal_joint_actions(1-self.team)
#                 samples = self.rng.sample(opp_actions, k=min(3,len(opp_actions))) if opp_actions else [dict()]
#                 tot=0.0
#                 for oa in samples:
#                     sim2 = sim.clone(); sim2.step(a, oa); tot += sim2.state.eval_static(self.team)
#                 v = tot / max(1,len(samples))
#             if v>bestv: bestv=v; best=a
#         return best or {}

# # Alpha-Beta (turn mode exact, simul approximated 1-ply expectation)
# def alphabeta(env: 'KabbadiEnv', team:int, depth:int, alpha:float, beta:float):
#     w = env.winner()
#     if w is not None:
#         return (1e6 if w==team else (0.0 if w==-1 else -1e6), {})
#     if depth==0: return (env.state.eval_static(team), {})
#     cur = env.current_team()
#     actions = _greedy_candidates(env, cur, k_per_player=3)
#     if cur==team:   # max
#         val, best = -1e18, actions[0] if actions else {}
#         for a in actions:
#             sim = env.clone(); sim.step(a, None)
#             v,_ = alphabeta(sim, team, depth-1, alpha, beta)
#             if v>val: val, best = v, a
#             alpha = max(alpha, val)
#             if beta<=alpha: break
#         return (val, best)
#     else:           # min
#         val, best = 1e18, actions[0] if actions else {}
#         for a in actions:
#             sim = env.clone(); sim.step(a, None)
#             v,_ = alphabeta(sim, team, depth-1, alpha, beta)
#             if v<val: val, best = v, a
#             beta = min(beta, val)
#             if beta<=alpha: break
#         return (val, best)

# class AlphaBetaAgent(Agent):
#     def __init__(self, team:int, depth:int=3, seed:Optional[int]=None):
#         super().__init__(team, seed); self.depth=depth
#     def select_action(self, env: 'KabbadiEnv'):
#         if env.mode=='turn':
#             _, act = alphabeta(env, self.team, self.depth, -1e18, 1e18)
#             return act or {}
#         else:
#             # 1-ply expectation vs sampled opponent
#             actions = _greedy_candidates(env, self.team, 3)
#             opp = env.legal_joint_actions(1-self.team)
#             opp_samples = self.rng.sample(opp, k=min(5,len(opp))) if opp else [dict()]
#             best, bestv = None, -1e18
#             for a in actions:
#                 tot=0.0
#                 for oa in opp_samples:
#                     sim = env.clone(); sim.step(a, oa); tot += sim.state.eval_static(self.team)
#                 v = tot/max(1,len(opp_samples))
#                 if v>bestv: bestv=v; best=a
#             return best or {}

# # MCTS (UCT) for either mode
# class MCTSNode:
#     def __init__(self, env: 'KabbadiEnv', team_perspective:int, parent=None, action_taken=None):
#         self.env = env; self.team_perspective = team_perspective
#         self.parent = parent; self.action_taken = action_taken
#         self.children = []; self.N=0; self.W=0.0
#         self.untried = None

#     def player_to_move(self): return self.env.current_team()

#     def legal_actions(self):
#         return self.env.legal_joint_actions(self.player_to_move())

#     def expand(self):
#         if self.untried is None:
#             self.untried = self.legal_actions()
#         if not self.untried: return self
#         a = self.untried.pop()
#         child_env = self.env.clone()
#         if child_env.mode=='turn':
#             child_env.step(a, None)
#         else:
#             # sample opponent during expansion
#             opp_actions = child_env.legal_joint_actions(1-child_env.current_team())
#             oa = random.choice(opp_actions) if opp_actions else {}
#             child_env.step(a, oa)
#         ch = MCTSNode(child_env, self.team_perspective, parent=self, action_taken=a)
#         self.children.append(ch)
#         return ch

#     def best_child(self, c=1.414):
#         best, bestv = None, -1e18
#         for ch in self.children:
#             if ch.N==0: u = 1e18
#             else:
#                 u = (ch.W/ch.N) + c*math.sqrt(math.log(self.N+1)/ch.N)
#             if u>bestv: bestv=u; best=ch
#         return best

# def rollout(env: 'KabbadiEnv', team:int, depth:int=20) -> float:
#     for _ in range(depth):
#         w = env.winner()
#         if w is not None:
#             if w==team: return 1.0
#             if w==-1:   return 0.5
#             return 0.0
#         if env.mode=='turn':
#             a = random.choice(env.legal_joint_actions(env.current_team()))
#             env.step(a, None)
#         else:
#             a0 = random.choice(env.legal_joint_actions(0))
#             a1 = random.choice(env.legal_joint_actions(1))
#             env.step(a0, a1)
#     # heuristic fallback
#     v = env.state.eval_static(team)
#     return 1/(1+math.exp(-v/8.0))

# def mcts_policy(env: 'KabbadiEnv', team:int, iters:int=100, depth:int=20):
#     root = MCTSNode(env.clone(), team)
#     root.untried = root.legal_actions()
#     for _ in range(iters):
#         n = root
#         # selection
#         while n.untried is not None and len(n.untried)==0 and n.children:
#             n = n.best_child()
#         # expansion
#         if n.untried is None or len(n.untried)>0:
#             n = n.expand()
#         # simulation
#         sim = n.env.clone()
#         r = rollout(sim, team, depth)
#         # backprop
#         while n is not None:
#             n.N += 1; n.W += r; n = n.parent
#     if not root.children:
#         leg = root.legal_actions()
#         return random.choice(leg) if leg else {}
#     best = max(root.children, key=lambda ch: ch.N)
#     return best.action_taken or {}

# class MCTSAgent(Agent):
#     def __init__(self, team:int, iters:int=80, depth:int=20, seed:Optional[int]=None):
#         super().__init__(team, seed); self.iters=iters; self.depth=depth
#     def select_action(self, env: 'KabbadiEnv'):
#         random.seed(self.rng.randint(0,10**9))
#         return mcts_policy(env, self.team, self.iters, self.depth)

# # Running matches
# def play_game(env: 'KabbadiEnv', a0:Agent, a1:Agent, seed:Optional[int]=None, verbose:bool=False) -> int:
#     if seed is not None: random.seed(seed)
#     env.reset()
#     while True:
#         w = env.winner()
#         if w is not None: return w
#         if env.mode=='turn':
#             team = env.current_team()
#             act = (a0 if team==0 else a1).select_action(env)
#             env.step(act, None)
#         else:
#             env.step(a0.select_action(env), a1.select_action(env))

# def run_series(mode:str, ctorA, ctorB, nameA:str, nameB:str, episodes:int, env_kwargs=None, A_kwargs=None, B_kwargs=None, seed:int=7):
#     env_kwargs = env_kwargs or {}
#     A_kwargs = A_kwargs or {}; B_kwargs = B_kwargs or {}
#     wins = Counter()
#     for i in range(episodes):
#         env = KabbadiEnv(mode=mode, **env_kwargs)
#         if i%2==0:
#             a0 = ctorA(0, **A_kwargs); a1 = ctorB(1, **B_kwargs)
#             w = play_game(env, a0, a1, seed=seed+i)
#             if   w==0: wins[nameA]+=1
#             elif w==1: wins[nameB]+=1
#             else: wins['Draw']+=1
#         else:
#             a0 = ctorB(0, **B_kwargs); a1 = ctorA(1, **A_kwargs)
#             w = play_game(env, a0, a1, seed=seed+i)
#             if   w==0: wins[nameB]+=1
#             elif w==1: wins[nameA]+=1
#             else: wins['Draw']+=1
#     return {'Mode': mode, 'Matchup': f'{nameA} vs {nameB}', nameA:wins.get(nameA,0), nameB:wins.get(nameB,0), 'Draw':wins.get('Draw',0), 'Episodes': sum(wins.values())}

# def print_table(rows, columns):
#     """Pretty-print a list of dict rows with given column order."""
#     # compute column widths
#     widths = []
#     for c in columns:
#         w = max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows))
#         widths.append(w)
#     # header
#     header = " | ".join(str(c).ljust(w) for c, w in zip(columns, widths))
#     bar = "-+-".join("-" * w for w in widths)
#     print(header)
#     print(bar)
#     # rows
#     for r in rows:
#         line = " | ".join(str(r.get(c, "")).ljust(w) for c, w in zip(columns, widths))
#         print(line)


# def run_pair_as_row(
#     mode: str,
#     nameA: str, nameB: str,
#     ctorA, ctorB,
#     episodes: int,
#     env_kwargs=None,
#     A_kwargs=None, B_kwargs=None,
#     seed: int = 7,
# ):
#     """Wrap run_series to return a uniform row dict for pretty tables + winner/score."""
#     env_kwargs = env_kwargs or {}
#     A_kwargs = A_kwargs or {}
#     B_kwargs = B_kwargs or {}
#     res = run_series(
#         mode, ctorA, ctorB, nameA, nameB,
#         episodes=episodes,
#         env_kwargs=env_kwargs,
#         A_kwargs=A_kwargs,
#         B_kwargs=B_kwargs,
#         seed=seed
#     )
#     row = {
#         "Mode": mode.upper(),
#         "A": nameA,
#         "B": nameB,
#         "A_wins": res.get(nameA, 0),
#         "B_wins": res.get(nameB, 0),
#         "Draw": res.get("Draw", 0),
#         "Episodes": res.get("Episodes", episodes)
#     }
#     aw, bw, dr = row["A_wins"], row["B_wins"], row["Draw"]
#     row["Score"] = f"{aw}-{bw}-{dr}"
#     row["Winner"] = (row["A"] if aw > bw else (row["B"] if bw > aw else "Draw"))
#     return row
# AGENTS = {
#     'random': RandomAgent,
#     'greedy': GreedyAgent,
#     'ab': AlphaBetaAgent,
#     'mcts': MCTSAgent,
# }

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--mode', choices=['turn','simul'], default=None,
#                     help="Non-interactive single-mode run (leave empty to use --interactive).")
#     ap.add_argument('--episodes', type=int, default=20)
#     ap.add_argument('--max_steps', type=int, default=120)
#     ap.add_argument('--ab_depth', type=int, default=3)
#     ap.add_argument('--mcts_iters', type=int, default=120)
#     ap.add_argument('--mcts_depth', type=int, default=25)
#     ap.add_argument('--W', type=int, default=9)
#     ap.add_argument('--H', type=int, default=5)
#     ap.add_argument('--interactive', action='store_true',
#                     help="Use prompts + pretty tournament tables.")
#     args = ap.parse_args()

#     def _with_score_winner(row):
#         aw = int(row.get("A_wins", 0))
#         bw = int(row.get("B_wins", 0))
#         dr = int(row.get("Draw", 0))
#         row.setdefault("Score", f"{aw}-{bw}-{dr}")
#         row.setdefault("Winner", row["A"] if aw > bw else (row["B"] if bw > aw else "Draw"))
#         return row

#     # Iinteractive
#     if args.interactive:
#         print("=== Kabbadi AI — Interactive Setup ===")

#         def ask_int(prompt, default):
#             raw = input(f"{prompt} [{default}]: ").strip()
#             return int(raw) if raw else default

#         def ask_bool(prompt, default=False):
#             raw = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
#             if not raw:
#                 return default
#             return raw in ('y', 'yes')

#         W = ask_int("Enter grid width W", args.W)
#         H = ask_int("Enter grid height H", args.H)
#         max_steps = ask_int("Enter max number of game steps per episode", args.max_steps)
#         episodes = ask_int("Enter episodes per pairing (recommend 20/50/100)", args.episodes)

#         ab_depth = ask_int("Alpha-Beta search depth", args.ab_depth)
#         mcts_iters = ask_int("MCTS iterations per move", args.mcts_iters)
#         mcts_depth = ask_int("MCTS rollout depth", args.mcts_depth)

#         include_ab_in_sim = ask_bool("Include Alpha-Beta in SIMULTANEOUS mode?", default=False)

#         AGENT_LIST = [
#             ("Random",    RandomAgent,    {}),
#             ("Greedy",    GreedyAgent,    {"candidate_k": 3}),
#             ("AlphaBeta", AlphaBetaAgent, {"depth": ab_depth}),
#             ("MCTS",      MCTSAgent,      {"iters": mcts_iters, "depth": mcts_depth}),
#         ]

#         def env_kwargs(mode):
#             return {"W": W, "H": H, "max_steps": max_steps}

#         # TURN
#         print("\n=== TURN-BASED MATCHUPS ===")
#         rows_turn = []
#         for nameA, ctorA, kwargsA in AGENT_LIST:
#             for nameB, ctorB, kwargsB in AGENT_LIST:
#                 if nameA == nameB:
#                     continue
#                 print(f"Running TURN:  {nameA} vs {nameB} ...", flush=True)
#                 row = run_pair_as_row(
#                     mode="turn",
#                     nameA=nameA, nameB=nameB,
#                     ctorA=ctorA, ctorB=ctorB,
#                     episodes=episodes,
#                     env_kwargs=env_kwargs("turn"),
#                     A_kwargs=kwargsA, B_kwargs=kwargsB,
#                 )
#                 rows_turn.append(_with_score_winner(row))

#         print_table(rows_turn, ["Mode","A","B","A_wins","B_wins","Draw","Episodes","Score","Winner"])
        
#         # SIMUL (AlphaBeta optional)
#         print("\n=== SIMULTANEOUS MATCHUPS ({} AlphaBeta) ===".format(
#             "INCLUDING" if include_ab_in_sim else "EXCEPT"
#         ))
#         sim_list = AGENT_LIST if include_ab_in_sim else [t for t in AGENT_LIST if t[0] != "AlphaBeta"]

#         rows_sim = []
#         for nameA, ctorA, kwargsA in sim_list:
#             for nameB, ctorB, kwargsB in sim_list:
#                 if nameA == nameB:
#                     continue
#                 print(f"Running SIMUL: {nameA} vs {nameB} ...", flush=True)
#                 row = run_pair_as_row(
#                     mode="simul",
#                     nameA=nameA, nameB=nameB,
#                     ctorA=ctorA, ctorB=ctorB,
#                     episodes=episodes,
#                     env_kwargs=env_kwargs("simul"),
#                     A_kwargs=kwargsA, B_kwargs=kwargsB,
#                 )
#                 rows_sim.append(_with_score_winner(row))

#         print_table(rows_sim, ["Mode","A","B","A_wins","B_wins","Draw","Episodes","Score","Winner"])
#         print("\nDone.")
#         return

#     # NON-INTERACTIVE 
#     if args.mode is None:
#         print("No --mode provided. Tip: run with --interactive for the full tournament UI,")
#         print("or specify --mode turn|simul to run the default three pairings.")
#         return

#     env_kwargs = {'W': args.W, 'H': args.H, 'max_steps': args.max_steps}
#     results = []
#     results.append(run_series(args.mode, GreedyAgent, RandomAgent, 'greedy','random',
#                               args.episodes, env_kwargs, {'candidate_k':3}, {}))
#     results.append(run_series(args.mode, AlphaBetaAgent, GreedyAgent, 'ab','greedy',
#                               args.episodes, env_kwargs, {'depth':args.ab_depth}, {'candidate_k':3}))
#     results.append(run_series(args.mode, MCTSAgent, GreedyAgent, 'mcts','greedy',
#                               args.episodes, env_kwargs, {'iters':args.mcts_iters, 'depth':args.mcts_depth}, {'candidate_k':3}))

#     # shape rows + computed winner/score
#     rows = []
#     for r in results:
#         nameA, nameB = r['Matchup'].split(' vs ')
#         row = {
#             "Mode": r["Mode"].upper(),
#             "A": nameA, "B": nameB,
#             "A_wins": r.get(nameA,0),
#             "B_wins": r.get(nameB,0),
#             "Draw": r.get("Draw",0),
#             "Episodes": r.get("Episodes", args.episodes),
#         }
#         rows.append(_with_score_winner(row))

#     print(f"=== Results ({args.mode}, episodes={args.episodes}) ===")
#     print_table(rows, ["Mode","A","B","A_wins","B_wins","Draw","Episodes","Score","Winner"])
#     print("\nSummaries:")
#     for r in rows:
#         print(f"[{r['Mode']}] {r['A']} vs {r['B']} → {r['Winner']} (score {r['Score']}, episodes {r['Episodes']})")

# if __name__ == "__main__":
#     main()














# AI Assignment- 2, Nishant Kumar (2022326)
import math, random, itertools
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import Counter

@dataclass(frozen=True)
class Move:
    dx: int
    dy: int

UP = Move(0, -1)
DOWN = Move(0, 1)
LEFT = Move(-1, 0)
RIGHT = Move(1, 0)
STAY = Move(0, 0)
ALL_MOVES = (UP, DOWN, LEFT, RIGHT, STAY)

@dataclass
class Player:
    team: int
    idx: int
    x: int
    y: int
    alive: bool = True
    has_treasure: bool = False

    @property
    def pos(self):
        return (self.x, self.y)

    def copy(self):
        return Player(self.team, self.idx, self.x, self.y, self.alive, self.has_treasure)

@dataclass
class Treasure:
    team: int
    x: int
    y: int
    at_base: bool = True

    def copy(self):
        return Treasure(self.team, self.x, self.y, self.at_base)

@dataclass
class KabbadiState:
    W: int
    H: int
    players: List[Player]
    treasures: List[Treasure]
    to_move: int
    mode: str
    step: int = 0
    max_steps: int = 120

    def copy(self):
        return KabbadiState(
            self.W,
            self.H,
            [p.copy() for p in self.players],
            [t.copy() for t in self.treasures],
            self.to_move,
            self.mode,
            self.step,
            self.max_steps,
        )

    def half_split_x(self):
        return self.W // 2

    def in_enemy_half(self, team: int, x: int) -> bool:
        mid = self.half_split_x()
        return (x >= mid) if team == 0 else (x < mid)

    def in_own_half(self, team: int, x: int) -> bool:
        return not self.in_enemy_half(team, x)

    def team_players(self, team: int) -> List[Player]:
        return [p for p in self.players if p.team == team and p.alive]

    def enemy_team(self, team: int) -> int:
        return 1 - team

    def treasure_of(self, team: int) -> Treasure:
        return self.treasures[team]

    def enemy_treasure(self, team: int) -> Treasure:
        return self.treasures[1 - team]

    def treasure_carrier(self, team: int) -> Optional[Player]:
        for p in self.players:
            if p.team == team and p.alive and p.has_treasure:
                return p
        return None

    def is_terminal(self) -> Optional[int]:
        # Win if treasure carrier is in own half
        for team in (0, 1):
            c = self.treasure_carrier(team)
            if c is not None and self.in_own_half(team, c.x):
                return team
        # Win by elimination
        alive0 = any(p.alive for p in self.players if p.team == 0)
        alive1 = any(p.alive for p in self.players if p.team == 1)
        if alive0 and not alive1:
            return 0
        if alive1 and not alive0:
            return 1
        # Draw on step limit
        if self.step >= self.max_steps:
            return -1
        return None

    def manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def eval_static(self, team: int) -> float:
        # Heuristic: approach enemy treasure, escort carrier home, keep ours safe, value alive players.
        s = 0.0
        mid = self.half_split_x()

        def own_edge_dist(pl: Player):
            if team == 0:
                return max(0, mid - 1 - pl.x)
            else:
                return max(0, pl.x - mid)

        # our players
        for p in self.team_players(team):
            s += 0.5  # alive
            if p.has_treasure:
                s += 8.0 - 0.7 * own_edge_dist(p)
            else:
                et = self.enemy_treasure(team)
                s += 5.0 - 0.4 * self.manhattan(p.pos, (et.x, et.y))

        # opponent gets negative
        opp = 1 - team
        for p in self.team_players(opp):
            s -= 0.5  # their alive
            if p.has_treasure:
                # bad for us: they carry OUR treasure
                if opp == 0:
                    s -= 8.0 - 0.7 * max(0, mid - 1 - p.x)
                else:
                    s -= 8.0 - 0.7 * max(0, p.x - mid)
            else:
                t = self.treasure_of(team)
                s -= max(0.0, 4.0 - 0.3 * self.manhattan(p.pos, (t.x, t.y)))

        if self.treasure_of(team).at_base:
            s += 0.8
        return s


# Environment
class KabbadiEnv:
    def __init__(self, W=9, H=5, mode="turn", max_steps=120, seed=None):
        assert mode in ("turn", "simul")
        self.W, self.H, self.mode, self.max_steps = W, H, mode, max_steps
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        H, W = self.H, self.W
        t0 = Treasure(team=0, x=1, y=H // 2, at_base=True)
        t1 = Treasure(team=1, x=W - 2, y=H // 2, at_base=True)
        p00 = Player(0, 0, 1, max(0, H // 2 - 1))
        p01 = Player(0, 1, 1, min(H - 1, H // 2 + 1))
        p10 = Player(1, 0, W - 2, max(0, H // 2 - 1))
        p11 = Player(1, 1, W - 2, min(H - 1, H // 2 + 1))
        self.state = KabbadiState(
            W,
            H,
            [p00, p01, p10, p11],
            [t0, t1],
            to_move=0,
            mode=self.mode,
            step=0,
            max_steps=self.max_steps,
        )
        return self.state

    def clone(self):
        env = KabbadiEnv(self.W, self.H, self.mode, self.max_steps)
        env.state = self.state.copy()
        return env

    def in_bounds(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H

    def legal_joint_actions(self, team: int) -> List[Dict[int, Move]]:
        pls = [p for p in self.state.players if p.team == team and p.alive]
        if not pls:
            return [dict()]
        team_idxs = [p.idx for p in pls]
        acts = []
        for combo in itertools.product(ALL_MOVES, repeat=len(pls)):
            if not any(m.dx or m.dy for m in combo):
                continue
            ok = True
            for p, mv in zip(pls, combo):
                nx, ny = p.x + mv.dx, p.y + mv.dy
                if not self.in_bounds(nx, ny):
                    ok = False
                    break
            if ok:
                acts.append({idx: mv for idx, mv in zip(team_idxs, combo)})
        return acts or [dict()]

    # internals
    def _apply_joint_action(self, state: KabbadiState, team: int, action: Dict[int, Move]):
        # move only that team's alive players
        for p in state.players:
            if p.alive and p.team == team:
                mv = action.get(p.idx, STAY)
                nx, ny = p.x + mv.dx, p.y + mv.dy
                if 0 <= nx < state.W and 0 <= ny < state.H:
                    p.x, p.y = nx, ny
        # treasure pickup
        enemy_t = state.enemy_treasure(team)
        if enemy_t.at_base:
            for p in state.players:
                if p.alive and p.team == team and (p.x, p.y) == (enemy_t.x, enemy_t.y):
                    p.has_treasure = True
                    enemy_t.at_base = False
                    break

    def _resolve_captures(self, state: KabbadiState):
        # players in enemy half sharing square with opponent get captured
        pos_map: Dict[Tuple[int, int], List[int]] = {}
        for i, p in enumerate(state.players):
            if p.alive:
                pos_map.setdefault((p.x, p.y), []).append(i)
        to_cap = set()
        for xy, idxs in pos_map.items():
            if len(idxs) < 2:
                continue
            teams_present = set(state.players[i].team for i in idxs if state.players[i].alive)
            if len(teams_present) == 2:
                for i in idxs:
                    p = state.players[i]
                    if state.in_enemy_half(p.team, p.x):
                        to_cap.add(i)
        for i in to_cap:
            p = state.players[i]
            p.alive = False
            if p.has_treasure:
                # drop: enemy treasure returns to base
                t = state.treasures[1 - p.team]
                t.at_base = True
                p.has_treasure = False

    def step_turn(self, action_team):
        self._apply_joint_action(self.state, self.state.to_move, action_team)
        self._resolve_captures(self.state)
        self.state.step += 1
        self.state.to_move = 1 - self.state.to_move

    def step_simul(self, a0, a1):
        st = self.state
        # stage 1: compute intended positions by index (avoid unhashable obj)
        intents = [(p.x, p.y) for p in st.players]
        for i, p in enumerate(st.players):
            if p.alive:
                act = a0 if p.team == 0 else a1
                mv = act.get(p.idx, STAY)
                nx, ny = p.x + mv.dx, p.y + mv.dy
                if 0 <= nx < st.W and 0 <= ny < st.H:
                    intents[i] = (nx, ny)
        # stage 2: apply moves
        for i, p in enumerate(st.players):
            if p.alive:
                p.x, p.y = intents[i]
        # stage 3: treasure pickup (either team may land)
        for team in (0, 1):
            enemy_t = st.enemy_treasure(team)
            if enemy_t.at_base:
                for p in st.players:
                    if p.alive and p.team == team and (p.x, p.y) == (enemy_t.x, enemy_t.y):
                        p.has_treasure = True
                        enemy_t.at_base = False
                        break
        # stage 4: captures
        self._resolve_captures(st)
        st.step += 1
        st.to_move = 1 - st.to_move

    def current_team(self):
        return self.state.to_move

    def winner(self):
        return self.state.is_terminal()

    def step(self, action_team0, action_team1=None):
        if self.mode == "turn":
            self.step_turn(action_team0)
        else:
            if action_team1 is None:
                raise ValueError("Simultaneous mode requires two actions")
            self.step_simul(action_team0, action_team1)


# Agents
class Agent:
    def __init__(self, team: int, seed: Optional[int] = None):
        self.team = team
        self.rng = random.Random(seed)

    def select_action(self, env: "KabbadiEnv") -> Dict[int, Move]:
        raise NotImplementedError


class RandomAgent(Agent):
    def select_action(self, env: "KabbadiEnv"):
        legal = env.legal_joint_actions(self.team)
        return self.rng.choice(legal) if legal else {}


def _target_for_player(st: KabbadiState, p: Player) -> Tuple[int, int]:
    if p.has_treasure:
        tx = 0 if p.team == 0 else st.W - 1
        return (tx, p.y)
    else:
        et = st.enemy_treasure(p.team)
        return (et.x, et.y)


def _greedy_candidates(env: "KabbadiEnv", team: int, k_per_player: int = 3):
    st = env.state
    pls = [p for p in st.players if p.team == team and p.alive]
    if not pls:
        return [dict()]
    per = []
    for p in pls:
        tx, ty = _target_for_player(st, p)
        scored = []
        for mv in (UP, DOWN, LEFT, RIGHT, STAY):
            nx, ny = p.x + mv.dx, p.y + mv.dy
            if not env.in_bounds(nx, ny):
                continue
            d = abs(nx - tx) + abs(ny - ty)
            scored.append((d, mv))
        scored.sort(key=lambda x: x[0])
        keep = [mv for _, mv in scored[:k_per_player]]
        if STAY not in keep:
            keep.append(STAY)
        per.append(keep)
    acts = []
    for combo in itertools.product(*per):
        if any(m.dx or m.dy for m in combo):
            acts.append({p.idx: mv for p, mv in zip(pls, combo)})
    return acts or env.legal_joint_actions(team)


class GreedyAgent(Agent):
    def __init__(self, team: int, seed: Optional[int] = None, candidate_k: int = 3):
        super().__init__(team, seed)
        self.k = candidate_k

    def select_action(self, env: "KabbadiEnv"):
        best, bestv = None, -1e18
        actions = _greedy_candidates(env, self.team, self.k)
        for a in actions:
            sim = env.clone()
            if sim.mode == "turn":
                sim.step(a, None)
                v = sim.state.eval_static(self.team)
            else:
                opp_actions = sim.legal_joint_actions(1 - self.team)
                samples = (
                    self.rng.sample(opp_actions, k=min(3, len(opp_actions)))
                    if opp_actions
                    else [dict()]
                )
                tot = 0.0
                for oa in samples:
                    sim2 = sim.clone()
                    sim2.step(a, oa)
                    tot += sim2.state.eval_static(self.team)
                v = tot / max(1, len(samples))
            if v > bestv:
                bestv = v
                best = a
        return best or {}


# Alpha-Beta (turn mode exact, simul approximated 1-ply expectation)
def alphabeta(env: "KabbadiEnv", team: int, depth: int, alpha: float, beta: float):
    w = env.winner()
    if w is not None:
        return (1e6 if w == team else (0.0 if w == -1 else -1e6), {})
    if depth == 0:
        return (env.state.eval_static(team), {})
    cur = env.current_team()
    actions = _greedy_candidates(env, cur, k_per_player=3)
    if cur == team:  # max
        val, best = -1e18, actions[0] if actions else {}
        for a in actions:
            sim = env.clone()
            sim.step(a, None)
            v, _ = alphabeta(sim, team, depth - 1, alpha, beta)
            if v > val:
                val, best = v, a
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return (val, best)
    else:  # min
        val, best = 1e18, actions[0] if actions else {}
        for a in actions:
            sim = env.clone()
            sim.step(a, None)
            v, _ = alphabeta(sim, team, depth - 1, alpha, beta)
            if v < val:
                val, best = v, a
            beta = min(beta, val)
            if beta <= alpha:
                break
        return (val, best)


class AlphaBetaAgent(Agent):
    def __init__(self, team: int, depth: int = 3, seed: Optional[int] = None):
        super().__init__(team, seed)
        self.depth = depth

    def select_action(self, env: "KabbadiEnv"):
        if env.mode == "turn":
            _, act = alphabeta(env, self.team, self.depth, -1e18, 1e18)
            return act or {}
        else:
            # 1-ply expectation vs sampled opponent
            actions = _greedy_candidates(env, self.team, 3)
            opp = env.legal_joint_actions(1 - self.team)
            opp_samples = (
                self.rng.sample(opp, k=min(5, len(opp))) if opp else [dict()]
            )
            best, bestv = None, -1e18
            for a in actions:
                tot = 0.0
                for oa in opp_samples:
                    sim = env.clone()
                    sim.step(a, oa)
                    tot += sim.state.eval_static(self.team)
                v = tot / max(1, len(opp_samples))
                if v > bestv:
                    bestv = v
                    best = a
            return best or {}


# MCTS (UCT) for either mode
class MCTSNode:
    def __init__(self, env: "KabbadiEnv", team_perspective: int, parent=None, action_taken=None):
        self.env = env
        self.team_perspective = team_perspective
        self.parent = parent
        self.action_taken = action_taken
        self.children: List["MCTSNode"] = []
        self.N = 0
        self.W = 0.0
        self.untried: Optional[List[Dict[int, Move]]] = None

    def player_to_move(self):
        return self.env.current_team()

    def legal_actions(self):
        return self.env.legal_joint_actions(self.player_to_move())

    def expand(self):
        if self.untried is None:
            self.untried = self.legal_actions()
        if not self.untried:
            return self
        a = self.untried.pop()
        child_env = self.env.clone()
        if child_env.mode == "turn":
            child_env.step(a, None)
        else:
            # sample opponent during expansion
            opp_actions = child_env.legal_joint_actions(1 - child_env.current_team())
            oa = random.choice(opp_actions) if opp_actions else {}
            child_env.step(a, oa)
        ch = MCTSNode(child_env, self.team_perspective, parent=self, action_taken=a)
        self.children.append(ch)
        return ch

    def best_child(self, c=1.414):
        best, bestv = None, -1e18
        for ch in self.children:
            if ch.N == 0:
                u = 1e18
            else:
                u = (ch.W / ch.N) + c * math.sqrt(math.log(self.N + 1) / ch.N)
            if u > bestv:
                bestv = u
                best = ch
        return best


def rollout(env: "KabbadiEnv", team: int, depth: int = 20) -> float:
    for _ in range(depth):
        w = env.winner()
        if w is not None:
            if w == team:
                return 1.0
            if w == -1:
                return 0.5
            return 0.0
        if env.mode == "turn":
            a = random.choice(env.legal_joint_actions(env.current_team()))
            env.step(a, None)
        else:
            a0 = random.choice(env.legal_joint_actions(0))
            a1 = random.choice(env.legal_joint_actions(1))
            env.step(a0, a1)
    # heuristic fallback
    v = env.state.eval_static(team)
    return 1 / (1 + math.exp(-v / 8.0))


def mcts_policy(env: "KabbadiEnv", team: int, iters: int = 100, depth: int = 20):
    root = MCTSNode(env.clone(), team)
    root.untried = root.legal_actions()
    for _ in range(iters):
        n = root
        # selection
        while n.untried is not None and len(n.untried) == 0 and n.children:
            n = n.best_child()
        # expansion
        if n.untried is None or len(n.untried) > 0:
            n = n.expand()
        # simulation
        sim = n.env.clone()
        r = rollout(sim, team, depth)
        # backprop
        while n is not None:
            n.N += 1
            n.W += r
            n = n.parent
    if not root.children:
        leg = root.legal_actions()
        return random.choice(leg) if leg else {}
    best = max(root.children, key=lambda ch: ch.N)
    return best.action_taken or {}


class MCTSAgent(Agent):
    def __init__(self, team: int, iters: int = 80, depth: int = 20, seed: Optional[int] = None):
        super().__init__(team, seed)
        self.iters = iters
        self.depth = depth

    def select_action(self, env: "KabbadiEnv"):
        random.seed(self.rng.randint(0, 10 ** 9))
        return mcts_policy(env, self.team, self.iters, self.depth)


# Running matches
def play_game(env: "KabbadiEnv", a0: Agent, a1: Agent, seed: Optional[int] = None, verbose: bool = False) -> int:
    if seed is not None:
        random.seed(seed)
    env.reset()
    while True:
        w = env.winner()
        if w is not None:
            return w
        if env.mode == "turn":
            team = env.current_team()
            act = (a0 if team == 0 else a1).select_action(env)
            env.step(act, None)
        else:
            env.step(a0.select_action(env), a1.select_action(env))


def run_series(
    mode: str,
    ctorA,
    ctorB,
    nameA: str,
    nameB: str,
    episodes: int,
    env_kwargs=None,
    A_kwargs=None,
    B_kwargs=None,
    seed: int = 7,
):
    env_kwargs = env_kwargs or {}
    A_kwargs = A_kwargs or {}
    B_kwargs = B_kwargs or {}
    wins = Counter()
    for i in range(episodes):
        env = KabbadiEnv(mode=mode, **env_kwargs)
        if i % 2 == 0:
            a0 = ctorA(0, **A_kwargs)
            a1 = ctorB(1, **B_kwargs)
            w = play_game(env, a0, a1, seed=seed + i)
            if w == 0:
                wins[nameA] += 1
            elif w == 1:
                wins[nameB] += 1
            else:
                wins["Draw"] += 1
        else:
            a0 = ctorB(0, **B_kwargs)
            a1 = ctorA(1, **A_kwargs)
            w = play_game(env, a0, a1, seed=seed + i)
            if w == 0:
                wins[nameB] += 1
            elif w == 1:
                wins[nameA] += 1
            else:
                wins["Draw"] += 1
    return {
        "Mode": mode,
        "Matchup": f"{nameA} vs {nameB}",
        nameA: wins.get(nameA, 0),
        nameB: wins.get(nameB, 0),
        "Draw": wins.get("Draw", 0),
        "Episodes": sum(wins.values()),
    }


def print_table(rows, columns):
    """Pretty-print a list of dict rows with given column order."""
    # compute column widths
    widths = []
    for c in columns:
        w = max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows))
        widths.append(w)
    # header
    header = " | ".join(str(c).ljust(w) for c, w in zip(columns, widths))
    bar = "-+-".join("-" * w for w in widths)
    print(header)
    print(bar)
    # rows
    for r in rows:
        line = " | ".join(str(r.get(c, "")).ljust(w) for c, w in zip(columns, widths))
        print(line)


def run_pair_as_row(
    mode: str,
    nameA: str,
    nameB: str,
    ctorA,
    ctorB,
    episodes: int,
    env_kwargs=None,
    A_kwargs=None,
    B_kwargs=None,
    seed: int = 7,
):
    """Wrap run_series to return a uniform row dict for pretty tables + winner/score."""
    env_kwargs = env_kwargs or {}
    A_kwargs = A_kwargs or {}
    B_kwargs = B_kwargs or {}
    res = run_series(
        mode,
        ctorA,
        ctorB,
        nameA,
        nameB,
        episodes=episodes,
        env_kwargs=env_kwargs,
        A_kwargs=A_kwargs,
        B_kwargs=B_kwargs,
        seed=seed,
    )
    row = {
        "Mode": mode.upper(),
        "A": nameA,
        "B": nameB,
        "A_wins": res.get(nameA, 0),
        "B_wins": res.get(nameB, 0),
        "Draw": res.get("Draw", 0),
        "Episodes": res.get("Episodes", episodes),
    }
    aw, bw, dr = row["A_wins"], row["B_wins"], row["Draw"]
    row["Score"] = f"{aw}-{bw}-{dr}"
    row["Winner"] = row["A"] if aw > bw else (row["B"] if bw > aw else "Draw")
    return row


AGENTS = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "ab": AlphaBetaAgent,
    "mcts": MCTSAgent,
}


def main():
    """
    Instance 1:
    Turn-based Kabbadi (mode='turn'), GreedyAgent vs RandomAgent.
    """
    mode = "turn"
    episodes = 50
    env_kwargs = {"W": 9, "H": 5, "max_steps": 120}

    row = run_pair_as_row(
        mode=mode,
        nameA="Greedy",
        nameB="Random",
        ctorA=GreedyAgent,
        ctorB=RandomAgent,
        episodes=episodes,
        env_kwargs=env_kwargs,
        A_kwargs={"candidate_k": 3},
        B_kwargs={},
        seed=7,
    )

    print("=== Instance 1: Turn-based Greedy vs Random ===")
    print_table(
        [row],
        ["Mode", "A", "B", "A_wins", "B_wins", "Draw", "Episodes", "Score", "Winner"],
    )
    print("\nSummary:")
    print(
        f"[{row['Mode']}] {row['A']} vs {row['B']} → {row['Winner']} "
        f"(score {row['Score']}, episodes {row['Episodes']})"
    )


if __name__ == "__main__":
    main()