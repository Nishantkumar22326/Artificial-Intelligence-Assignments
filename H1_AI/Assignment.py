                                 # AI Assignment-1 (Nishant Kumar, 2022326)
"""
Container Loading Problem — Pure A* with fast admissible heuristics

Problem Representation
• State:
  - Yard: a tuple of LIFO stacks (bottom→top) of container IDs; only stack tops are pickable.
  - Ship: a tuple of columns (bottom→top) with per-column capacity `col_cap`. Columns [0..left_cols-1] are Left, others Right.
  - Goal: yard is empty AND every column has exactly `col_cap` containers (all items loaded).

• Actions:
  - Pick the top container from any non-empty yard stack and place it on any ship column with space.

• Constraints:
  1) Destination friendliness (soft): placing dest=d above any container with an earlier destination (< d) in the same column is
     allowed but penalized (“rehandle” cost).
  2) Capacity/balance:
     - Hard capacity: no column may exceed `col_cap`.
     - All containers must be loaded: total containers = n_cols * col_cap.
     - Balance penalty is charged only at the goal if |W_L - W_R| > τ * total_weight.

• Instance/Input:
  - Layout: n_dest, n_yard, n_cols, left_cols, col_cap.
  - Containers: dict {id → (dest:int, weight:float)} with dest=0 unloading first.
  - Yard configuration: list of stacks (lists) of container IDs (bottom→top).
  - Cost parameters: A (a_move), B (b_move), REHANDLE (rehandle_factor), λ (lambda_imbalance), τ (tau).

• Objective:
  Minimize total cost = Σ (move distance cost + immediate unload-blocking penalty) + final imbalance penalty, where
    - distance cost = (A + B·weight) · |yard_stack_index - column_index|
    - unload-blocking penalty = REHANDLE · (A + B·w_placed) · (# below in that column with dest < placed.dest)
    - final imbalance penalty (applied at goal only) = λ · max(0, |W_L - W_R| - τ · total_weight)

Solution Outline:
Pure A* search. The heuristic h is admissible and fast:
  • distance_LB: for each remaining container, min distance to any non-full column (using index gaps).
  • inversion_LB: conservative future rehandle lower bound from current “below earlier-destination” counts.
  • imbalance_LB: unavoidable final imbalance beyond the allowed τ·total_weight.
Dominated-action pruning drops column choices that are worse on both (distance cost, immediate inversion penalty). A fixed expansion
cap (`MAX_EXPANSIONS`) bounds runtime; if the cap is hit before reaching a goal, the run reports failure (cost = −1).

Runtime Behavior:
• `run_test_case_tiny()`: very small deterministic case (4 containers) to sanity-check logic.
• `run_test_case_1()`: standard random case (seed=42), friend-style header (no path) + minimal details.
• `analyze_scaling()`: optional scaling study across larger random instances.
"""

from __future__ import annotations

import dataclasses
import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import random
import time

# -------------------- Tunables --------------------

DEFAULT_A_MOVE = 1.0
DEFAULT_B_MOVE = 0.4
DEFAULT_REHANDLE_FACTOR = 2.0
DEFAULT_LAMBDA_IMBALANCE = 1.0
DEFAULT_TAU = 0.10

MAX_EXPANSIONS = 600_000   # Per-test cap for A* node expansions (edit if needed)


# -------------------- Data structures --------------------
@dataclass(frozen=True)
class Container:
    """A single container with an integer destination and a positive weight."""
    id: int
    dest: int      # 0 unloads earliest
    weight: float


@dataclass
class Problem:
    """
    Immutable problem description (layout + costs + container set).
    - n_cols: total columns; first `left_cols` are Left, remaining are Right.
    """
    # Layout
    n_dest: int
    n_yard: int
    n_cols: int
    left_cols: int
    col_cap: int

    # Cost parameters
    a_move: float = DEFAULT_A_MOVE
    b_move: float = DEFAULT_B_MOVE
    rehandle_factor: float = DEFAULT_REHANDLE_FACTOR
    lambda_imbalance: float = DEFAULT_LAMBDA_IMBALANCE
    tau: float = DEFAULT_TAU

    # Entities
    containers: Dict[int, Container] = dataclasses.field(default_factory=dict)
    yard_stacks: List[List[int]] = dataclasses.field(default_factory=list)

    # Precomputed distances
    _dist: List[List[int]] = dataclasses.field(default_factory=list, init=False)

    def init_precompute(self):
        """Precompute |yard_idx - col_idx| distances for all yard stacks and ship columns."""
        self._dist = [[abs(y - c) for c in range(self.n_cols)] for y in range(self.n_yard)]

    def distance(self, yard_idx: int, col_idx: int) -> int:
        """Return precomputed distance between a yard stack and a column index."""
        return self._dist[yard_idx][col_idx] if self._dist else abs(yard_idx - col_idx)

    def move_time(self, weight: float) -> float:
        """Per-move time component: A + B * weight."""
        return self.a_move + self.b_move * weight

    def total_weight(self) -> float:
        """Total weight of all containers."""
        return sum(c.weight for c in self.containers.values())

    def side_of(self, col_idx: int) -> str:
        """'L' for Left side (col < left_cols), otherwise 'R'."""
        return "L" if col_idx < self.left_cols else "R"


@dataclass(frozen=True)
class State:
    """
    Search state:
    - yard: tuple of stacks (bottom→top) of container IDs.
    - cols: tuple of ship columns (bottom→top).
    Goal: yard empty AND every column size == col_cap.
    """
    yard: Tuple[Tuple[int, ...], ...]
    cols: Tuple[Tuple[int, ...], ...]


    def is_goal(self, problem: Problem) -> bool:
        """True iff yard is empty and all columns are filled to capacity."""
        for st in self.yard:
            if st:
                return False
        for col in self.cols:
            if len(col) != problem.col_cap:
                return False
        return True

    def available_yard_tops(self) -> Iterable[Tuple[int, int]]:
        """Yield (yard_stack_index, top_container_id) for all non-empty stacks."""
        for i, stack in enumerate(self.yard):
            if stack:
                yield i, stack[-1]

    def columns_with_space(self, problem: Problem) -> List[int]:
        """Indices of columns that still have space (len(col) < col_cap)."""
        return [j for j, col in enumerate(self.cols) if len(col) < problem.col_cap]

    def left_right_weights(self, problem: Problem) -> Tuple[float, float]:
        """(W_L, W_R): total weights currently loaded on left and right sides."""
        wL = 0.0
        wR = 0.0
        for j, col in enumerate(self.cols):
            side = problem.side_of(j)
            for cid in col:
                w = problem.containers[cid].weight
                if side == "L": wL += w
                else: wR += w
        return wL, wR


@dataclass(order=True)
class Node:
    """A* node with (f, g, h) and parent/action for path reconstruction. Ordered by f."""
    f: float
    g: float = dataclasses.field(compare=False)
    h: float = dataclasses.field(compare=False)
    state: State = dataclasses.field(compare=False)
    parent: Optional["Node"] = dataclasses.field(compare=False, default=None)
    action: Optional[Tuple[int, int, int]] = dataclasses.field(compare=False, default=None)
    # action = (yard_idx, col_idx, container_id)


#  Costs

def incremental_unload_penalty(problem: Problem, state: State, col_idx: int, new_cid: int) -> float:
    """
    Immediate unload penalty added when placing container `new_cid` onto column `col_idx`:
      REHANDLE * move_time(weight(new_cid)) * (# below with dest < dest(new_cid))
    """
    new_dest = problem.containers[new_cid].dest
    new_w = problem.containers[new_cid].weight
    below = state.cols[col_idx]
    cnt = sum(1 for cid in below if problem.containers[cid].dest < new_dest)
    return cnt * problem.rehandle_factor * problem.move_time(new_w)


# Heuristics

def precompute_below_lt_counts(problem: Problem, state: State) -> List[List[int]]:
    """
    For each column j and each destination d, compute how many containers currently
    in column j have dest < d. Returns table counts[j][d].
    """
    counts = [[0] * problem.n_dest for _ in range(problem.n_cols)]
    for j, col in enumerate(state.cols):
        freq = [0] * problem.n_dest
        for cid in col:
            freq[problem.containers[cid].dest] += 1
        cum = 0
        for d in range(problem.n_dest):
            counts[j][d] = cum
            cum += freq[d]
    return counts


def fast_distance_lb(problem: Problem, state: State) -> float:
    """
    Admissible LB on remaining distance cost:
    For each remaining container, use the min distance to any non-full column (ignores capacity coupling).
    """
    cols_space = state.columns_with_space(problem)
    if not cols_space:
        return 0.0
    lb = 0.0
    for y, stack in enumerate(state.yard):
        if not stack:
            continue
        md = min(problem.distance(y, c) for c in cols_space)
        for cid in stack:
            w = problem.containers[cid].weight
            lb += problem.move_time(w) * md
    return lb


def fast_inversion_lb(problem: Problem, state: State) -> float:
    """
    Admissible LB on future unload penalties:
    For each remaining container (dest d), take min over current columns of
    (# below with dest < d) * REHANDLE * move_time(weight(container)).
    """
    cols_space = state.columns_with_space(problem)
    if not cols_space:
        return 0.0
    below_lt = precompute_below_lt_counts(problem, state)
    lb = 0.0
    for stack in state.yard:
        for cid in stack:
            c = problem.containers[cid]
            mt = problem.move_time(c.weight)
            inv_unit = problem.rehandle_factor * mt
            lb += inv_unit * min(below_lt[j][c.dest] for j in cols_space)
    return lb


def imbalance_lb(problem: Problem, state: State) -> float:
    """
    Admissible LB on final imbalance penalty:
    Even with best remaining placements, side difference cannot be reduced below
    max(0, |WL-WR| - remaining_weight). Any excess over τ*TotalW is penalized by λ.
    """
    WL, WR = state.left_right_weights(problem)
    totalW = problem.total_weight()
    remainingW = totalW - (WL + WR)
    delta = abs(WL - WR)
    best_possible = max(0.0, delta - remainingW)
    allowed = problem.tau * totalW
    over = max(0.0, best_possible - allowed)
    return problem.lambda_imbalance * over


def heuristic(problem: Problem, state: State) -> float:
    """
    Fast admissible heuristic:
      h = distance_LB + inversion_LB + imbalance_LB
    """
    n_remaining = sum(len(st) for st in state.yard)
    if n_remaining == 0:
        return imbalance_lb(problem, state)
    return fast_distance_lb(problem, state) + fast_inversion_lb(problem, state) + imbalance_lb(problem, state)


# Successor generation 

def successors(problem: Problem, state: State, prune_dominated: bool = True) -> Iterable[Tuple[State, Tuple[int,int,int], float]]:
    """
    Generate successor states by placing any available yard-top container on any column with space.
    Dominated-action pruning removes column choices that are strictly worse on both
    (distance cost, immediate inversion penalty).
    """
    cols_space = state.columns_with_space(problem)
    if not cols_space:
        return
    eps = 1e-9
    for y_idx, cid in state.available_yard_tops():
        w = problem.containers[cid].weight
        mt = problem.move_time(w)
        cands: List[Tuple[int, float, float]] = []
        for c_idx in cols_space:
            load_time = mt * problem.distance(y_idx, c_idx)
            inv_pen = incremental_unload_penalty(problem, state, c_idx, cid)
            cands.append((c_idx, load_time, inv_pen))
        if prune_dominated and len(cands) > 1:
            kept: List[Tuple[int, float, float]] = []
            for i, (ci, li, vi) in enumerate(cands):
                dominated = False
                for j, (cj, lj, vj) in enumerate(cands):
                    if j == i: continue
                    if (lj <= li + eps) and (vj <= vi + eps) and ((lj < li - eps) or (vj < vi - eps)):
                        dominated = True; break
                if not dominated: kept.append((ci, li, vi))
            cands = kept
        cands.sort(key=lambda t: (t[1], t[2], t[0]))
        for c_idx, load_time, inv_pen in cands:
            step_cost = load_time + inv_pen
            new_yard = [list(st) for st in state.yard]
            new_yard[y_idx] = new_yard[y_idx][:-1]
            new_cols = [list(col) for col in state.cols]
            new_cols[c_idx] = new_cols[c_idx] + [cid]
            next_state = State(
                yard=tuple(tuple(st) for st in new_yard),
                cols=tuple(tuple(col) for col in new_cols),
            )
            yield next_state, (y_idx, c_idx, cid), step_cost


# Pure A* with explicit cap reporting 

def astar(problem: Problem, start: State, max_expansions: int) -> Tuple[Optional[Node], bool, int]:
    """
    Pure A* search (w=1.0). Returns (goal_node_or_None, hit_cap: bool, expansions: int).
    The final imbalance penalty is added only when a goal is popped.
    """
    h0 = heuristic(problem, start)
    start_node = Node(f=h0, g=0.0, h=h0, state=start)
    open_heap: List[Node] = [start_node]
    best_g: Dict[State, float] = {start: 0.0}
    expansions = 0

    while open_heap:
        current = heapq.heappop(open_heap)
        expansions += 1
        if expansions > max_expansions:
            return None, True, expansions  # cap hit

        if current.state.is_goal(problem):
            WL, WR = current.state.left_right_weights(problem)
            totalW = problem.total_weight()
            allowed = problem.tau * totalW
            over = max(0.0, abs(WL - WR) - allowed)
            current.g += problem.lambda_imbalance * over
            current.f = current.g
            return current, False, expansions

        for nstate, action, step_cost in successors(problem, current.state, prune_dominated=True):
            g2 = current.g + step_cost
            if nstate in best_g and g2 >= best_g[nstate]:
                continue
            best_g[nstate] = g2
            h2 = heuristic(problem, nstate)
            f2 = g2 + h2
            heapq.heappush(open_heap, Node(f=f2, g=g2, h=h2, state=nstate, parent=current, action=action))

    return None, False, expansions  # queue exhausted without goal


# Instance builders 

def random_instance(seed: int, n_dest: int, n_yard: int, n_cols: int, left_cols: int, col_cap: int,
                    weight_low: float, weight_high: float) -> Tuple[Problem, State]:
    """
    Create a random instance consistent with the model:
    - Exactly n_cols * col_cap containers are generated so the ship is fully loaded at goal.
    - Yard stacks are filled bottom→top by shuffling container IDs.
    """
    rng = random.Random(seed)
    N = n_cols * col_cap
    containers: Dict[int, Container] = {}
    for i in range(N):
        dest = rng.randrange(n_dest)
        weight = rng.uniform(weight_low, weight_high)
        containers[i] = Container(id=i, dest=dest, weight=weight)
    yard_stacks: List[List[int]] = [[] for _ in range(n_yard)]
    order = list(range(N)); rng.shuffle(order)
    for idx, cid in enumerate(order):
        yard_stacks[idx % n_yard].append(cid)  # bottom→top
    start = State(
        yard=tuple(tuple(st) for st in yard_stacks),
        cols=tuple(tuple() for _ in range(n_cols)),
    )
    prob = Problem(
        n_dest=n_dest, n_yard=n_yard, n_cols=n_cols, left_cols=left_cols, col_cap=col_cap,
        containers=containers, yard_stacks=[list(st) for st in yard_stacks],
    )
    prob.init_precompute()
    return prob, start


#  Replay to compute totals

def reconstruct_actions(goal: Node) -> List[Tuple[int,int,int]]:
    """Return the action list from start to goal: [(yard_idx, col_idx, cid), ...]."""
    seq: List[Tuple[int,int,int]] = []
    node = goal
    while node and node.action is not None:
        seq.append(node.action)
        node = node.parent
    seq.reverse()
    return seq


def compute_breakdown(problem: Problem, start: State, actions: List[Tuple[int,int,int]]):
    """
    Replay actions from the start state to accumulate totals without printing steps.
    Returns (dist_total, inv_total, imb_pen, total, final_state).
    """
    curr = start
    dist_total = 0.0
    inv_total = 0.0
    for (y, c, cid) in actions:
        w = problem.containers[cid].weight
        mt = problem.move_time(w)
        dist_total += mt * problem.distance(y, c)
        inv_total += incremental_unload_penalty(problem, curr, c, cid)

        # Apply action to advance state
        new_yard = [list(st) for st in curr.yard]
        new_yard[y].pop()
        new_cols = [list(col) for col in curr.cols]
        new_cols[c].append(cid)
        curr = State(yard=tuple(tuple(st) for st in new_yard),
                     cols=tuple(tuple(col) for col in new_cols))

    WL, WR = curr.left_right_weights(problem)
    imb_allowed = problem.tau * problem.total_weight()
    imb_pen = problem.lambda_imbalance * max(0.0, abs(WL - WR) - imb_allowed)
    total = dist_total + inv_total + imb_pen
    return dist_total, inv_total, imb_pen, total, curr


# Optional: scaling analysis

def analyze_scaling():
    """
    Run pure A* on progressively larger random instances and print a compact summary.
    Uses the same MAX_EXPANSIONS as the main test. Reports solved runs, cap hits, and averages over solved ones.
    """
    cases = [
        dict(n_dest=3, n_yard=4, n_cols=6,  col_cap=3, weight_low=1.0, weight_high=5.0),
        dict(n_dest=3, n_yard=4, n_cols=6,  col_cap=4, weight_low=1.0, weight_high=5.0),
        dict(n_dest=3, n_yard=5, n_cols=8,  col_cap=3, weight_low=1.0, weight_high=5.0),
        dict(n_dest=4, n_yard=5, n_cols=8,  col_cap=4, weight_low=1.0, weight_high=6.0),
        dict(n_dest=4, n_yard=6, n_cols=10, col_cap=4, weight_low=1.0, weight_high=6.0),
    ]
    seeds = [200, 201, 202]

    print("\n=== Scaling analysis (pure A*) ===")
    print(f"(cap: MAX_EXPANSIONS={MAX_EXPANSIONS})")
    print("N  colsxcap  yard  dest | solved/total cap_hits | avg_cost  avg_steps  avg_exp  avg_ms")
    print("-- --------- ----- ----- | ------------ ------- | -------- ---------- -------- --------")

    for case in cases:
        n_cols = case["n_cols"]; col_cap = case["col_cap"]
        left_cols = n_cols // 2
        N = n_cols * col_cap

        solved = 0
        cap_hits = 0
        exhausted = 0
        cost_sum = steps_sum = exp_sum = time_sum_ms = 0.0

        for sd in seeds:
            problem, start = random_instance(
                seed=sd,
                n_dest=case["n_dest"],
                n_yard=case["n_yard"],
                n_cols=n_cols,
                left_cols=left_cols,
                col_cap=col_cap,
                weight_low=case["weight_low"],
                weight_high=case["weight_high"],
            )
            t0 = time.perf_counter()
            goal, hit_cap, expansions = astar(problem, start, max_expansions=MAX_EXPANSIONS)
            t1 = time.perf_counter()
            dt_ms = (t1 - t0) * 1000.0

            if goal is not None:
                actions = reconstruct_actions(goal)
                solved += 1
                cost_sum  += goal.g
                steps_sum += len(actions)
                exp_sum   += expansions
                time_sum_ms += dt_ms
            else:
                if hit_cap:
                    cap_hits += 1
                else:
                    exhausted += 1

        if solved > 0:
            avg_cost  = cost_sum / solved
            avg_steps = steps_sum / solved
            avg_exp   = exp_sum / solved
            avg_ms    = time_sum_ms / solved
        else:
            avg_cost = avg_steps = avg_exp = avg_ms = float('nan')

        print(f"{N:<2} {n_cols}x{col_cap:<6} {case['n_yard']:<5} {case['n_dest']:<5} | "
              f"{solved}/{len(seeds):<12} {cap_hits:<7} | "
              f"{avg_cost:>8.3f} {avg_steps:>10.1f} {avg_exp:>8.0f} {avg_ms:>8.1f}")

        if cap_hits or exhausted:
            note_parts = []
            if cap_hits:
                note_parts.append(f"cap hits={cap_hits}")
            if exhausted:
                note_parts.append(f"exhausted={exhausted}")
            print("   ↳ Note:", ", ".join(note_parts))


#  Tiny test (4 containers)

def run_test_case_tiny() -> None:
    """Very small deterministic case: 4 containers, easy to verify (no path printed)."""
    # 2 columns × cap 2 = 4 containers; 2 yard stacks; 2 destinations; left_cols=1
    problem, start = random_instance(
        seed=7, n_dest=2, n_yard=2, n_cols=2, left_cols=1, col_cap=2,
        weight_low=1.0, weight_high=3.0
    )

    goal, hit_cap, expansions = astar(problem, start, max_expansions=MAX_EXPANSIONS)

    N = problem.n_cols * problem.col_cap
    weights_in_id_order = [problem.containers[i].weight for i in range(N)]

    print("CONTAINER LOADING SOLVER TEST CASE (TINY)")
    print("==================================================")
    print(f"Test Case: {N} containers, {problem.n_cols} columns × cap {problem.col_cap}, "
          f"{problem.n_yard} yard stacks, {problem.n_dest} destinations")
    print("Container weights:", [round(w, 2) for w in weights_in_id_order])
    print(f"Total weight: {problem.total_weight():.2f}")
    print("Starting nodes explored: 0")
    print(f"Ending nodes explored: {expansions}")
    success = (goal is not None)
    print(f"Success: {'YES' if success else 'NO'}")
    print(f"Total cost: {goal.g:.3f}" if success else "Total cost: -1")

    if not success:
        if hit_cap:
            print("\nNote: A* expansion cap too low "
                  f"(cap={MAX_EXPANSIONS}). Increase MAX_EXPANSIONS.")
        else:
            print("\nNote: search queue exhausted without goal "
                  "(instance may be infeasible).")
        return

    actions = reconstruct_actions(goal)
    dist_total, inv_total, imb_pen, _, _ = compute_breakdown(problem, start, actions)

    print("\nDetails")
    print("-------")
    print(f"A* settings: MAX_EXPANSIONS={MAX_EXPANSIONS}")
    print(f"Steps (moves): {len(actions)}, A* expansions: {expansions}")
    line = f"Cost breakdown: distance={dist_total:.3f}"
    if inv_total > 1e-9:
        line += f", unload_penalty={inv_total:.3f}"
    if imb_pen > 1e-9:
        line += f", final_imbalance={imb_pen:.3f}"
    print(line)


# Standard test case (seed=42)

def run_test_case_1() -> None:
    """Build instance (seed=42), run pure A*, print friend-style header + minimal details (no path)."""
    problem, start = random_instance(
        seed=42, n_dest=3, n_yard=4, n_cols=6, left_cols=3, col_cap=3,
        weight_low=1.0, weight_high=5.0
    )

    goal, hit_cap, expansions = astar(problem, start, max_expansions=MAX_EXPANSIONS)

    N = problem.n_cols * problem.col_cap
    weights_in_id_order = [problem.containers[i].weight for i in range(N)]

    print("CONTAINER LOADING SOLVER TEST CASE")
    print("==================================================")
    print(f"Test Case: {N} containers, {problem.n_cols} columns × cap {problem.col_cap}, "
          f"{problem.n_yard} yard stacks, {problem.n_dest} destinations")
    print("Container weights:", [round(w, 2) for w in weights_in_id_order])
    print(f"Total weight: {problem.total_weight():.2f}")
    print("Starting nodes explored: 0")
    print(f"Ending nodes explored: {expansions}")
    success = (goal is not None)
    print(f"Success: {'YES' if success else 'NO'}")
    print(f"Total cost: {goal.g:.3f}" if success else "Total cost: -1")

    if not success:
        if hit_cap:
            print("\nNote: A* expansion cap too low "
                  f"(cap={MAX_EXPANSIONS}). Increase MAX_EXPANSIONS.")
        else:
            print("\nNote: search queue exhausted without goal "
                  "(instance may be infeasible).")
        return

    actions = reconstruct_actions(goal)
    dist_total, inv_total, imb_pen, _, _ = compute_breakdown(problem, start, actions)

    print("\nDetails")
    print("-------")
    print(f"A* settings: MAX_EXPANSIONS={MAX_EXPANSIONS}")
    print(f"Steps (moves): {len(actions)}, A* expansions: {expansions}")
    line = f"Cost breakdown: distance={dist_total:.3f}"
    if inv_total > 1e-9:
        line += f", unload_penalty={inv_total:.3f}"
    if imb_pen > 1e-9:
        line += f", final_imbalance={imb_pen:.3f}"
    print(line)


if __name__ == "__main__":
    # Run both tests by default; comment/uncomment as needed
    run_test_case_1()
    # run_test_case_tiny()

    # Optional: run the scaling study
    # analyze_scaling()