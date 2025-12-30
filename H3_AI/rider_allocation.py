# rider_allocation.py
# Baseline solver for Rider–Order allocation (fuel + time optimal)

import itertools
import math

INF = 10**9


class RiderAllocationProblem:
    def __init__(self, nodes, edges, riders, orders):
        """
        nodes: list of node labels, e.g. ['A','B','C','D','E','F']
        edges: dict {(u, v): distance} undirected
        riders: dict {rider_id: start_node}
        orders: dict {order_id: {'pickup': node, 'drop': node}}
        """
        self.nodes = nodes
        self.edges = edges
        self.riders = riders
        self.orders = orders

        self.idx = {n: i for i, n in enumerate(nodes)}
        self.n = len(nodes)
        self.dist = [[INF] * self.n for _ in range(self.n)]

        for i in range(self.n):
            self.dist[i][i] = 0

        # Build undirected graph
        for (u, v), w in edges.items():
            i, j = self.idx[u], self.idx[v]
            if w < self.dist[i][j]:
                self.dist[i][j] = self.dist[j][i] = w

        # Floyd–Warshall to get all-pairs shortest paths
        for k in range(self.n):
            for i in range(self.n):
                dik = self.dist[i][k]
                if dik == INF:
                    continue
                for j in range(self.n):
                    if self.dist[i][j] > dik + self.dist[k][j]:
                        self.dist[i][j] = dik + self.dist[k][j]

    def d(self, u, v):
        """Shortest distance between nodes u and v."""
        return self.dist[self.idx[u]][self.idx[v]]

    # ---------- sequencing of pickups / drops ----------

    def _valid_sequences(self, order_ids):
        """
        All valid sequences of ('P', oid) / ('D', oid)
        such that each D comes after its own P.
        """
        if not order_ids:
            return [[]]

        res = []
        remaining_P = set(order_ids)
        remaining_D = set(order_ids)

        def backtrack(seq, remP, remD, picked):
            if not remP and not remD:
                res.append(seq.copy())
                return

            # Pick up any remaining order
            for o in list(remP):
                remP.remove(o)
                picked.add(o)
                seq.append(('P', o))
                backtrack(seq, remP, remD, picked)
                seq.pop()
                picked.remove(o)
                remP.add(o)

            # Drop any picked-but-not-dropped order
            for o in list(remD):
                if o in picked:
                    remD.remove(o)
                    seq.append(('D', o))
                    backtrack(seq, remP, remD, picked)
                    seq.pop()
                    remD.add(o)

        backtrack([], remaining_P, remaining_D, set())
        return res

    def _min_distance_for_rider(self, start_pos, assigned_orders):
        """
        For a given rider and set of assigned orders, find the best
        pickup/drop sequence and its distance.
        """
        if not assigned_orders:
            return 0, []

        best = (INF, None)
        sequences = self._valid_sequences(assigned_orders)

        for seq in sequences:
            pos = start_pos
            total = 0
            for kind, oid in seq:
                if kind == 'P':
                    node = self.orders[oid]['pickup']
                else:
                    node = self.orders[oid]['drop']
                total += self.d(pos, node)
                pos = node

            if total < best[0]:
                best = (total, seq)

        return best

    def solve(self):
        """
        Try all assignments of orders to riders, and all valid sequences per rider.
        Return two optima:
          - fuel-optimal: min sum of distances over riders
          - time-optimal: min max distance over riders
        """
        order_ids = list(self.orders.keys())
        rider_ids = list(self.riders.keys())

        best_fuel = (math.inf, None)
        best_time = (math.inf, None)

        # All mappings: each order -> some rider
        for assignment in itertools.product(rider_ids, repeat=len(order_ids)):
            assign_map = {r: [] for r in rider_ids}
            for oid, r in zip(order_ids, assignment):
                assign_map[r].append(oid)

            rider_info = {}
            for r in rider_ids:
                dist_r, seq = self._min_distance_for_rider(self.riders[r], assign_map[r])
                rider_info[r] = {
                    'distance': dist_r,
                    'sequence': seq,
                    'orders': assign_map[r]
                }

            total_fuel = sum(info['distance'] for info in rider_info.values())
            elapsed_time = max(info['distance'] for info in rider_info.values())

            if total_fuel < best_fuel[0]:
                best_fuel = (total_fuel, rider_info)

            if elapsed_time < best_time[0]:
                best_time = (elapsed_time, rider_info)

        return best_fuel, best_time


# ---------- Example instances to use for HW3 ----------

def example_instance_1():
    """
    Same as the one in the assignment statement.
    """
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    edges = {
        ('A', 'B'): 2,
        ('A', 'C'): 4,
        ('B', 'C'): 1,
        ('C', 'D'): 2,
        ('C', 'E'): 3,
        ('D', 'F'): 3,
        ('D', 'E'): 2,
        ('E', 'F'): 2,
    }
    riders = {'R1': 'A', 'R2': 'D'}
    orders = {
        'O1': {'pickup': 'B', 'drop': 'E'},
        'O2': {'pickup': 'C', 'drop': 'F'},
        'O3': {'pickup': 'A', 'drop': 'D'},
    }
    return nodes, edges, riders, orders


def example_instance_2():
    """
    Modified graph, 2 riders, 3 orders.
    """
    nodes = ['A', 'B', 'C', 'D', 'E']
    edges = {
        ('A', 'B'): 3,
        ('A', 'C'): 2,
        ('B', 'C'): 2,
        ('B', 'D'): 4,
        ('C', 'D'): 3,
        ('C', 'E'): 5,
        ('D', 'E'): 1,
    }
    riders = {'R1': 'A', 'R2': 'D'}
    orders = {
        'O1': {'pickup': 'B', 'drop': 'E'},
        'O2': {'pickup': 'C', 'drop': 'D'},
        'O3': {'pickup': 'A', 'drop': 'C'},
    }
    return nodes, edges, riders, orders


def example_instance_3():
    """
    Single rider with 3 orders.
    """
    nodes = ['A', 'B', 'C', 'D']
    edges = {
        ('A', 'B'): 2,
        ('B', 'C'): 2,
        ('C', 'D'): 2,
        ('A', 'D'): 5,
    }
    riders = {'R1': 'A'}
    orders = {
        'O1': {'pickup': 'B', 'drop': 'D'},
        'O2': {'pickup': 'C', 'drop': 'A'},
        'O3': {'pickup': 'A', 'drop': 'C'},
    }
    return nodes, edges, riders, orders


def solve_and_print(label, nodes, edges, riders, orders):
    print(f"\n=== {label} ===")
    prob = RiderAllocationProblem(nodes, edges, riders, orders)
    best_fuel, best_time = prob.solve()

    print("Fuel-optimal solution:")
    print("  Total fuel:", best_fuel[0])
    for r, info in best_fuel[1].items():
        print(f"    {r}: distance={info['distance']}, orders={info['orders']}, sequence={info['sequence']}")

    print("Time-optimal solution:")
    print("  Elapsed time:", best_time[0])
    for r, info in best_time[1].items():
        print(f"    {r}: distance={info['distance']}, orders={info['orders']}, sequence={info['sequence']}")


if __name__ == "__main__":
    solve_and_print("Instance 1", *example_instance_1())
    solve_and_print("Instance 2", *example_instance_2())
    solve_and_print("Instance 3", *example_instance_3())