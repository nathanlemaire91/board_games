import math
import random
import time
from itertools import count

from awale import BoardState


class Node:
    def __init__(self, state: BoardState, parent: "Node | None" = None, move: int | None = None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children: list[Node] = []
        self.untried_moves = [] if state.is_over() else state.get_possible_moves()
        self.visits = 0
        self.wins = 0.0
        # Wins are counted from the point of view of the player who made the move leading here
        self.player_just_moved = state.opponent

    def is_fully_expanded(self):
        return not self.untried_moves

    def is_terminal(self):
        return not self.untried_moves and not self.children

    def best_child(self, exploration: float):
        log_visits = math.log(self.visits)
        return max(
            self.children,
            key=lambda child: child.wins / child.visits + exploration * math.sqrt(log_visits / child.visits),
        )

    def expand(self):
        move = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        state = self.state.copy()
        state.make_move(move)
        child = Node(state, parent=self, move=move)
        self.children.append(child)
        return child

    def update(self, winner):
        self.visits += 1
        if winner is None:
            self.wins += 0.5
        elif winner == self.player_just_moved:
            self.wins += 1


class MCTS:
    """UCT search. With reuse, the subtree of the position reached since the last search
    (our move, then usually the opponent's reply) is kept, along with its statistics.

    With `move_time`, each search runs for that many seconds, `iterations` being an
    optional cap (None: no cap)."""

    def __init__(
        self,
        iterations: int | None = 1000,
        exploration: float = math.sqrt(2),
        max_rollout_moves: int = 200,
        reuse: bool = True,
        move_time: float | None = None,
    ):
        if iterations is None and move_time is None:
            raise ValueError("unlimited iterations need a time limit")
        self.iterations = iterations
        self.move_time = move_time
        self.exploration = exploration
        # Awale games can loop without captures, so rollouts are cut off and scored on captured seeds
        self.max_rollout_moves = max_rollout_moves
        self.reuse = reuse
        self.previous_root: Node | None = None

    def reusable_root(self, state: BoardState) -> Node | None:
        """The node for `state` among the previous root and its children and grandchildren."""
        if self.previous_root is None:
            return None
        frontier = [self.previous_root]
        for _ in range(3):
            for node in frontier:
                if node.state == state:
                    node.parent = None  # Detach, so backpropagation stops here and the rest is freed
                    return node
            frontier = [child for node in frontier for child in node.children]
        return None

    def search(self, state: BoardState) -> Node:
        root = (self.reusable_root(state) if self.reuse else None) or Node(state.copy())
        if self.reuse:
            self.previous_root = root
        deadline = math.inf if self.move_time is None else time.perf_counter() + self.move_time
        for iteration in count():
            # At least one iteration, so the root has a child to play
            if iteration == self.iterations or (iteration and time.perf_counter() > deadline):
                break
            node = self.select(root)
            if not node.is_terminal():
                node = node.expand()
            winner = self.rollout(node.state)
            self.backpropagate(node, winner)
        return root

    def select(self, node: Node) -> Node:
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.exploration)
        return node

    def rollout(self, state: BoardState):
        state = state.copy()
        for _ in range(self.max_rollout_moves):
            if state.is_over():
                break
            moves = state.get_possible_moves()
            if not moves:
                break
            state.make_move(random.choice(moves))
        # A rollout cut short is scored on captured seeds, like a finished game
        return state.winner()

    def backpropagate(self, node: Node | None, winner):
        while node is not None:
            node.update(winner)
            node = node.parent

    def best_move(self, state: BoardState) -> int | None:
        root = self.search(state)
        if not root.children:
            return None  # Game is already over
        return max(root.children, key=lambda child: child.visits).move


if __name__ == "__main__":
    from awale import new_game

    state = new_game()
    root = MCTS(iterations=2000).search(state)
    for child in sorted(root.children, key=lambda c: c.move):
        print(f"move {child.move}: visits={child.visits} win rate={child.wins / child.visits:.2f}")
