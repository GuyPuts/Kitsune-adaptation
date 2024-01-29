from collections import namedtuple
from math import floor, pow, log, ceil
import random


class QDigest():
    class Node():
        def __init__(self, id, is_leaf=False, initial_count=1):
            self.id = id
            self.count = initial_count
            self.is_leaf = is_leaf

        def inc(self, amount=1):
            self.count += amount

        def __lt__(self, other):
            return self.id < other.id

        def __repr__(self):
            return f"{{id={self.id}, cnt={self.count}}}"

        @property
        def is_root(self):
            return self.id == 1

        def parent_id(self):
            if self.is_root:
                return None
            return int(floor(self.id / 2.0))

        def children(self):
            return (2 * self.id, 2 * self.id + 1)

        def sibling_id(self):
            if self.is_root:
                return None
            return self.id + 1 if self.id % 2 == 0 else self.id - 1

    empty_node = Node(id=-1, initial_count=0)

    def __init__(self, universe_size, compression_factor):
        self.size = universe_size
        self.digest = []
        self.id = 1
        self.k = compression_factor

    def __add__(self, other):
        if self.k != other.k:
            raise ValueError("Compression factors of two digests not the same")

        digest = QDigest(max(self.size, other.size), self.k)
        digest.digest = self.digest[:]
        for node in other.digest:
            digest._insert_or_modify_node(node.id, node.count)

        digest.compress()

        return digest

    @property
    def n(self):
        return sum(x.count for x in self.digest)

    @property
    def height(self):
        return int(ceil(log(self.size, 2)))

    def _get_node(self, node_id):
        try:
            node = next(x for x in self.digest if x.id == node_id)
            return node
        except StopIteration:
            return QDigest.empty_node

    def _remove_node(self, node_id):
        node = self._get_node(node_id)
        if node is not QDigest.empty_node:
            self.digest.remove(node)

    def violates_prop_1(self, node):
        if node.is_root or node.is_leaf:
            return False
        else:
            return node.count <= int(floor(self.n / self.k))

    def violates_prop_2(self, node):
        sibling_count = self._get_node(node.sibling_id()).count
        parent_count = self._get_node(node.parent_id()).count

        return node.count + sibling_count + parent_count <= int(floor(self.n / self.k))

    def _insert_or_modify_node(self, node_id, inc_by=1):
        current = self._get_node(node_id)
        if current is not QDigest.empty_node:
            current.inc(inc_by)
        else:
            current = QDigest.Node(id=node_id, initial_count=inc_by)
            self.digest.append(current)

    def insert(self, value):
        if value > self.size:
            raise ValueError()

        id_for_leaf_node = int(pow(2, self.height) + value - 1)
        self._insert_or_modify_node(node_id=id_for_leaf_node, inc_by=1)

    def compress(self):
        for l in range(self.height, 0, -1):
            level_l_nodes = sorted((x for x in self.digest if pow(2, l) <= x.id < pow(2, l + 1)), key=lambda x: x.id)
            for node in level_l_nodes:
                if self.violates_prop_2(node):
                    merged_count = node.count + self._get_node(node.sibling_id()).count
                    self._insert_or_modify_node(node_id=node.parent_id(), inc_by=merged_count)
                    self._remove_node(node.id)
                    self._remove_node(node.sibling_id())

    def quantile_query(self, fractions):
        running_sums = {}
        results = {}
        for fraction in fractions:
            if not 0 < fraction < 1:
                raise ValueError("Fraction should be between 0 and 1 exclusive")
            else:
                running_sums[str(fraction)] = 0
                results[str(fraction)] = None

        def _get_node_for_traversal(node_id):
            current = self._get_node(node_id)
            if current.id == -1:
                current = QDigest.Node(id=node_id, initial_count=0)
            return current

        def basic_dfs(node):
            if node and node.id < self.size * 2:
                left_node, right_node = (_get_node_for_traversal(x) for x in node.children())

                for child_node in basic_dfs(left_node):
                    yield child_node
                for child_node in basic_dfs(right_node):
                    yield child_node

                for fraction in fractions:
                    if not results[str(fraction)] and node.count + running_sums[str(fraction)] < fraction * self.n:
                        running_sums[str(fraction)] = running_sums[str(fraction)] + node.count
                    else:
                        results[str(fraction)] = running_sums[str(fraction)]

        [x for x in basic_dfs(_get_node_for_traversal(1))]
        return results
