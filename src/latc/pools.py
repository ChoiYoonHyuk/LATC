"""Public pool coupling and label-independent geometry selection (L.2, L.3, P.1)."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from .algebra import augmented_rank


@dataclass(frozen=True)
class PoolDesign:
    pools: tuple[np.ndarray, np.ndarray]
    blocks: tuple[np.ndarray, np.ndarray]
    remainders: tuple[np.ndarray, np.ndarray]
    kind: str
    flat_available: tuple[bool, bool] = (False, False)

    def validate(self) -> None:
        if len(self.blocks[0]) != len(self.blocks[1]):
            raise ValueError("Designated regional blocks must have equal size")
        for p, b, c in zip(self.pools, self.blocks, self.remainders):
            if len(set(map(int, p))) != len(p):
                raise ValueError("Repeated public coordinates")
            if len(set(map(int, b))) != len(b) or len(set(map(int, c))) != len(c):
                raise ValueError("Repeated block/remainder coordinates")
            if set(map(int, b)) & set(map(int, c)) or set(map(int, p)) != set(map(int, b)) | set(map(int, c)):
                raise ValueError("Pool must be the disjoint union of block and remainder")


def make_design(pools: tuple[np.ndarray, np.ndarray], blocks: tuple[np.ndarray, np.ndarray],
                kind: str, flat_available: tuple[bool, bool] = (False, False)) -> PoolDesign:
    remainders = tuple(p[~np.isin(p, b)] for p, b in zip(pools, blocks))
    design = PoolDesign(pools, blocks, remainders, kind, flat_available)
    design.validate()
    return design


def first_four_check(pool: np.ndarray) -> np.ndarray:
    """First pair-sum collision in lexicographic public-position order."""
    seen: dict[int, tuple[int, int]] = {}
    p = list(map(int, pool))
    for i, a in enumerate(p):
        for b in p[i + 1:]:
            s = a ^ b
            if s in seen:
                c, d = seen[s]
                # Equal sums of distinct pairs cannot share exactly one endpoint.
                return np.array([c, d, a, b], dtype=np.int64)
            seen[s] = (a, b)
    raise ValueError("The supplied public pool contains no four-point check")


def two_disjoint_checks(pool: np.ndarray) -> np.ndarray:
    first = first_four_check(pool)
    second = first_four_check(pool[~np.isin(pool, first)])
    return np.concatenate((first, second))


def find_affine_flat(pool: np.ndarray) -> np.ndarray | None:
    """Exhaustive eight-point affine-flat search, not a random candidate search.

    Group pairs by XOR in order of their first public-index occurrence. Two
    pairs make a plane. Two distinct cosets of the same plane-direction space
    make a three-flat. Every three-flat admits this decomposition (Appendix P.1).
    """
    values = list(map(int, pool))
    if len(set(values)) != len(values):
        raise ValueError("A pool must contain distinct coordinates")
    groups: dict[int, list[tuple[int, int]]] = {}
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            groups.setdefault(a ^ b, []).append((a, b))
    planes: dict[tuple[int, int, int], tuple[int, tuple[int, ...]]] = {}
    public_index = {x: i for i, x in enumerate(values)}
    for s, pairs in groups.items():
        for (a, b), (c, d) in combinations(pairs, 2):
            direction = tuple(sorted((s, a ^ c, b ^ c)))
            plane = (a, b, c, d)
            coset = min(plane)
            previous = planes.get(direction)
            if previous is not None and previous[0] != coset:
                result = np.array(sorted(previous[1] + plane, key=public_index.__getitem__), dtype=np.int64)
                if not is_affine_flat(result):
                    raise ArithmeticError("Internal affine-flat search validation failed")
                return result
            if previous is None:
                planes[direction] = (coset, plane)
    return None


def is_affine_flat(points: np.ndarray) -> bool:
    p = tuple(map(int, points))
    if len(p) != 8 or len(set(p)) != 8 or augmented_rank(p) != 4:
        return False
    translated = {x ^ p[0] for x in p}
    return all((x ^ y) in translated for x in translated for y in translated)


def paired_pool_designs(d: int, rng: np.random.Generator, q: int | None = None,
                        block_size: int = 8, include_flat: bool = True
                        ) -> tuple[dict[str, PoolDesign], tuple[np.ndarray, np.ndarray]]:
    m = 1 << d
    q = (d + 1) ** 2 if q is None else int(q)
    if not 0 < block_size <= q <= m or block_size & (block_size - 1):
        raise ValueError("Require a power-of-two block size <= q <= 2^d; q is never silently clipped")
    permutations = tuple(rng.permutation(m) for _ in range(2))
    random_pools = tuple(p[:q].copy() for p in permutations)
    block = np.arange(block_size, dtype=np.int64)
    structured = tuple(np.concatenate((block, p[p >= block_size][:q - block_size])) for p in permutations)
    designs = {
        "prefix": make_design(random_pools, tuple(p[:block_size] for p in random_pools), "prefix"),
        "subcube": make_design(structured, (block.copy(), block.copy()), "subcube"),
    }
    if block_size == 8:
        checks = tuple(two_disjoint_checks(p) for p in random_pools)
        designs["two_check"] = make_design(random_pools, checks, "two_check")
        if include_flat:
            flats = tuple(find_affine_flat(p) for p in random_pools)
            available = tuple(f is not None for f in flats)
            blocks = tuple(f if f is not None else c for f, c in zip(flats, checks))
            designs["affine_flat"] = make_design(random_pools, blocks, "affine_flat", available)
    targets = tuple(np.setdiff1d(np.arange(m), np.union1d(a, b))
                    for a, b in zip(random_pools, structured))
    if sum(map(len, targets)) == 0:
        raise ValueError("No common targets remain outside the union of both pools")
    return designs, targets
