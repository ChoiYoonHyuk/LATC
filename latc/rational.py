from __future__ import annotations
from fractions import Fraction
import itertools
from .affine import clean_labels


def integer_target_weights(d: int, eta: Fraction):
    if d not in (1, 2) or not Fraction(0) <= eta < Fraction(1, 2):
        raise ValueError("rational validation supports d=1,2 and 0<=eta<1/2")
    m = 1 << d
    p, q = eta.numerator, eta.denominator
    clean = []
    for beta, slope in itertools.product(range(2), range(m)):
        ys = clean_labels(d, beta, slope)
        clean.append(sum(int(v) << i for i, v in enumerate(ys)))
    reliable = []
    for y in range(1 << m):
        reliable.append(sum((q-p)**(m-(y^task).bit_count()) * p**((y^task).bit_count()) for task in clean))

    weights = [reliable[y & ((1 << m)-1)]+reliable[y >> m] for y in range(1 << (2*m))]
    return weights


def rational_history_costs(d: int, pool: tuple[int, ...], eta: Fraction):
    weights = integer_target_weights(d, eta)
    m, N = 1 << d, 2*(1 << d)
    global_vertices = [j*m+x for j in (0, 1) for x in pool]
    n = len(global_vertices)
    if n > 8:
        raise ValueError("at most 8 eligible identities in rational validation")
    total = sum(weights)
    costs, masks, counts = [], [], []
    for h in range(3**n):
        remaining = h
        population_mask = observed = local_mask = count = 0
        for j, v in enumerate(global_vertices):
            digit = remaining % 3
            remaining //= 3
            if digit:
                population_mask |= 1 << v
                observed |= (digit-1) << v
                local_mask |= 1 << j
                count += 1
        matched = [(y, w) for y, w in enumerate(weights) if w and y & population_mask == observed]
        P = sum(w for y, w in matched)
        if P:
            ones = [sum(w for y, w in matched if y & (1 << v)) for v in range(N)]
            costs.append(Fraction(sum(A*(P-A) for A in ones), N*total*P))
        else:
            costs.append(Fraction(0))
        masks.append(local_mask)
        counts.append(count)
    return costs, masks, counts


def rational_optima(d: int, pool: tuple[int, ...], eta: Fraction):
    cost, masks, counts = rational_history_costs(d, pool, eta)
    n = 2*len(pool)
    powers = [3**j for j in range(n)]
    F = [[None]*len(cost) for _ in range(n+1)]
    for terminal_set in range(1 << n):
        positions = [j for j in range(n) if terminal_set & (1 << j)]
        k = len(positions)
        mapping = [0]
        for j in positions:
            mapping = mapping+[h+powers[j] for h in mapping]+[h+2*powers[j] for h in mapping]
        marginal = [Fraction(0)]*len(mapping)
        for local_h in range(len(mapping)-1, -1, -1):
            absent = next((3**j for j in range(k) if local_h//(3**j) % 3 == 0), None)
            value = cost[mapping[local_h]] if absent is None else marginal[local_h+absent]+marginal[local_h+2*absent]
            marginal[local_h] = value
            old = F[k][mapping[local_h]]
            if old is None or value < old:
                F[k][mapping[local_h]] = value
    fixed, batch, adaptive = [], [], []
    for b in range(n+1):
        fixed.append(F[b][0])
        sums = {}
        for h in range(len(cost)):
            if counts[h] <= b:
                sums[masks[h]] = sums.get(masks[h], Fraction(0))+F[b][h]
        batch.append(min(sums.values()))
        values = [None]*len(cost)
        for h in range(len(cost)-1, -1, -1):
            if counts[h] == b:
                values[h] = cost[h]
            elif counts[h] < b:
                values[h] = min(values[h+p]+values[h+2*p] for p in powers if h//p % 3 == 0)
        adaptive.append(values[0])
    return {"fixed": fixed, "two_batch": batch, "adaptive": adaptive}
