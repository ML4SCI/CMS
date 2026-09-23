"""Generate the Cayley (multiplication) table for the Clifford algebra Cl(1,3).

Cl(1,3) is the geometric algebra of Minkowski spacetime with metric
signature (+,-,-,-). It has 2^4 = 16 basis elements (multivector components):

  Grade 0: 1 (scalar)                         -> index 0
  Grade 1: e0, e1, e2, e3 (vectors)           -> indices 1..4
  Grade 2: e01, e02, e03, e12, e13, e23        -> indices 5..10
  Grade 3: e012, e013, e023, e123              -> indices 11..14
  Grade 4: e0123 (pseudoscalar)                -> index 15

Metric: e0^2 = +1, e1^2 = -1, e2^2 = -1, e3^2 = -1
"""

from __future__ import annotations

from dataclasses import dataclass

BASIS_LABELS = [
    "",
    "0", "1", "2", "3",
    "01", "02", "03", "12", "13", "23",
    "012", "013", "023", "123",
    "0123",
]

METRIC = {0: 1, 1: -1, 2: -1, 3: -1}
GRADE_RANGES = [(0, 1), (1, 5), (5, 11), (11, 15), (15, 16)]
GRADE_SIZES = [1, 4, 6, 4, 1]
INNER_PRODUCT_SIGNS = [1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1]


def _canonical_sign_and_indices(indices):
    lst = list(indices)
    sign = 1
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
                sign *= -1
    return sign, tuple(lst)


def _reduce_basis(indices):
    sign = 1
    lst = list(indices)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(lst) - 1:
            if lst[i] == lst[i + 1]:
                sign *= METRIC[lst[i]]
                lst.pop(i)
                lst.pop(i)
                changed = True
            else:
                i += 1
    return sign, tuple(lst)


def _label_to_index(label):
    return BASIS_LABELS.index(label)


def _indices_to_label(indices):
    return "".join(str(i) for i in indices)


def _label_to_indices(label):
    if label == "":
        return ()
    return tuple(int(c) for c in label)


@dataclass
class CayleyEntry:
    a: int
    b: int
    c: int
    sign: int


def compute_cayley_table():
    entries = []
    for a_idx in range(16):
        for b_idx in range(16):
            a_indices = _label_to_indices(BASIS_LABELS[a_idx])
            b_indices = _label_to_indices(BASIS_LABELS[b_idx])
            combined = a_indices + b_indices
            sort_sign, sorted_indices = _canonical_sign_and_indices(combined)
            reduce_sign, reduced = _reduce_basis(sorted_indices)
            total_sign = sort_sign * reduce_sign
            if total_sign != 0:
                c_label = _indices_to_label(reduced)
                if c_label in BASIS_LABELS:
                    c_idx = _label_to_index(c_label)
                    entries.append(CayleyEntry(a=a_idx, b=b_idx, c=c_idx, sign=total_sign))
    return entries


def cayley_table_by_output():
    table = compute_cayley_table()
    by_output = {i: [] for i in range(16)}
    for e in table:
        by_output[e.c].append((e.a, e.b, e.sign))
    return by_output


def compute_linear_basis_sparse():
    """Compute the sparse structure of the 10 equivariant linear basis elements.

    For SO+(1,3), the 10 basis elements are:
      0..4: grade projections
      5..9: grade projection * pseudoscalar (RIGHT multiply by e0123, matching lgatr)
    """
    grade_starts = [0, 1, 5, 11, 15]
    grade_ends = [1, 5, 11, 15, 16]
    bases = []

    for k in range(5):
        mapping = {}
        for i in range(grade_starts[k], grade_ends[k]):
            mapping[(i, i)] = 1.0
        bases.append(mapping)

    cayley = compute_cayley_table()
    pseudo_mult = {}
    for entry in cayley:
        if entry.b == 15:  # RIGHT multiply by pseudoscalar
            pseudo_mult[entry.a] = (entry.c, entry.sign)

    for k in range(5):
        mapping = {}
        for i in range(grade_starts[k], grade_ends[k]):
            if i in pseudo_mult:
                out_idx, sign = pseudo_mult[i]
                mapping[(i, out_idx)] = float(sign)
        bases.append(mapping)

    return bases
