"""Code-generate the sparse equivariant linear map structure."""

from .cayley_table import compute_linear_basis_sparse


def get_linear_basis_grouped_by_output():
    bases = compute_linear_basis_sparse()
    grouped = {i: [] for i in range(16)}
    for basis_idx, mapping in enumerate(bases):
        for (inp, out), weight in mapping.items():
            grouped[out].append((inp, basis_idx, weight))
    return grouped


def get_linear_basis_grouped_by_input():
    bases = compute_linear_basis_sparse()
    grouped = {i: [] for i in range(16)}
    for basis_idx, mapping in enumerate(bases):
        for (inp, out), weight in mapping.items():
            grouped[inp].append((out, basis_idx, weight))
    return grouped
