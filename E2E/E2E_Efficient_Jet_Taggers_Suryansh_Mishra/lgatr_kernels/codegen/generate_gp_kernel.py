"""Code-generate the hardcoded geometric product expressions from the Cayley table."""

from .cayley_table import cayley_table_by_output


def generate_gp_triton_body(x_prefix="x", y_prefix="y", out_prefix="out"):
    by_output = cayley_table_by_output()
    lines = []
    for c in range(16):
        terms = by_output[c]
        parts = []
        for a, b, sign in terms:
            if sign > 0:
                parts.append(f"{x_prefix}{a} * {y_prefix}{b}")
            else:
                parts.append(f"(-{x_prefix}{a} * {y_prefix}{b})")
        expr = " + ".join(parts)
        lines.append(f"{out_prefix}{c} = {expr}")
    return lines
