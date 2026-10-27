"""Shared arithmetic-trace helpers for solvers.

Ported from Tong Huikang's `reasoners/store_types.py`. Generates the step-by-step
long-multiplication and long-division traces that show the model HOW to do the
arithmetic, not just what answer to produce. The hypothesis (per the v6-v9 arc
research findings) is that this format-rich reasoning is what got Tong's 0.85
LB: SFT teaches arithmetic only if it sees the arithmetic.

Two main entry points:
    long_multiplication_lines(a_str, b_str) -> (lines, result_str)
    long_division_lines(num_str, den_str, max_decimal_digits=3) -> (lines, quotient_str)

Plus a few decimal-string utilities (truncate_3dp, pad_dp, cast_dp_pair) used by
the trace generators.
"""

from __future__ import annotations


def _fmt_int_with_dp(value: int, dp: int) -> str:
    """Format an integer as a decimal string with *dp* decimal places."""
    if dp == 0:
        return str(value)
    s = str(value).zfill(dp + 1)
    s = s[: len(s) - dp] + "." + s[len(s) - dp :]
    s = s.lstrip("0") or "0"
    if s.startswith("."):
        s = "0" + s
    return s


def truncate_3dp(s: str) -> str:
    """Truncate a decimal string to at most 3 decimal places (no rounding)."""
    if "." not in s:
        return s
    integer, frac = s.split(".")
    if len(frac) <= 3:
        return s
    return integer + "." + frac[:3]


def _dp_count(s: str) -> int:
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def pad_dp(s: str, n: int) -> str:
    """Pad a decimal string to exactly *n* decimal places."""
    if "." not in s:
        s = s + "."
    integer, frac = s.split(".")
    return integer + "." + frac.ljust(n, "0")


def cast_dp_pair(a: str, b: str) -> tuple[str, str, int, int]:
    """Pad *a* and *b* to the same number of decimal places.

    Returns (a_padded, b_padded, target_dp, target_dp).
    """
    da, db = _dp_count(a), _dp_count(b)
    target = max(da, db)
    return pad_dp(a, target), pad_dp(b, target), target, target


def long_multiplication_lines(a_str: str, b_str: str) -> tuple[list[str], str]:
    """Generate step-by-step multiplication of two decimal numbers.

    Decomposes *b* into place-value components, multiplies *a* by each, then
    shows a running sum.

    Returns (lines, result_str) where result_str is the exact product.
    """
    a_dp = len(a_str.split(".")[1]) if "." in a_str else 0
    b_dp = len(b_str.split(".")[1]) if "." in b_str else 0
    total_dp = a_dp + b_dp

    a_int = int(a_str.replace(".", "").lstrip("0") or "0")
    b_int = int(b_str.replace(".", "").lstrip("0") or "0")

    lines: list[str] = []

    b_digits_str = str(abs(b_int))
    b_num_digits = len(b_digits_str)

    components: list[tuple[str, int, str]] = []
    for i in range(b_num_digits - 1, -1, -1):
        d = int(b_digits_str[i])
        if d == 0:
            continue
        comp_scaled = d * (10 ** (b_num_digits - 1 - i))
        comp_display = _fmt_int_with_dp(comp_scaled, b_dp)
        if b_dp > 0:
            comp_display = pad_dp(comp_display, b_dp)

        product_int = a_int * comp_scaled
        product_display = _fmt_int_with_dp(product_int, total_dp)
        if total_dp > 0:
            product_display = pad_dp(product_display, total_dp)

        components.append((comp_display, product_int, product_display))

    for comp_display, _, product_display in components:
        lines.append(f"{a_str} * {comp_display} = {product_display}")

    if len(components) >= 2:
        running = components[0][1]
        for i in range(1, len(components)):
            running_display = _fmt_int_with_dp(running, total_dp)
            if total_dp > 0:
                running_display = pad_dp(running_display, total_dp)
            running += components[i][1]
            sum_display = _fmt_int_with_dp(running, total_dp)
            if total_dp > 0:
                sum_display = pad_dp(sum_display, total_dp)
            lines.append(f"{running_display} + {components[i][2]} = {sum_display}")

    total = a_int * b_int
    result_str = _fmt_int_with_dp(total, total_dp)
    return lines, result_str


def long_division_lines(
    numerator_str: str, denominator_str: str, max_decimal_digits: int = 3
) -> tuple[list[str], str]:
    """Generate long-division steps via repeated subtraction.

    Returns (lines, result_str) where result_str is the truncated quotient.
    """
    n_dp = len(numerator_str.split(".")[1]) if "." in numerator_str else 0
    d_dp = len(denominator_str.split(".")[1]) if "." in denominator_str else 0
    max_dp = max(n_dp, d_dp)

    num = int(round(float(numerator_str) * 10**max_dp))
    den = int(round(float(denominator_str) * 10**max_dp))

    lines: list[str] = []
    acc = 0
    decimal_digits = 0

    def fmt_acc() -> str:
        if decimal_digits == 0:
            return str(acc)
        s = str(acc).zfill(decimal_digits + 1)
        return s[:-decimal_digits] + "." + s[-decimal_digits:]

    def fmt_scale() -> str:
        if decimal_digits == 0:
            return "1"
        return "0." + "0" * (decimal_digits - 1) + "1"

    def fmt_line(n: int) -> str:
        return f"= {fmt_acc()} + {fmt_scale()} * {n} / {den}"

    lines.append(fmt_line(num))

    while decimal_digits <= max_decimal_digits:
        if num >= den:
            num -= den
            acc += 1
            lines.append(fmt_line(num))
        else:
            decimal_digits += 1
            if decimal_digits > max_decimal_digits:
                break
            num *= 10
            acc *= 10
            lines.append(fmt_line(num))

    if decimal_digits > max_decimal_digits:
        decimal_digits = max_decimal_digits
    return lines, fmt_acc()
