def b_number(x, k=5):
    if abs(x - round(x)) < 10**(-k):
        return str(int(round(x)))
    return f"{x:.{k}f}"


def b_row(row, b=None, k=5):
    row_str = "  ".join(b_number(x, k) for x in row)
    if b is not None:
        return f"[ {row_str} | {b_number(b, k)} ]"
    else:
        return f"[ {row_str} ]"