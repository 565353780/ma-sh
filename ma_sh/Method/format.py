def formatFloat(x):
    if abs(x) < 1e-3 or abs(x) > 1e3:
        return f"{x:.3e}"
    else:
        return f"{x:.3f}"
