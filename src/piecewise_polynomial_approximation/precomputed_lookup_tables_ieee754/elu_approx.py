import torch
import torch.nn as nn

intervals = [-30.0, -7.5, -3.75, -0.9375, -0.46875, 0.0, 30.0]
coefficients = [[-0.9889920442688401, 0.003047055441468161, 0.0003218915969749054, 1.6305361992392117e-05, 3.9819418424980397e-07, 3.769187046043999e-09], [-0.43563932137200295, 0.38427406179056617, 0.10846918928105001, 0.015719614659454578, 0.001160868257485965, 3.474399612868807e-05], [-0.016186147041483065, 0.9430734105827774, 0.41900302545267787, 0.10562453882432216, 0.014653210751903855, 0.0008679966608076343], [-7.533227471570239e-05, 0.999299661860994, 0.49730564770765395, 0.16114546246136074, 0.035228905661868884, 0.004134750469131741], [0.0, 0.9993309071134653, 0.49082181692396926, 0.1322974583308178, 0.0, 0.0], [0.0, 1.0, -4.7679999999999996e-18, 0.0, 0.0, 0.0]]

def approx_elu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)

    def poly_eval(z: torch.Tensor, coeffs):
        # Evaluate the polynomial sum of coeffs[k] * z^k
        result = torch.zeros_like(z)
        for power, c in enumerate(coeffs):
            result += c * (z ** power)
        return result

    num_intervals = len(intervals)
    for i in range(num_intervals - 1):
        lower = intervals[i]
        upper = intervals[i + 1]
        mask = (x >= lower) & (x < upper)
        out[mask] = poly_eval(x[mask], coefficients[i])

    # Values >= the last boundary
    mask_last = (x >= intervals[-1])
    out[mask_last] = poly_eval(x[mask_last], coefficients[-1])

    # Values < the first boundary
    mask_first = (x < intervals[0])
    out[mask_first] = poly_eval(x[mask_first], coefficients[0])

    return out
