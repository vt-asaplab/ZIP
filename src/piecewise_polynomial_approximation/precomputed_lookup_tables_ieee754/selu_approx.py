import torch
import torch.nn as nn

intervals = [-30.0, -7.5, -3.75, -0.46875, 0.0, 30.0]
coefficients = [[-1.7205771226278586, 0.012660480843849742, 0.0017168023593436368, 0.00011995813439253591, 4.567088890446612e-06, 9.0095392507988e-08, 7.21557954564e-10], [-0.5022733140475331, 0.9757194740465867, 0.3310536077803436, 0.06214933723586095, 0.006748607711799715, 0.0003989734088576161, 9.975013170419291e-06], [-0.002072580288248131, 1.745913620631212, 0.8512459685470609, 0.2606814868913774, 0.05187005934899104, 0.006142532632646231, 0.00032592866344188277], [0.0, 1.7569230090845935, 0.8629135128075367, 0.23259207428718534, 0.0, 0.0, 0.0], [4.4401999999999997e-16, 1.0507009873554802, 9.535999999999999e-18, 0.0, 0.0, 0.0, 0.0]]

def approx_selu(x: torch.Tensor) -> torch.Tensor:
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
