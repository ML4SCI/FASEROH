from typing import List

import numpy as np
from sympy import lambdify


def expr_to_func(sympy_expr, variables: List[str]):
    def cot(x):
        return 1 / np.tan(x)

    def acot(x):
        return 1 / np.arctan(x)

    def coth(x):
        return 1 / np.tanh(x)

    return lambdify(
        variables,
        sympy_expr,
        modules=["numpy", {"cot": cot, "acot": acot, "coth": coth}],
    )
