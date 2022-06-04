"""
A class for representing and generating expressions in Polish notation

Expressions are generated with the algorithm descibed in Appendix C of DEEP LEARNING FOR SYMBOLIC MATHEMATICS (Guillaume Lample, Francois Charton 2019)
which aims to weight deep, shallow, left-leaning, and right leaning expression trees all equally

We used polish notation for this project as it can more concisely represent expressions than infix notation as it never needs parenthesis

"""
import random
import numpy as np
import sympy.core.numbers
from sympy import *


class Expression:
    """
    dictionary of operations with their corresponding arity as keys
    """
    _ops = {
        'sin': 1,
        'cos': 1,
        'tan': 1,
        'square': 1,
        'cube': 1,
        'exp': 1,
        'log': 1,

        '+': 2,
        '-': 2,
        '*': 2,
        '/': 2,
        '**': 2,
    }

    """
    Maps operations to anonymous function that generates an infix expression
    """
    _infix_reps = {
        'sin': lambda args: f'sin({args[0]})',
        'cos': lambda args: f'cos({args[0]})',
        'tan': lambda args: f'tan({args[0]})',
        'square': lambda args: f'({args[0]})**2',
        'cube': lambda args: f'({args[0]})**3',
        'exp': lambda args: f'exp({args[0]})',
        'log': lambda args: f'log({args[0]})',

        '+': lambda args: f'({args[0]})+({args[1]})',
        '-': lambda args: f'({args[0]})-({args[1]})',
        '*': lambda args: f'({args[0]})*({args[1]})',
        '/': lambda args: f'({args[0]})/({args[1]})',
        '**': lambda args: f'({args[0]})**({args[1]})'
    }
    """
    unnormalized probabilities of each unary op
    """
    _unary_op_probs = {
        'sin': 1,
        'cos': 1,
        'tan': 2,
        'square': 4,
        'cube': 3,
        'exp': 2,
        'log': 1
    }
    """
    unnormalized probabilities of each binary op
    """
    _bin_op_probs = {
        '+': 3,
        '-': 3,
        '*': 2,
        '/': 2,
        '**': 1,
    }
    """
    maps sympy functions to ones contained in this class
    """
    _from_sympy = {
        sin: 'sin',
        cos: 'cos',
        tan: 'tan',
        exp: 'exp',
        log: 'log',

        Add: '+',
        Mul: '*',
        Pow: '**',
    }
    """
      Generates a numpy array representing counts of possible trees of n internal nodes generated from e empty nodes
      D(0, n) = 0
      D(e, 0) = L ** e
      D(e, n) = L * D(e - 1, n) + p_1 * D(e, n - 1) + p_2 * D(e + 1, n - 1)
  
      from Appendix C.2 of DEEP LEARNING FOR SYMBOLIC MATHEMATICS (Guillaume Lample, Francois Charton 2019)
    """

    def _unary_binary_dist(self, size):

        # generating transposed version
        D = np.zeros((size * 2 + 1, size))
        D[:, 0] = self._num_leaves ** np.arange(size * 2 + 1)
        D[0, 0] = 0

        for n in range(1, size):
            for e in range(1, size * 2):
                D[e, n] = self._num_leaves * D[e - 1, n] + self._num_unary_ops * D[e, n - 1] + self._num_bin_ops * D[
                    e + 1, n - 1]
        return D[:, :size + 1]

    """
  
  
    Samples a position of a node and arity
  
    from Appendix C.3 of DEEP LEARNING FOR SYMBOLIC MATHEMATICS (Guillaume Lample, Francois Charton 2019)
  
    Parameters
      e -- number of empty nodes to sample from
      n -- number of operations
    """

    def _sample(self, e, n):

        P = np.zeros((e, 2))

        for k in range(e):
            P[k, 0] = (self._num_leaves ** k) * self._num_unary_ops * self._unary_binary_dist[e - k][n - 1]
        for k in range(e):
            P[k, 1] = (self._num_leaves ** k) * self._num_bin_ops * self._unary_binary_dist[e - k + 1][n - 1]

        P /= self._unary_binary_dist[e, n]
        k = np.random.choice(2 * e, p=P.T.flatten())

        arity = 1 if k < e else 2
        k = k % e
        return k, arity

    def _choose_unary_op(self):
        return np.random.choice(tuple(self._unary_op_probs.keys()), p=self._unary_op_norm_prob)

    def _choose_bin_op(self):
        return np.random.choice(tuple(self._bin_op_probs.keys()), p=self._bin_op_norm_prob)

    def _choose_leaf(self):
        if random.random() < 0.3:
            return 'x'
        return random.randrange(0, 10)

    def _gen_from_sympy(self, expr):
        self._rep = []
        stack = [expr]
        while (len(stack) != 0):
            expr = stack.pop()
            # print(expr, self._rep)
            if isinstance(expr, Symbol):
                self._rep.append(str(expr))
            elif isinstance(expr, Integer):
                self._rep.append(str(expr))
            elif isinstance(expr, Rational):
                self._rep.append('/')

                args = str(expr).split('/')
                self._rep.append(str(args[0]))
                self._rep.append(str(args[1]))
            elif expr == E:
                self._rep.append('e')
            elif expr == pi:
                self._rep.append('pi')
            elif expr == I:
                self._rep.append('i')


            else:
                for i in range(len(expr.args) - 1):
                    self._rep.append(self._from_sympy[type(expr)])

                for item in expr.args:
                    stack.append(item)

    def _gen_random(self, num_ops):
        self._num_leaves = 1
        self._num_bin_ops = len(self._bin_op_probs.keys())
        self._num_unary_ops = len(self._unary_op_probs.keys())

        self._unary_binary_dist = self._unary_binary_dist(num_ops + 1)

        self._bin_op_norm_prob = np.fromiter(self._bin_op_probs.values(), dtype=float)
        self._bin_op_norm_prob /= self._bin_op_norm_prob.sum()

        self._unary_op_norm_prob = np.fromiter(self._unary_op_probs.values(), dtype=float)
        self._unary_op_norm_prob /= self._unary_op_norm_prob.sum()

        rep = [None]
        e = 1
        skipped = 0
        for n in range(num_ops, 0, - 1):
            k, arity = self._sample(e, n)
            skipped += k
            if arity == 1:
                op = self._choose_unary_op()

                # O(N) is bad for this. TODO: change to a dynamic programming approach so it is O(1) per iteration of parent loop
                encountered_empty = 0
                pos = 0
                for i in range(len(rep)):
                    if (rep[i] == None):
                        encountered_empty += 1
                    if encountered_empty == skipped + 1:
                        pos = i
                        break

                rep = rep[:pos] + [op] + [None] + rep[pos + 1:]
                e = e - k
            else:
                op = self._choose_bin_op()

                encountered_empty = 0
                pos = 0
                for i in range(len(rep)):
                    if (rep[i] == None):
                        encountered_empty += 1
                    if encountered_empty == skipped + 1:
                        pos = i
                        break

                rep = rep[:pos] + [op] + [None] + [None] + rep[pos + 1:]
                e = e - k + 1

        for i in range(len(rep)):
            if (rep[i] is None):
                rep[i] = self._choose_leaf()
        self._rep = rep

    def __init__(self, expr=None, num_ops=None):

        if (expr is not None):
            self._gen_from_sympy(expr)
        else:
            self._gen_random(num_ops)

    def to_infix(self):
        stack = []

        for i in range(len(self._rep) - 1, -1, -1):
            token = self._rep[i]

            if token in self._ops:
                arity = self._ops[token]

                args = stack[-arity:]
                stack = stack[:-arity]

                stack.append(self._infix_reps[token](args))
            else:
                stack.append(token)
        return stack.pop()

    def get_rep(self):
        return self._rep
    def get_sympy(self):
        return parse_expr(self.to_infix())
    def get_histogram(self, interval=(0,1), bins=5):
        x = Symbol('x')
        f = self.get_sympy()
        F = integrate(f, Symbol('x'))
        ret = []
        dif = interval[1] - interval[0]
        for i in range(0, bins + 1):
            cur = interval[0] + i * dif/bins
            ret.append(F.subs(x, cur))
        for i in range(len(ret) - 1, 0, -1):
            ret[i] = (ret[i] - ret[i-1]).evalf()
            if type(ret[i]) != sympy.core.numbers.Float:
                return None
        return ret[1:]

def main():
    expr = Expression(num_ops=5)
    print(expr.get_histogram())
main()