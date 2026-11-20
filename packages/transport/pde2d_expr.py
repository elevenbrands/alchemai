"""Safe text expressions for the 2-D field solver (`packages.transport.pde2d`).

Same philosophy as `packages.control.nmpc.parse_expression` (AST whitelist, never eval/exec, no attribute access, no
subscripts, no comparisons) with a larger function table (two-argument functions, erf, abs, step) and the constant pi.
Evaluation is vectorised with numpy.
"""
from __future__ import annotations

import ast
import math

import numpy as np
from scipy import special

MAX_EXPR_CHARS = 2000
MAX_EXPR_DEPTH = 60

FUNCS1 = {"exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tan": np.tan,
          "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
          "abs": np.abs, "erf": special.erf, "erfc": special.erfc,
          "step": lambda x: np.where(np.asarray(x) > 0.0, 1.0, 0.0)}
FUNCS2 = {"min": np.minimum, "max": np.maximum, "atan2": np.arctan2, "pow": np.power}
CONSTANTS = {"pi": math.pi}
_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
ALLOWED_TEXT = ("números, nombres declarados, pi, + − * / ** (o ^), signo y las funciones "
                + ", ".join(sorted(FUNCS1)) + " (un argumento) y " + ", ".join(sorted(FUNCS2)) + " (dos argumentos)")
RESERVED = set(FUNCS1) | set(FUNCS2) | set(CONSTANTS)


class Expr:
    """A checked expression: `names` are the declared names it uses; call it with a dict name -> float/ndarray."""

    def __init__(self, text, names: set[str], where: str):
        if isinstance(text, bool) or text is None:
            raise ValueError(f"{where}: falta la expresión (texto o número)")
        if isinstance(text, (int, float)):
            if not math.isfinite(text):
                raise ValueError(f"{where}: el número debe ser finito")
            text = repr(float(text))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{where}: falta la expresión (texto o número)")
        if len(text) > MAX_EXPR_CHARS:
            raise ValueError(f"{where}: expresión demasiado larga (máximo {MAX_EXPR_CHARS} caracteres)")
        try:
            tree = ast.parse(text.strip().replace("^", "**"), mode="eval")
        except SyntaxError as ex:
            raise ValueError(f"{where}: expresión con sintaxis inválida ({ex.msg}): '{text}'") from None
        self.text = text
        self.where = where
        self.names: set[str] = set()
        self._allowed = names
        self._check(tree.body, 0)
        self._node = tree.body

    def _check(self, node, depth):
        w = self.where
        if depth > MAX_EXPR_DEPTH:
            raise ValueError(f"{w}: expresión demasiado anidada")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"{w}: solo se admiten constantes numéricas")
        elif isinstance(node, ast.Name):
            if node.id in CONSTANTS:
                return
            if node.id not in self._allowed:
                raise ValueError(f"{w}: el nombre '{node.id}' no está declarado aquí; nombres válidos: "
                                 + ", ".join(sorted(self._allowed)))
            self.names.add(node.id)
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            self._check(node.left, depth + 1)
            self._check(node.right, depth + 1)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            self._check(node.operand, depth + 1)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise ValueError(f"{w}: llamada no permitida; se admiten {ALLOWED_TEXT}")
            n = 1 if node.func.id in FUNCS1 else 2 if node.func.id in FUNCS2 else 0
            if n == 0 or len(node.args) != n:
                raise ValueError(f"{w}: función '{node.func.id}' no admitida o con número de argumentos incorrecto; "
                                 f"se admiten {ALLOWED_TEXT}")
            for a in node.args:
                self._check(a, depth + 1)
        else:
            raise ValueError(f"{w}: elemento no permitido en la expresión ({type(node).__name__}); se admiten "
                             f"{ALLOWED_TEXT}")

    def _eval(self, node, env):
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return CONSTANTS[node.id] if node.id in CONSTANTS else env[node.id]
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](self._eval(node.left, env), self._eval(node.right, env))
        if isinstance(node, ast.UnaryOp):
            v = self._eval(node.operand, env)
            return -v if isinstance(node.op, ast.USub) else v
        args = [self._eval(a, env) for a in node.args]
        return (FUNCS1.get(node.func.id) or FUNCS2[node.func.id])(*args)

    def __call__(self, env: dict, shape=None):
        """Evaluate; with `shape` the result is broadcast to that shape (constants become full arrays)."""
        with np.errstate(all="ignore"):
            v = self._eval(self._node, env)
        if shape is not None:
            v = np.broadcast_to(np.asarray(v, dtype=float), shape).copy()
        return v
