"""Safe compiler of user-written mathematical expressions into numpy closures.

Extends the idea of `packages.control.nmpc.parse_expression` (an `ast` whitelist, nothing is ever passed to eval/exec, no
attribute access, no subscripts, no lambdas) with what general ODE/BVP/PDE right-hand sides need:

  * functions of one argument: exp log log10 sqrt sin cos tan sinh cosh tanh asin acos atan abs sign erf erfc floor,
    heaviside(x) (0 for x < 0, 0.5 at 0, 1 for x > 0);
  * functions of several arguments: min(a, b, ...), max(a, b, ...), atan2(y, x), pow(a, b), where(cond, a, b);
  * comparisons < <= > >= (one operator), worth 1.0 when true and 0.0 when false, so switches can be written as
    `(t > 5)*q` or `where(T > Tc, a, b)`;
  * the constant `pi` (unless the user declares a name `pi`); `^` is accepted as `**`.

The tree is checked once and turned into nested closures, so each evaluation costs only numpy calls on whole arrays
(vectorised over the grid or over the collocation nodes)."""
from __future__ import annotations

import ast
import keyword
import math
from typing import Callable

import numpy as np
from scipy import special

from packages.control.nmpc import MAX_EXPR_CHARS, MAX_EXPR_DEPTH

_UNARY = {"exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tan": np.tan,
          "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
          "abs": np.abs, "sign": np.sign, "erf": special.erf, "erfc": special.erfc, "floor": np.floor,
          "heaviside": lambda x: np.heaviside(x, 0.5)}
_NARY = {"min": (2, None), "max": (2, None), "atan2": (2, 2), "pow": (2, 2), "where": (3, 3)}
FUNCTIONS = tuple(_UNARY) + tuple(_NARY)
CONSTANTS = {"pi": math.pi}
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
_CMP = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal}
ALLOWED_TEXT = ("números, nombres declarados, + - * / ** (o ^), paréntesis, comparaciones < <= > >= (valen 1 o 0), la "
                "constante pi y las funciones " + ", ".join(FUNCTIONS))


def check_name(name: str, what: str) -> str:
    """Validate an identifier chosen by the user (variable, parameter, definition)."""
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"{what}: '{name}' no es un nombre válido (letras, dígitos y _, sin empezar por dígito)")
    if name in FUNCTIONS:
        raise ValueError(f"{what}: '{name}' es el nombre de una función y no puede usarse como nombre")
    return name


def _reduce(func, fns):
    def f(env):
        out = fns[0](env)
        for g in fns[1:]:
            out = func(out, g(env))
        return out
    return f


def _build(node: ast.AST, names: set[str], where: str, depth: int) -> Callable[[dict], object]:
    if depth > MAX_EXPR_DEPTH:
        raise ValueError(f"{where}: expresión demasiado anidada")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"{where}: solo se admiten constantes numéricas")
        value = float(node.value)
        return lambda env: value
    if isinstance(node, ast.Name):
        key = node.id
        if key in names:
            return lambda env: env[key]
        if key in CONSTANTS:
            value = CONSTANTS[key]
            return lambda env: value
        raise ValueError(f"{where}: el nombre '{key}' no está declarado; nombres disponibles: "
                         f"{', '.join(sorted(names)) or '(ninguno)'}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        op, a, b = _BIN[type(node.op)], _build(node.left, names, where, depth + 1), _build(node.right, names, where, depth + 1)
        return lambda env: op(a(env), b(env))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        a = _build(node.operand, names, where, depth + 1)
        return (lambda env: -a(env)) if isinstance(node.op, ast.USub) else a
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP:
        op, a, b = _CMP[type(node.ops[0])], _build(node.left, names, where, depth + 1), \
            _build(node.comparators[0], names, where, depth + 1)
        return lambda env: op(a(env), b(env)) * 1.0
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
            called = node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__
            raise ValueError(f"{where}: función no admitida '{called}'; se admiten {', '.join(FUNCTIONS)}")
        fname, args = node.func.id, [_build(a, names, where, depth + 1) for a in node.args]
        if fname in _UNARY:
            if len(args) != 1:
                raise ValueError(f"{where}: {fname}() lleva exactamente un argumento")
            func, a = _UNARY[fname], args[0]
            return lambda env: func(a(env))
        lo, hi = _NARY[fname]
        if len(args) < lo or (hi is not None and len(args) > hi):
            raise ValueError(f"{where}: {fname}() lleva {lo if hi == lo else f'al menos {lo}'} argumentos")
        if fname == "min":
            return _reduce(np.minimum, args)
        if fname == "max":
            return _reduce(np.maximum, args)
        if fname == "atan2":
            return lambda env: np.arctan2(args[0](env), args[1](env))
        if fname == "pow":
            return lambda env: np.power(args[0](env), args[1](env))
        return lambda env: np.where(args[0](env) != 0, args[1](env), args[2](env))
    raise ValueError(f"{where}: elemento no permitido en la expresión ({type(node).__name__}); se admiten {ALLOWED_TEXT}")


def compile_expression(text, names: set[str], where: str) -> Callable[[dict], object]:
    """Compile one expression (text, or a plain number) into `f(env) -> float | ndarray`; ValueError in Spanish.

    `env` maps every declared name to a float or to arrays that broadcast together."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        if not math.isfinite(text):
            raise ValueError(f"{where}: el número debe ser finito")
        value = float(text)
        return lambda env: value
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{where}: falta la expresión (texto o número)")
    if len(text) > MAX_EXPR_CHARS:
        raise ValueError(f"{where}: expresión demasiado larga (máximo {MAX_EXPR_CHARS} caracteres)")
    try:
        tree = ast.parse(text.strip().replace("^", "**"), mode="eval")
    except SyntaxError as ex:
        raise ValueError(f"{where}: expresión con sintaxis inválida ({ex.msg}): «{text}»") from None
    return _build(tree.body, set(names), f"{where} «{text}»", 0)


def parse_parameters(parameters: dict | None, where: str = "parameters") -> dict[str, float]:
    """Validate {name: finite number}."""
    out: dict[str, float] = {}
    if parameters is None:
        return out
    if not isinstance(parameters, dict):
        raise ValueError(f"{where} debe ser un diccionario {{nombre: número}}")
    for k, v in parameters.items():
        check_name(k, where)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{where}: el valor de '{k}' debe ser un número finito")
        out[k] = float(v)
    return out


def declare(groups: list[tuple[str, list[str]]]) -> set[str]:
    """Union of the name groups [(label, names)], refusing duplicates across groups."""
    seen: dict[str, str] = {}
    for label, names in groups:
        for n in names:
            check_name(n, label)
            if n in seen:
                raise ValueError(f"el nombre '{n}' está repetido ({seen[n]} y {label}); cada nombre debe ser único")
            seen[n] = label
    return set(seen)


def compile_definitions(definitions: dict | None, names: set[str], where: str = "definitions"):
    """Intermediate algebraic quantities {name: expression}, evaluated in the given order; each may use the previous ones.

    Returns (list of (name, closure), names extended with the definitions)."""
    out, names = [], set(names)
    if definitions is None:
        return out, names
    if not isinstance(definitions, dict):
        raise ValueError(f"{where} debe ser un diccionario {{nombre: expresión}} (se evalúan en ese orden)")
    for k, text in definitions.items():
        check_name(k, where)
        if k in names:
            raise ValueError(f"{where}: el nombre '{k}' ya está en uso")
        out.append((k, compile_expression(text, names, f"{where}['{k}']")))
        names.add(k)
    return out, names


def apply_definitions(defs, env: dict) -> dict:
    """Evaluate the compiled definitions into `env` (in place) and return it."""
    for k, f in defs:
        env[k] = f(env)
    return env
