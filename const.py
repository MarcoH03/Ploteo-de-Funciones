from enum import Enum
import math
import operator
import os
import pickle
import re


class TOKENS_TYPES(Enum):
    FLOAT = "float"
    INT = "int"
    FUNCTION_CALL = "function_call"
    BUILD_IN_FUNCTION_CALL = "build_in_function_call"
    LPAR = "LPAR"  # "("
    RPAR = "RPAR"  # ")"
    IDENT = "ident"
    SYMBOL = "symbol"
    ARGS_SEPARATOR = "args_separator"  # ","


PATTERN = re.compile(
    r'(?P<float>\d+\.\d+)|'
    r'(?P<int>\d+)|'
    r'(?P<function_call>\w+\()|'
    r'(?P<LPAR>\()|(?P<RPAR>\))|'
    r'(?P<ident>\w+)|(?P<symbol>[+\-*/^])|'
    r'(?P<args_separator>,)|'
    r'(?:\s*)'
)


# Diccionario de operadores con su precedencia, asociativa y función correspondiente
OPERATORS = {
    '+': (1, 'L', 2, operator.add),
    '-': (1, 'L', 2, operator.sub),
    '*': (2, 'L', 2, operator.mul),
    '/': (2, 'L', 2, operator.truediv),
    '^': (3, 'R', 2, operator.pow),
    'neg': (4, 'R', 1, operator.neg)
}


USERS_FUNCTIONS = {}
if os.path.exists('.save_functions.txt'):
    with open(f'.save_functions.txt', mode='rb') as f:
        USERS_FUNCTIONS = pickle.load(f)


# Diccionario de funciones matematicas
BUILD_IN_FUNCTIONS = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'cot': lambda x: 1 / math.tan(x),
    'log': math.log,
    'log10': math.log10,
    'sqrt': math.sqrt,
    "exp": math.exp,
}


OPERATORS = {
    **OPERATORS,
    **{
        fname: (4, 'R', 1, function)
        for fname, function in BUILD_IN_FUNCTIONS.items()
    }
}

CONSTANTS = {
    'pi': math.pi,
    'e': math.e
}
