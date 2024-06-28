import pickle
from multiprocessing import Pool
import re
import operator
import math
from collections import deque

from matplotlib import pyplot as plt
import numpy as np

# Diccionario de operadores con su precedencia, asociatividad y funcion correspondiente
OPERATORS = {
    '+': (1, 'L', operator.add),
    '-': (1, 'L', operator.sub),
    '*': (2, 'L', operator.mul),
    '/': (2, 'L', operator.truediv),
    '^': (3, 'R', operator.pow),
    'neg': (4, 'R', operator.neg)
}

# Diccionario de funciones matematicas
FUNCTIONS = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'cot': lambda x: 1 / math.tan(x),
    'log': math.log,
    'log10': math.log10,
    'sqrt': math.sqrt
}

# Diccionario de constantes matematicas
CONSTANTS = {
    'pi': math.pi,
    'e': math.e
}

#funciones definidas por el usuario
func_defs = {}

#primero: Genera los puntos a plotear y distribulle la tarea de calcularlos en varios procesos
def handle_gen(func, *args):
    save_database()

    (args, _) = func_defs[func]
    params = []
    for _ in range(len(args)):
        params.append(
            list(
                np.arange(
                    _rango[0],
                    _rango[1],
                    _rango[2]
                )
            )
        )

    if len(args) == 1:
        params_1 = params[0]

        result = []
        with Pool(5) as p:
            result = p.map(
                _handle_gen,
                [
                    (func, param_1)
                    for param_1 in params_1
                ]
            )
        result = list(zip(params_1, result))

        fig, ax = plt.subplots()
        np_result = np.array(result)
        # Plot some data on the Axes.
        ax.plot(np_result[:, 0], np_result[:, 1])
        plt.show()

#segundo: Actualiza la base de datos de funciones definidas por el usuario
def save_database():
    with open(f'.save_functions.txt', mode='wb') as f:
        pickle.dump(func_defs, f)

#segundo: Carga la base de datos de funciones definidas por el usuario
def load_database():
    global func_defs
    with open(f'.save_functions.txt', mode='rb') as f:
        func_defs = pickle.load(f)

#tercero: recibe los parametros de la funcion en un punto y los evalua 
def _handle_gen(args):
    func_name = args[0]
    args = [*args[1:]]
    print(f"Generando grafico de {func_name} con parametros {args}")
    load_database()

    func_desciption = func_defs[func_name]

    return evaluate_function(func_desciption, args)

#cuarto: Ordena lo que recive de _handle_gen y lo manda a evaluar
def evaluate_function(func, args):
    params, body = func
    if len(params) != len(args):
        raise ValueError("Numero incorrecto de argumentos")
    return evaluate_expression(body, dict(zip(params, args)))

#quinto: Boberia 
def evaluate_expression(expression, vars):

    return evaluate_postfix(expression, vars)

#sexto: Evalua la funcion en el punto y devuelve el numero
def evaluate_postfix(postfix, vars):
    stack = []
    while postfix:
        token = postfix.popleft()
        if isinstance(token, float):
            stack.append(token)
        elif isinstance(token, tuple):
            func_name, arg_count = token
            args = [stack.pop() for _ in range(arg_count)]
            if func_name in func_defs:
                stack.append(evaluate_function(
                    func_defs[func_name], args[::-1]))
            else:
                stack.append(FUNCTIONS[func_name](*args[::-1]))
        elif token in vars:
            stack.append(vars[token])
        elif token in CONSTANTS:
            stack.append(CONSTANTS[token])
        elif token in OPERATORS:
            if len(stack) < 2 and token != 'neg':
                raise ValueError(
                    f"Error: no hay suficientes operandos para el operador '{token}'")
            b = stack.pop()
            a = stack.pop() if token != 'neg' else 0
            stack.append(OPERATORS[token][2](a, b))
        else:
            raise ValueError(f"Token desconocido: {token}")

    return stack[0]

#2 primero: Primer paso para definir una nueva funcion. Manda la funcion a decomponer en trozos trabajables
def handle_define(func_def):
    try:
        func_name, func_params, func_body = parse_function_definition(func_def)
        func_defs[func_name] = (func_params, func_body)
        return f"Funcion '{func_name}' definida exitosamente."
    except ValueError as e:
        return f"Error: {e}"

#2 segundo: Descompone la funcion en trozos trabajables, identifica lo elemento por lo que son y los manda al Shyard
def parse_function_definition(func_def):
    match = re.match(
        r'^\s*([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*=\s*(.+)\s*$', func_def)
    if not match:
        raise ValueError("Definicion de funcion invalida")
    name, params, body = match.groups()
    params = [p.strip() for p in params.split(',')] if params else []

    tokens = pre_process_expression(body)
    idents = {text for t_name, text in tokens if t_name == 'ident'}

    undefined = idents - set(params) - set(OPERATORS) - \
        set(FUNCTIONS) - set(CONSTANTS)
    if undefined:
        raise ValueError(f"Parametros no definidos: {', '.join(undefined)}")

    _shunting_yard = shunting_yard(tokens)

    return name, params, _shunting_yard

#2 tercero: Identifica los elementos del cuerpo por lo que son (numeros operaciones funciones etc)
def pre_process_expression(expression):
    matches = re.finditer(
        r'(?P<float>-?\d+\.\d+)|(?P<int>-?\d+)|(?P<function_call>\w+\()|(?P<LPAR>\()|(?P<RPAR>\))|(?P<ident>\w+)|(?P<symbol>[+\-*/^])|(?P<args_separator>,)|(?:\s*)',
        expression
    )

    tokens = [
        (token.lastgroup, token.group())
        for token in matches
        if token.lastgroup is not None
    ]
    return tokens

#2 cuarto: Devuelve la forma arreglada del cuerpo de la funcion 
def shunting_yard(tokens):
    output, ops = deque(), []
    arg_count_stack = []

    for t_name, token in tokens:
        if t_name in {'float', 'int'}:
            output.append(float(token))
        elif t_name == 'ident':
            output.append(token)
        elif t_name == 'function_call':
            func_name = token[:-1]
            if func_name in FUNCTIONS or func_name in func_defs:
                ops.append(func_name)
                arg_count_stack.append(0)
                ops.append('(')
            else:
                print([*func_defs.keys()])
                raise ValueError(f"Token desconocido: {func_name}")
        elif token == ',':
            while ops and ops[-1] != '(':
                output.append(ops.pop())
            if arg_count_stack:
                arg_count_stack[-1] += 1
        elif token in OPERATORS:
            if (not output or output[-1] in OPERATORS or output[-1] in {'(', ')'}) and token == '-':
                token = 'neg'
            while (ops and ops[-1] in OPERATORS and
                    ((OPERATORS[token][1] == 'L' and OPERATORS[token][0] <= OPERATORS[ops[-1]][0]) or
                     (OPERATORS[token][1] == 'R' and OPERATORS[token][0] < OPERATORS[ops[-1]][0]))):
                output.append(ops.pop())
            ops.append(token)
        elif token == '(':
            ops.append(token)
        elif token == ')':
            while ops and ops[-1] != '(':
                output.append(ops.pop())
            ops.pop()
            if ops and ops[-1] not in OPERATORS and ops[-1] not in {'(', ')'}:
                func = ops.pop()
                arg_count = arg_count_stack.pop() + 1
                output.append((func, arg_count))

    while ops:
        output.append(ops.pop())

    return output


#3 primero: Trata de evaluar la exprecion y salta error si hay un problema
def handle_evaluate(expression):
    try:
        result = full_evaluate_expression(expression, {})
        return result
    except (ValueError, ZeroDivisionError) as e:
        return f"Error: {e}"

#3 segundo: Ordena la eprecion a evaluar y la manda a calcular 
def full_evaluate_expression(expression, vars):

    tokens = pre_process_expression(expression)
    _shunting_yard = shunting_yard(tokens)

    return evaluate_postfix(_shunting_yard, vars)



def log_interaction(log_file, interaction):
    with open(log_file, 'a') as f:
        f.write(interaction + '\n')


def handle_list(func_defs):
    if func_defs:
        return "Funciones definidas:\n" + "\n".join(f"{name}({', '.join(params)}) = {body}" for name, (params, body) in func_defs.items())
    else:
        return "Aun no hay funciones definidas."


_rango = [-5, 5, 0.01]


def test__handle_gen():
    _handle_gen('f', 1)
