from itertools import product
import pickle
from multiprocessing import Pool, cpu_count
import re

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  #for 3d plotting
import numpy as np

from const import *
from rpn_notation import evaluate_function, evaluate_postfix, shunting_yard


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
        set(BUILD_IN_FUNCTIONS) - set(CONSTANTS)
    if undefined:
        raise ValueError(f"Parametros no definidos: {', '.join(undefined)}")

    _shunting_yard = shunting_yard(tokens).tokens

    return name, params, _shunting_yard


def pre_process_expression(expression):
    matches = re.finditer(
        PATTERN,
        expression
    )

    tokens = [
        (TOKENS_TYPES(token.lastgroup), token.group())
        for token in matches
        if token.lastgroup is not None
    ]

    tokens = [
        (t_type, t_value[:-1])
        if t_type == TOKENS_TYPES.FUNCTION_CALL
        else (t_type, t_value)
        for (t_type, t_value)
        in tokens
    ]
    return tokens


def evaluate_expression(expression, vars):

    tokens = pre_process_expression(expression)
    _shunting_yard = shunting_yard(tokens).tokens

    return evaluate_postfix(_shunting_yard, vars)


def log_interaction(log_file, interaction):
    with open(log_file, 'a') as f:
        f.write(interaction + '\n')


_rango = [-3, 3, 0.01]


def _handle_gen(args):
    func_name, config_args = args[0]
    args = [*args[1:]]
    func_description = USERS_FUNCTIONS[func_name]
    return [*args, evaluate_function(func_description, args)]


def save_database():
    print("Guardando base de datos")
    print(f"Funciones definidas: {[*USERS_FUNCTIONS.keys()]}")

    with open(f'.save_functions.txt', mode='wb') as f:
        pickle.dump(USERS_FUNCTIONS, f)


# region handlers ...

def handle_define(func_def):

    func_name, func_params, func_body = parse_function_definition(func_def)

    try:
        USERS_FUNCTIONS[func_name] = dict(
            func_params=func_params,
            func_body=func_body,
            func_def=func_def
        )
        print(f"Función '{func_name}' definida exitosamente.")
    except ValueError as e:
        print(f"Error al definir la función '{func_name}': {e}")


def handle_evaluate(expression):
    try:
        return evaluate_expression(expression, {})
    except (ValueError, ZeroDivisionError) as e:
        return f"Error: {e}"


def handle_list(func_defs):
    if func_defs:
        return "Funciones definidas:\n" + "\n".join(f"{name}({', '.join(params)}) = {body}" for name, (params, body) in func_defs.items())
    else:
        return "Aun no hay funciones definidas."


def handle_gen(function_name, *args):
    save_database()

    function_description = USERS_FUNCTIONS[function_name]
    params_definition = function_description['func_params']
    body = function_description['func_body']
    func_def = function_description['func_def']

    params = []
    for _ in range(len(params_definition)):
        params.append(
            list(
                np.arange(
                    _rango[0],
                    _rango[1],
                    _rango[2]
                )
            )
        )

    result = []
    commands = [args[i] for i in range(len(args)) if args[i].startswith("--")]
    args = args[:args.index(commands[0])]
    
    with Pool(int(cpu_count())-1) as p:
        result = p.map(
            _handle_gen,
            product([(function_name, args)], *params)
        )

    np_result = np.array(result)
    #fig, ax = plt.subplots()

    # Plot some data on the Axes.
    if np_result.shape[1] == 2:
        fig, ax = plt.subplots()
        ax.plot(np_result[:, 0], np_result[:, 1])
    elif np_result.shape[1] == 3:
        if "--animate:True" in commands:
            data = np_result
            fig, ax = plt.subplots()

            # Extraer valores únicos de z
            zs = np.unique(data[:, 1])

            # Preparar la figura y los ejes
            ax.set_xlim(np_result[:, 0].min(),
                        np_result[:, 0].max())  # Límites para x
            # Límites para y (considerando el desplazamiento máximo)
            ax.set_ylim(np_result[:, 2].min(), np_result[:, 2].max())
            line, = ax.plot([], [], 'b-', lw=3)  # Línea inicial vacía

            # Título y etiquetas
            ax.set_title(func_def)
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            def init():
                """Inicializa la animación limpiando la línea."""
                line.set_data([], [])
                return line,

            def update(z):
                """Actualiza la figura para un valor de z dado."""
                # Filtrar los datos para el z actual
                filtered_data = data[data[:, 1] == z]
                x = filtered_data[:, 0]
                y = filtered_data[:, 2]
                line.set_data(x, y)  # Establecer los nuevos datos de la línea
                return line,

            # Crear la animación
            ani = FuncAnimation(
                fig,
                update,
                frames=zs,
                init_func=init,
                blit=True,
                # repeat=True,
                interval=1000/30
            )
        elif "--animate:False" in commands:
            data = np_result
            fig = plt.figure() 
            ax = fig.add_subplot(111, projection='3d')

            zs = np.array(data[:, 2])
            xs,ys = data[:, 0], data[:, 1]
            
            ax.plot3D(xs, ys, zs)

            # Título y etiquetas
            ax.set_title(func_def)
            ax.set_xlabel('x')
            ax.set_ylabel('y')

    plt.show()

# endregion handlers ...
