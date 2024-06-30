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

plt.style.use('ggplot')

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
_core = int(cpu_count()-1)


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

      
#region Func 2D NA

def plot_2d_NA_scatter(np_result):
    fig, ax = plt.subplots()
    ax.scatter(np_result[:, 0], np_result[:, 1])
    
    plt.show()
    
def plot_2d_NA_polar(np_result):
    fig = plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.plot(np_result[:, 0], np_result[:, 1])
    
    plt.show()

def plot_2d_NA_line(np_result):
    fig, ax = plt.subplots()
    ax.plot(np_result[:, 0], np_result[:, 1])
    
    plt.show()

#endregion Func 2D NA

#region Func 2D animated

def plot_2d_A_scatter(np_result, func_def):
    data = np_result
    fig, ax = plt.subplots()
    zs = np.unique(data[:, 1])
    ax.set_xlim(np_result[:, 0].min(), np_result[:, 0].max())
    ax.set_ylim(np_result[:, 2].min(), np_result[:, 2].max())
    line = ax.scatter([], [], s=10)
    ax.set_title(func_def)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    def init():
        line.set_offsets(np.empty((0, 2)))
        return line,

    def update(z):
        filtered_data = data[data[:, 1] == z]
        x = filtered_data[:, 0]
        y = filtered_data[:, 2]
        line.set_offsets(np.c_[x, y])
        return line,

    ani = FuncAnimation(fig, update, frames=zs, init_func=init, blit=True, interval=1000/30)
    plt.show()
    
def plot_2d_A_polar(np_result, func_def):
    data = np_result
    fig = plt.figure()
    ax = plt.subplot(111, polar=True)
    zs = np.unique(data[:, 1])
    line, = ax.plot([], [], 'b-', lw=3)
    ax.set_title(func_def)
    ax.set_xlabel('theta')
    ax.set_ylabel('R')

    def update(z):
        filtered_data = data[data[:, 1] == z]
        x = filtered_data[:, 0]
        y = filtered_data[:, 2]
        line.set_data(x, y)
        return line,

    ani = FuncAnimation(fig, update, frames=zs, blit=True, interval=1000/30)
    plt.show()
    
def plot_2d_A_line(np_result, func_def):
    data = np_result
    fig, ax = plt.subplots()
    zs = np.unique(data[:, 1])
    ax.set_xlim(np_result[:, 0].min(), np_result[:, 0].max())
    ax.set_ylim(np_result[:, 2].min(), np_result[:, 2].max())
    line, = ax.plot([], [], 'b-', lw=3)
    ax.set_title(func_def)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def init():
        line.set_data([], [])
        return line,

    def update(z):
        filtered_data = data[data[:, 1] == z]
        x = filtered_data[:, 0]
        y = filtered_data[:, 2]
        line.set_data(x, y)
        return line,

    ani = FuncAnimation(fig, update, frames=zs, init_func=init, blit=True, interval=1000/30)
    plt.show()

#endregion Func 2D animated

# region Func 3D NA

def plot_3d_NA_contour(np_result, func_def, range_update):
    dimensionFila = int(abs(range_update[1][1] - range_update[1][0])/range_update[1][2])
    dimensionColumna = int(abs(range_update[0][1] - range_update[0][0])/range_update[0][2])
    xs = np_result[:, 0].reshape((dimensionFila, dimensionColumna))
    ys = np_result[:, 1].reshape((dimensionFila, dimensionColumna))
    zs = np_result[:, 2].reshape((dimensionFila, dimensionColumna))

    fig, ax = plt.subplots()
    co = ax.contourf(xs, ys, zs, cmap='viridis', edgecolor='none')
    fig.colorbar(co, shrink=0.5, aspect=5)
    ax.set_title(func_def)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    
def plot_3d_NA_scatter3d(np_result, func_def, range_update):
    dimensionFila = int(abs(range_update[1][1] - range_update[1][0])/range_update[1][2])
    dimensionColumna = int(abs(range_update[0][1] - range_update[0][0])/range_update[0][2])
    xs = np_result[:, 0].reshape((dimensionFila, dimensionColumna))
    ys = np_result[:, 1].reshape((dimensionFila, dimensionColumna))
    zs = np_result[:, 2].reshape((dimensionFila, dimensionColumna))

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, cmap='viridis', edgecolor='none')
    ax.set_title(func_def)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    
def plot_3d_NA_surface(np_result, func_def, range_update):
    dimensionFila = int(abs(range_update[1][1] - range_update[1][0])/range_update[1][2])
    dimensionColumna = int(abs(range_update[0][1] - range_update[0][0])/range_update[0][2])
    xs = np_result[:, 0].reshape((dimensionFila, dimensionColumna))
    ys = np_result[:, 1].reshape((dimensionFila, dimensionColumna))
    zs = np_result[:, 2].reshape((dimensionFila, dimensionColumna))

    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xs, ys, zs, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title(func_def)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

#endregion Func 3D NA

#region Func 3D animated

def animate_3d(np_result):
    data = np_result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(np_result[:, 0].min(), np_result[:, 0].max())
    ax.set_ylim(np_result[:, 1].min(), np_result[:, 1].max())
    ax.set_zlim(np_result[:, 3].min(), np_result[:, 3].max())
    line, = ax.plot3D([], [], [], 'b-', lw=3)
    ts = np.unique(data[:, 2])

    def update(ts):
        filtered_data = data[data[:, 2] == ts]
        x = filtered_data[:, 0]
        y = filtered_data[:, 1]
        z = filtered_data[:, 3]
        line.set_data_3d(x, y, z)
        return line,

    ani = FuncAnimation(fig, update, frames=ts, blit=True, interval=1000/30)
    plt.show()

#endregion Func 3D animated


_Default_2d_NA = plot_2d_NA_line
_Default_2d_A = plot_2d_A_line
_Default_3d_NA = plot_3d_NA_surface

_Default_2D_animate_or_not = False



def handle_config(*args):
    global _rango 
    global _core
    global _Default_2D_animate_or_not
    global _Default_2d_NA
    global _Default_2d_A
    global _Default_3d_NA
    
    commands = [args[i] for i in range(len(args)) if args[i].startswith("--")]
    for token in commands:
        if "--range:" in token:
            new_range = token.split("--range:")[1].split(" ")[0]

            # Expresión regular
            regex = r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)"

            # Extracción de los valores usando la expresión regular
            for match in re.finditer(regex, new_range):
                _rango = [float(match.group(i+1)) for i in range(3)]
                
        if "--cores:" in token:
            max_Core = int(cpu_count()-1)
            new_core = int(token.split("--cores:")[1].split(" ")[0])
            _core = new_core if new_core <= max_Core else _core
        
        if "--animate:" in token:
            value = token.split(":")[1].split(" ")[0]
            if value == "True":
                _Default_2D_animate_or_not = True
                
        if "--kind_2d_NA:" in token:
            value = token.split("--kind_2d_NA:")[1].split(" ")[0]
            if value == "scatter":
                _Default_2d_NA = plot_2d_NA_scatter
            elif value == "polar":
                _Default_2d_NA = plot_2d_NA_polar
            elif value == "line":
                _Default_2d_NA = plot_2d_NA_line
                
        if "--kind_2d_A:" in token:
            value = token.split("--kind_2d_A:")[1].split(" ")[0]
            if value == "scatter":
                _Default_2d_A = plot_2d_A_scatter
            elif value == "polar":
                _Default_2d_A = plot_2d_A_polar
            elif value == "line":
                _Default_2d_A = plot_2d_A_line
                
        if "--kind_3d_NA:" in token:
            value = token.split("--kind_3d_NA:")[1].split(" ")[0]
            if value == "contour":
                _Default_3d_NA = plot_3d_NA_contour
            elif value == "scatter3d":
                _Default_3d_NA = plot_3d_NA_scatter3d
            elif value == "surface":
                _Default_3d_NA = plot_3d_NA_surface
  
#region handle_gen

def handle_gen(function_name, *args):
    save_database()
    
    max_Core = int(cpu_count()-1)

    function_description = USERS_FUNCTIONS[function_name]
    params_definition = function_description['func_params']
    body = function_description['func_body']
    func_def = function_description['func_def']

    params = []
    
    #updated values
    coreNumber = _core
    range_update = np.array([_rango] * len(params_definition))
    to_animate_or_not = _Default_2D_animate_or_not
    
    
    print(f"los parametros por default son {_rango} y {_core} y {_Default_2D_animate_or_not} y {_Default_2d_NA} y {_Default_2d_A} y {_Default_3d_NA}")
    
    commands = [args[i] for i in range(len(args)) if args[i].startswith("--")]
    
    for token in commands:
        if "--range:" in token:
            proto_range = token.split("--range:")[1].split(" ")[0]

            # Expresión regular
            regex = r"(?:(?P<param>[txyz])=\((?P<inicio>-?\d+(?:\.\d+)?),(?P<final>-?\d+(?:\.\d+)?)\);?)"

            # almacenar los valores extraídos
            valores = {}

            # Extracción de los valores usando la expresión regular
            for match in re.finditer(regex, proto_range):
                valores[match.group("param")] = [float(match.group("inicio")), float(match.group("final"))]

            # Creación de range_update basado en los valores extraídos
            for ix,l in enumerate(params_definition):
                if l in valores:
                    range_update[ix] = valores[l] +[range_update[ix][-1]]
        
        if "--step:" in token:
            proto_step = token.split("--step:")[1].split(" ")[0]
            
            for ix,l in enumerate(params_definition):
                range_update[ix] = list(range_update[ix][:-1]) + [float(proto_step)]
        
        if "--cores:" in token:
            proto_core = int(token.split("--cores:")[1].split(" ")[0])
            coreNumber = proto_core if proto_core <= max_Core else coreNumber
        
        if "--animate:" in token:
            value = token.split(":")[1].split(" ")[0]
            if value == "True":
                to_animate_or_not = True
            else:
                to_animate_or_not = False
            
            
            
    for i in range(len(params_definition)):
        params.append(
            list(
                np.arange(
                    range_update[i][0],
                    range_update[i][1],
                    range_update[i][2]
                )
            )
        )
        # if left is parentesiis:
        #     params[i] = params[i][1:]
        # if righ is corchete:
        #     params[i] = params[i] + [range_update[i][1]]

    result = []
    
    if len(commands)!= 0:
        args = args[:args.index(commands[0])]

    with Pool(coreNumber) as p:
        result = p.map(
            _handle_gen,
            product([(function_name, args)], *params)
        )

    np_result = np.array(result)
    #fig, ax = plt.subplots()

    # Plot some data on the Axes.
    if np_result.shape[1] == 2:
        #region plot2D not animated
        if "--kind:scatter" in commands:
            plot_2d_NA_scatter(np_result)
            
        elif "--kind:polar" in commands:
            plot_2d_NA_polar(np_result)
            
        elif "--kind:line" in commands:
            plot_2d_NA_line(np_result)
        else:
            _Default_2d_NA(np_result)
        #endregion plot2D not animated
             
    elif np_result.shape[1] == 3:
        
        if to_animate_or_not:
            
            #region plot2D animated
            
            if "--kind:scatter" in commands:
                plot_2d_A_scatter(np_result, func_def)
                
            elif "--kind:polar" in commands:
                plot_2d_A_polar(np_result, func_def)
                
            elif "--kind:line" in commands: #kind:line o plot
                plot_2d_A_line(np_result, func_def)
            else:
                _Default_2d_A(np_result, func_def)
                
            #endregion plot2D animated
        else:
            #region plot3D not animated
            if "--kind:contour" in commands:
                plot_3d_NA_contour(np_result, func_def, range_update)
                
            elif "--kind:scatter3d" in commands:
                plot_3d_NA_scatter3d(np_result, func_def, range_update)
                
            elif "--kind:surface" in commands:
                plot_3d_NA_surface(np_result, func_def, range_update)
            
            else:
                _Default_3d_NA(np_result, func_def, range_update)
                
            #endregion plot3D not animated
    elif np_result.shape[1] == 4:
        data = np_result
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Preparar la figura y los ejes
        ax.set_xlim(np_result[:, 0].min(),np_result[:, 0].max())  # Límites para x
        ax.set_ylim(np_result[:, 1].min(), np_result[:, 1].max()) # Límites para y (considerando el desplazamiento máximo)
        ax.set_zlim(np_result[:, 3].min(), np_result[:, 3].max())
        line, = ax.plot3D([], [], [], 'b-', lw=3)  # Línea inicial vacía
        
        ts = np.unique(data[:, 2])
        
        def update(ts):
                """Actualiza la figura para un valor de ts dado."""
                # Filtrar los datos para el z actual
                filtered_data = data[data[:, 2] == ts]
                x = filtered_data[:, 0]
                y = filtered_data[:, 1]
                z = filtered_data[:, 3]
                line.set_data_3d(x, y, z)  # Establecer los nuevos datos de la línea
                return line,
            
        #animación
        ani = FuncAnimation(
            fig,
            update,
            frames=ts,
            blit=True,
            # repeat=True,
            interval=1000/30
        )
        

        plt.show()
#endregion handle_gen
# endregion handlers ...
