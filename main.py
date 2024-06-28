
import multiprocessing
from tools import handle_define, handle_evaluate, handle_gen
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool


#primero: Identifico la operacion que la linea del texto debe realizar
def process_line(line):
    if line.startswith("/"):
        tokens = line[1:].split(" ")
        handle_buildin_function(tokens)
        print("voy a ejecutar uno de los comandos reservados")

    elif "=" in line:
        handle_define(line)
    else:
        return handle_evaluate(line)
    
#segundo: Reviso si la operacion es una funcion reservada y de ser asi la ejecuto
def handle_buildin_function(tokens):
    buildin_function = tokens[0]
    assert buildin_function in buildin_functions, "no conozco esta funcion"
    fun = buildin_functions[buildin_function]
    result = fun(*tokens[1:])
    return result

#tercero: Si el comando es load entonces defino una funcion nueva o evaluo una ya definida en un punto especifico
def handle_load(path, *args):
    print(path)
    with open(file=path) as file:
        for ix, line in enumerate(file.readlines()):
            if "=" in line:
                handle_define(line)
            else:
                result = handle_evaluate(line)
                print(f'line{ix}:{result}')

#Funciones reservadas
buildin_functions = {
    "gen": handle_gen,
    "load": handle_load
}


if __name__ == "__main__":
    lines = [
        "/load input.txt",
        "/gen f"
    ]
    for line in lines:
        try:
            process_line(line)
        except AssertionError as err:
            print(err)

    print("el programa finalizo correctamente")
