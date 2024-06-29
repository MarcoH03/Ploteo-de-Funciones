
from tools import handle_define, handle_evaluate, handle_gen

import matplotlib

matplotlib.use("TkAgg")


def handle_load(path, *args):
    print(path)
    with open(file=path) as file:
        lines = file.readlines()
        for ix, line in enumerate(lines):
            print(f'line{ix}:{line}')
            if "=" in line:
                handle_define(line)
            else:
                result = handle_evaluate(line)
                print(f'line{ix}:{result}')


build_in_functions = {
    "gen": handle_gen,
    "load": handle_load
}


def handle_build_in_function(tokens):
    tokens = [*tokens]
    build_in_function = tokens.pop(0)
    assert build_in_function in build_in_functions, "no conozco esta funcion"
    func = build_in_functions[build_in_function]
    result = func(*tokens)
    return result


def process_line(line):
    if line.startswith("/"):
        tokens = line[1:].split(" ")
        handle_build_in_function(tokens)
        print("voy a ejecutar uno de los comandos reservados")

    elif "=" in line:
        handle_define(line)
    else:
        return handle_evaluate(line)


if __name__ == "__main__":
    lines = [
        "/load input.txt",
        # "f(6)",
        "/gen f --kind=plane",
        "/gen j",
        # 'sin(0)'

    ]
    for line in lines:
        try:
            print(process_line(line))
        except AssertionError as err:
            print(err)

    print("el programa finalizo correctamente")
