from const import *


class RPN_Notation:
    def __init__(self, tokens):
        self.tokens = tokens


def process_function_call(function_token_description, tokens):
    token_type, function_name = function_token_description
    assert token_type == TOKENS_TYPES.FUNCTION_CALL, "Error: token no es una llamada a function"
    args = [[]]
    rpar_count = 1
    _token_type = None
    while len(tokens) > 0:

        current_arg = args[-1]

        token_description = tokens.pop(0)
        _token_type, _token_value = token_description

        if _token_type == TOKENS_TYPES.ARGS_SEPARATOR:
            assert rpar_count == 1, "Error: no se ha cerrado el paréntesis"
            args.append([])
            continue

        elif _token_type == TOKENS_TYPES.LPAR:
            rpar_count += 1

        elif _token_type == TOKENS_TYPES.RPAR:
            rpar_count -= 1
            if rpar_count == 0:
                token_type = TOKENS_TYPES.FUNCTION_CALL

                if function_name in USERS_FUNCTIONS:
                    function_description = USERS_FUNCTIONS[function_name]
                    assert len(function_description['func_params']) == len(
                        args), "Error: cantidad de parámetros incorrecta"
                else:
                    assert function_name in BUILD_IN_FUNCTIONS, "Error: función no definida"
                    assert len(
                        args) == 1, "Error: cantidad de parámetros incorrecta"
                    token_type = TOKENS_TYPES.BUILD_IN_FUNCTION_CALL
                break
        elif _token_type == TOKENS_TYPES.FUNCTION_CALL:
            current_arg.append(process_function_call(
                token_description, tokens))
            continue
        current_arg.append(token_description)

    assert _token_type == TOKENS_TYPES.RPAR, "Error: no se ha cerrado el paréntesis"
    return RPN_Notation([
        token_description
        for arg in args
        if len(arg) > 0
        for token_description in shunting_yard(arg).tokens
    ] + [(token_type, function_name)])


def shunting_yard(tokens):
    output_deque, ops_stack = [], []

    tokens_copy = [*tokens]

    while len(tokens_copy) > 0:

        ix = len(tokens) - len(tokens_copy)
        token_description = tokens_copy.pop(0)

        if isinstance(token_description, RPN_Notation):
            output_deque += token_description.tokens
            continue

        (token_type, token_value) = token_description

        assert token_type != TOKENS_TYPES.ARGS_SEPARATOR, "Error: token desconocido"

        if token_type in {
            TOKENS_TYPES.FLOAT,
            TOKENS_TYPES.INT
        }:
            output_deque.append((token_type, float(token_value)))

        elif token_type == TOKENS_TYPES.IDENT:
            output_deque.append(token_description)

        elif token_type == TOKENS_TYPES.SYMBOL:

            if token_value == '-':
                if ix == 0:
                    token_value = 'neg'

                if tokens[ix - 1][0] not in {
                    TOKENS_TYPES.FLOAT,
                    TOKENS_TYPES.INT,
                    TOKENS_TYPES.IDENT
                }:
                    token_value = 'neg'

            if len(ops_stack) == 0:
                ops_stack.append((token_type, token_value))
                continue
            last_op = ops_stack[-1]
            last_op_type, last_op_value = last_op

            if last_op_type == TOKENS_TYPES.LPAR:
                ops_stack.append(token_description)
                continue

            curr_precedence, curr_associativity, _, _ = OPERATORS[token_value]
            last_precedence, last_associativity, _, _ = OPERATORS[last_op_value]

            while len(ops_stack) > 0 and ((
                curr_associativity == 'L' and curr_precedence <= last_precedence)
                or (curr_associativity == 'R' and curr_precedence < last_precedence)
            ):
                output_deque.append(ops_stack.pop())

            ops_stack.append((token_type, token_value))

        elif token_type == TOKENS_TYPES.FUNCTION_CALL:
            output_deque.append(
                process_function_call(
                    token_description,
                    tokens_copy
                )
            )

        elif token_type == TOKENS_TYPES.LPAR:
            ops_stack.append(token_description)
        elif token_type == TOKENS_TYPES.RPAR:
            lpar_open = 1
            while len(ops_stack) > 0:
                token_last_op = ops_stack.pop()
                _token_type, _ = token_last_op
                if _token_type == TOKENS_TYPES.LPAR:
                    lpar_open -= 1
                    break
                output_deque.append(token_last_op)
            assert lpar_open == 0, "Error: no se ha cerrado el paréntesis"

    output_deque += ops_stack[::-1]

    # return RPN_Notation([output_deque])
    return RPN_Notation(
        [
            token
            for output in output_deque
            for token in (
                output.tokens
                if isinstance(output, RPN_Notation)
                else [output]
            )
        ]
    )


# Diccionario de constantes matemáticas
def evaluate_function(func, args):
    params = func['func_params']
    body = func['func_body']
    if len(params) != len(args):
        raise ValueError("Numero incorrecto de argumentos")
    return evaluate_postfix(
        body,
        {
            k: v for k, v in zip(params, args)
        })


def evaluate_postfix(rpn_notation, vars):
    postfix = [*rpn_notation]
    stack = []
    while postfix:
        token_type, token = postfix.pop(0)
        if token_type in {
            TOKENS_TYPES.FLOAT,
            TOKENS_TYPES.INT
        }:
            stack.append(token)
        elif token_type == TOKENS_TYPES.IDENT:
            if token in CONSTANTS:
                stack.append((CONSTANTS[token]))
            else:
                stack.append(vars[token])
                
        elif token_type == TOKENS_TYPES.FUNCTION_CALL:

            func_name = token
            assert func_name in USERS_FUNCTIONS, "Error: función no definida"

            function_descriptor = USERS_FUNCTIONS[func_name]
            arg_count = function_descriptor['func_params']
            args = [stack.pop() for _ in range(len(arg_count))]
            stack.append(
                evaluate_function(
                    USERS_FUNCTIONS[func_name], args[::-1]
                )
            )

        elif token_type in {
            TOKENS_TYPES.SYMBOL,
            TOKENS_TYPES.BUILD_IN_FUNCTION_CALL
        }:
            _, _, arity, func = OPERATORS[token]
            args = [stack.pop() for _ in range(arity)]
            stack.append(func(*args[::-1]))
        else:
            raise ValueError(f"Token desconocido: {token}")

    assert len(stack) == 1, "Error: expresion invalida"
    return stack[0]
