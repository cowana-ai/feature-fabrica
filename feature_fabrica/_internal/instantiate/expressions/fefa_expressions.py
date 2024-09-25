import keyword
import re
from collections import deque
from typing import Any

from hydra._internal.instantiate._instantiate2 import _is_target
from omegaconf import OmegaConf

from feature_fabrica.transform.registry import TransformationRegistry

# Define the precedence and associativity of operators
BASIC_MATH_OPERATORS = {
    '+': {'precedence': 1, 'transformation': 'SumReduce'},
    '-': {'precedence': 1, 'transformation': 'SubtractReduce'},
    '*': {'precedence': 2, 'transformation': 'MultiplyReduce'},
    '/': {'precedence': 2, 'transformation': 'DivideReduce'}
}
def is_operator(token):
    return token in BASIC_MATH_OPERATORS

def get_transformation(op):
    return BASIC_MATH_OPERATORS[op]['transformation']

def is_valid_variable_name(name: str) -> bool:
    # Check if the name is a valid identifier and is not a Python keyword
    return name.isidentifier() and not keyword.iskeyword(name)

def is_numeric(x: str) -> bool:
    try:
        float(x)
    except Exception:
        return False
    return True

def is_function(x: str) -> bool:
    return x.startswith('.')

def tokenize(expression: str):
    # Match numbers (integers, decimals), variable names, operators, parentheses, and function calls with parameters
    token_pattern = re.compile(r'\d+\.\d+|\d+|\b\w+\b|\.\w+\([^\)]*\)|[()+\-*/]')
    tokens = re.findall(token_pattern, expression)
    return tokens

def _is_valid_expression(expression: str) -> bool:
    parentheses_counter = 0
    # Split the expression based on the operators
    split_expression = tokenize(expression)
    # Remove any empty strings that may result from the split
    split_expression = [token.strip() for token in split_expression if token.strip()]

    if not split_expression or len(split_expression) < 3:
        return False

    needs_operand = True
    needs_operator = False
    # Iterate through each part of the split expression
    for token in split_expression:
        if not needs_operator and token == '(':
            parentheses_counter += 1
            needs_operand = True
        elif not needs_operand and token == ')':
            parentheses_counter -= 1
            needs_operand = False
            needs_operator = True
        elif needs_operand and (is_numeric(token) or is_valid_variable_name(token)):
            needs_operand = False
            needs_operator = True
        elif needs_operator and token in ['+', '*', '/', '-']:
            needs_operator = False
            needs_operand = True
        elif needs_operator and is_function(token):
            needs_operator = True
            needs_operand = False
        else:
            return False
    # Check if all parentheses were closed
    return parentheses_counter == 0 and not needs_operand


def infix_fefa_expression_to_postfix(expression: str):
    OPERATORS = {
        '+': {'precedence': 1},
        '-': {'precedence': 1},
        '*': {'precedence': 2},
        '/': {'precedence': 2}
    }

    def is_operator(token):
        return token in OPERATORS

    def precedence(token):
        return OPERATORS[token]['precedence']

    # Tokenize the expression
    tokens = tokenize(expression)
    output = []
    operator_stack = deque() # type: ignore

    for token in tokens:
        if is_numeric(token) or is_valid_variable_name(token):  # Numeric or variable
            output.append(token)
        elif is_function(token):  # Function call like .log(...)
            output.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output.append(operator_stack.pop())
            operator_stack.pop()  # Remove '('
        elif is_operator(token):
            while (operator_stack and operator_stack[-1] != '(' and
                   precedence(token) <= precedence(operator_stack[-1])):
                output.append(operator_stack.pop())
            operator_stack.append(token)

    # Pop any remaining operators
    while operator_stack:
        output.append(operator_stack.pop())

    return output

def split_function_call(expression: str):
    pattern = r'\.(\w+)\((.*)\)'
    match = re.match(pattern, expression.strip())

    if match:
        function_name = match.group(1)  # Get the function name
        arguments = match.group(2).split(',')  # Split the arguments by comma
        arguments = [arg.strip() for arg in arguments if arg.strip()]  # Remove whitespace
        return function_name, arguments  # Return function name and arguments as a list
    return None


def _hydrate_fefa_expression(expression: str, validate_expression: bool = False) -> Any:
    if validate_expression and not _is_valid_expression(expression):
        raise ValueError()

    def build_ast(postfix_tokens):
        """Build an AST (abstract syntax tree) from postfix tokens."""
        stack = []
        count_individual_steps = 0
        for token in postfix_tokens:
            if is_numeric(token) or is_valid_variable_name(token):  # If it's a variable or number
                stack.append(token)
            elif is_function(token):
                if len(stack) < 1:
                    raise ValueError(f"Insufficient operands for operator '{token}'")
                a = stack.pop()
                # Step 1: Split the token into function name and arguments
                fn_name_and_kwargs = split_function_call(token)

                if fn_name_and_kwargs:
                    fn_name = fn_name_and_kwargs[0]  # Get function name
                    kwargs_str = fn_name_and_kwargs[1]  # Get the list of arguments
                    # Step 2: Convert the arguments into a dictionary
                    kwargs = {}
                    for arg in kwargs_str:
                        if '=' in arg:  # Check if it's a keyword argument
                            key, value = arg.split('=', 1)
                            kwargs[key.strip()] = eval(value.strip())  # Use eval for dynamic evaluation
                        else:
                            kwargs[arg.strip()] = None  # Or handle non-keyword arguments as needed

                    # Step 3: Retrieve the transformation class and instantiate it
                    fn_class = TransformationRegistry.get_transformation_class_by_name(fn_name)
                else:
                    raise ValueError("Invalid function call format.")

                _hydrated_fn_class = {
                    "_target_": fn_class,
                    **kwargs
                }
                if isinstance(a, dict):
                    stack.append({f"fn_{count_individual_steps}": a,
                     f"fn_{count_individual_steps + 1}": _hydrated_fn_class,
                    })
                    count_individual_steps += 2
                else:
                    if is_numeric(a) or not is_valid_variable_name(a):
                        raise ValueError()
                    importer_fn_class = TransformationRegistry.get_transformation_class_by_name("import")
                    _hydrated_importer_fn_class = {
                        "_target_": importer_fn_class,
                        "feature": a
                    }
                    stack.append({f"fn_{count_individual_steps}": _hydrated_importer_fn_class,
                     f"fn_{count_individual_steps + 1}": _hydrated_fn_class,
                    })
                    count_individual_steps += 2

            elif is_operator(token):
                if len(stack) < 2:
                    raise ValueError(f"Insufficient operands for operator '{token}'")

                # Pop operands from the stack
                b = stack.pop()

                if not isinstance(b, dict) and is_numeric(b):
                    b = float(b)
                a = stack.pop()
                if not isinstance(a, dict) and is_numeric(a):
                    a = float(a)
                cur_operand = get_transformation(token)

                # If the operator is associative, try to group them into a single transformation
                if isinstance(a, dict):
                    if not isinstance(b, dict):
                        if _is_target(a) and cur_operand in a['_target_']:
                            a['iterable'].append(b)
                            stack.append(a)
                        else:
                            stack.append(
                                {
                                    '_target_': f'feature_fabrica.transform.{cur_operand}',
                                    'iterable': [
                                         {
                                            '_target_': 'feature_fabrica.models.PromiseValue',
                                            'transformation': a
                                        },
                                         b
                                    ]
                                }
                            )
                    else:
                        # Create a new transformation with the operands
                        stack.append({
                            '_target_': f'feature_fabrica.transform.{cur_operand}',
                            'iterable': [
                                            {
                                            '_target_': 'feature_fabrica.models.PromiseValue',
                                            'transformation': a
                                            },

                                           {
                                           '_target_': 'feature_fabrica.models.PromiseValue',
                                           'transformation': b
                                           }
                                        ]
                        })
                else:

                    # Create a new transformation with the operands
                    stack.append({
                        '_target_': f'feature_fabrica.transform.{cur_operand}',
                        'iterable': [a, b]
                    })


        if len(stack) != 1:
            raise ValueError(f"Unexpected result after processing: {stack}")

        return stack[0]  # The final AST should be the only element in the stack

    postfix_tokens = infix_fefa_expression_to_postfix(expression)
    # Build an AST from the postfix tokens
    ast = build_ast(postfix_tokens)

    # Wrap the expression in PromiseValue
    return OmegaConf.create(ast)
