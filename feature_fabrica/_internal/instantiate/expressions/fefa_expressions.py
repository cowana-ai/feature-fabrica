import ast
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

    if not split_expression:
        return False

    needs_operand = True
    can_be_initital_data = True
    needs_operator = False
    # Iterate through each part of the split expression
    for token in split_expression:
        if not needs_operator and token == '(':
            parentheses_counter += 1
            needs_operand = True
            continue
        elif (not needs_operand or can_be_initital_data) and token == ')':
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
        can_be_initital_data = False
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
        arguments_str = match.group(2).strip()  # Get the arguments as a string

        # Parse arguments using ast
        try:
            # Parse the function call arguments using ast
            parsed_args = ast.parse(f"f({arguments_str})").body[0].value.args # type: ignore[attr-defined]
            parsed_keywords = ast.parse(f"f({arguments_str})").body[0].value.keywords # type: ignore[attr-defined]

            # Check if there are any positional arguments
            if parsed_args and not parsed_keywords:
                raise ValueError("Positional arguments are not allowed.")

            # Convert the AST nodes back into readable Python objects for keyword arguments
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in parsed_keywords}

            return function_name, kwargs  # Return only keyword arguments as a dictionary

        except Exception as e:
            print(f"Error parsing arguments: {e}")
            return None
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
                # Step 1: Split the token into function name and arguments
                fn_name, kwargs = split_function_call(token)
                # Step 2: Retrieve the transformation class and instantiate it
                fn_class = TransformationRegistry.get_transformation_class_by_name(fn_name)

                _hydrated_fn_class = {
                    "_target_": fn_class,
                    **kwargs
                }

                a = stack.pop() if stack else None
                if a is None:
                    stack.append(_hydrated_fn_class)
                elif isinstance(a, dict):
                    if _is_target(a):
                        stack.append({f"fn_{count_individual_steps}": a,
                         f"fn_{count_individual_steps + 1}": _hydrated_fn_class,
                        })
                        count_individual_steps += 2
                    else:
                        a[f"fn_{count_individual_steps}"] = _hydrated_fn_class
                        count_individual_steps += 1
                        stack.append(a)
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
