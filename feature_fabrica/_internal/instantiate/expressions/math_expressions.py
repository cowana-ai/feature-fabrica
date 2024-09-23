import keyword
import re
from collections import deque
from typing import Any


def is_valid_variable_name(name: str) -> bool:
    # Check if the name is a valid identifier and is not a Python keyword
    return name.isidentifier() and not keyword.iskeyword(name)

def is_numeric(x: str) -> bool:
    try:
        float(x)
    except Exception:
        return False
    return True

def _is_valid_math_expression(expression: str) -> bool:
    parentheses_counter = 0
    # Split the expression based on the operators
    split_expression = re.split(r'(\(|\)|\+|\-|\*|\/)', expression)
    # Remove any empty strings that may result from the split
    split_expression = [token.strip() for token in split_expression if token.strip()]

    if not split_expression or len(split_expression) < 3:
        return False

    needs_operand = True
    needs_operator = False
    # Iterate through each part of the split expression
    for token in split_expression:
        if token == '(':
            parentheses_counter += 1
            needs_operand = True
        elif token == ')':
            parentheses_counter -= 1
        elif needs_operand and (is_numeric(token) or is_valid_variable_name(token)):
            needs_operand = False
            needs_operator = True
        elif needs_operator and token in ['+', '*', '/', '-']:
            needs_operator = False
            needs_operand = True
        else:
            return False
    # Check if all parentheses were closed
    return parentheses_counter == 0


def _hydrate_math_expression(expression: str, validate_expression: bool = False) -> Any:
    if validate_expression and not _is_valid_math_expression(expression):
        raise ValueError()

    # Define the precedence and associativity of operators
    OPERATORS = {
        '+': {'precedence': 1, 'associativity': 'L', 'transformation': 'SumReduce'},
        '-': {'precedence': 1, 'associativity': 'L', 'transformation': 'SubtractReduce'},
        '*': {'precedence': 2, 'associativity': 'L', 'transformation': 'MultiplyReduce'},
        '/': {'precedence': 2, 'associativity': 'L', 'transformation': 'DivideReduce'}
    }
    def is_operator(token):
        return token in OPERATORS

    def precedence(op):
        return OPERATORS[op]['precedence']

    def associativity(op):
        return OPERATORS[op]['associativity']

    def get_transformation(op):
        return OPERATORS[op]['transformation']

    def shunting_yard(tokens):
        """Convert infix expression to postfix (RPN) using the Shunting Yard algorithm."""
        output = []
        operators = deque()

        for token in tokens:
            if is_numeric(token) or is_valid_variable_name(token):  # Token is a variable or number
                output.append(token)
            elif token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()  # Pop the '('
            elif is_operator(token):
                while (operators and operators[-1] != '(' and
                       (associativity(token) == 'L' and precedence(token) <= precedence(operators[-1])) or
                       (associativity(token) == 'R' and precedence(token) < precedence(operators[-1]))):
                    output.append(operators.pop())
                operators.append(token)

        # Pop any remaining operators
        while operators:
            output.append(operators.pop())

        return output

    def build_ast(postfix_tokens):
        """Build an AST (abstract syntax tree) from postfix tokens."""
        stack = []
        last_operand = None
        for token in postfix_tokens:
            if is_numeric(token) or is_valid_variable_name(token):  # If it's a variable or number
                stack.append(token)
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
                        if cur_operand == last_operand:
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
                last_operand = cur_operand

        if len(stack) != 1:
            raise ValueError(f"Unexpected result after processing: {stack}")

        return stack[0]  # The final AST should be the only element in the stack
    # Split the expression based on the operators
    split_expression = re.split(r'(\(|\)|\+|\-|\*|\/)', expression)
    # Remove any empty strings that may result from the split
    split_expression = [token.strip() for token in split_expression if token.strip()]
    # Convert infix tokens to postfix (RPN) using the Shunting Yard algorithm
    postfix_tokens = shunting_yard(split_expression)

    # Build an AST from the postfix tokens
    ast = build_ast(postfix_tokens)

    # Wrap the expression in PromiseValue
    return ast
