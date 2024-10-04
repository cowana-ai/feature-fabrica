import keyword
import re
from functools import lru_cache

import feature_fabrica.transform.registry as registry

# Define operator precedence and corresponding transformations
BASIC_MATH_OPERATORS = {
    '+': {'precedence': 1, 'transformation': 'SumReduce'},
    '-': {'precedence': 1, 'transformation': 'SubtractReduce'},
    ',': {'precedence': 1, 'transformation': 'FeatureImporter'},
    '*': {'precedence': 2, 'transformation': 'MultiplyReduce'},
    '/': {'precedence': 2, 'transformation': 'DivideReduce'},
}
OPEN_PARENTHESIS = "("
CLOSE_PARENTHESIS = ")"
FUNCTION_PATTERN = r'\.(\w+)\((.*)\)'
#TOKEN_PATTERN = re.compile(r'\d+\.\d+|\d+|\b\w+\b|\.\w+\([^\)]*\)|[,()+\-*/]')
TOKEN_PATTERN = re.compile(r'\d+\.\d+|\d+|\b\w+:\w+\b|\b\w+\b|\.\w+\([^\)]*\)|[,()+\-*/]')

@lru_cache(maxsize=1024)
def is_operator(token: str) -> bool:
    """Check if the token is a mathematical operator."""
    return token in BASIC_MATH_OPERATORS

def get_precedence(token: str) -> int:
    return BASIC_MATH_OPERATORS[token]['precedence'] # type: ignore[return-value]

def get_transformation(op: str) -> str:
    """Get the corresponding transformation for the given operator."""
    return BASIC_MATH_OPERATORS[op]['transformation'] # type: ignore[return-value]

@lru_cache(maxsize=1024)
def is_valid_variable_name(name: str) -> bool:
    """Check if the name is a valid Python variable name (non-keyword and identifier)."""
    return (name.isidentifier() and not keyword.iskeyword(name)) or is_valid_promise_value(name)

@lru_cache(maxsize=1024)
def is_valid_promise_value(name: str) -> bool:
    if ":" in name:
        name_stage = name.split(":")
        if len(name_stage) == 2:
            name, transform_stage = name_stage
            return is_valid_variable_name(name) and transform_stage in registry.TransformationRegistry.registry
    return False

@lru_cache(maxsize=1024)
def is_numeric(token: str) -> bool:
    """Check if the token can be converted to a number."""
    try:
        float(token)
        return True
    except ValueError:
        return False

@lru_cache(maxsize=1024)
def is_function(token: str) -> bool:
    """Check if the token represents a function call."""
    match = re.match(FUNCTION_PATTERN, token.strip())
    return match is not None and match.group() == token
