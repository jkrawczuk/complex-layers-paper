INFINITY = 1e20
ZERO = 1e-8
EPSILON = 1e-10

def EQUAL_ZERO(value):
    return abs(value) < ZERO

def EQUALS_EPSILON(value1, value2):
    return abs(value1 - value2) < EPSILON