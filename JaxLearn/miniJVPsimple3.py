from dataclasses import dataclass

@dataclass
class DualNumber:
    primal : float
    tangent : float
  
def add(x, y):
    if isinstance(x, float) and isinstance(y, float):
        return x + y
    if isinstance(x, float) and not isinstance(y, float):
        #print("x is float and y is not float")####################
        return DualNumber(add(x, y.primal), add(0., y.tangent))
    if not isinstance(x, float) and isinstance(y, float):
        return DualNumber(add(x.primal, y), add(x.tangent, 0.))
    if not isinstance(x, float) and not isinstance(y, float):
        #print("x is not float and y is not float")####################
        return DualNumber(add(x.primal, y.primal), add(x.tangent, y.tangent))

def mul(x, y):
    if isinstance(x, float):
        if isinstance(y, float):
            return x * y
        else:
            #print("mul: x is float and y is not float")####################
            return DualNumber(mul(x, y.primal), mul(x, y.tangent))
    else:
        if isinstance(y, float):
            return DualNumber(mul(x.primal, y), mul(x.tangent, y))
        else:
            #print("mul: x is not float and y is not float")####################
            return DualNumber(mul(x.primal, y.primal),
                              add(mul(x.tangent, y.primal), mul(x.primal, y.tangent)))
def derivative(f, xp):
    y = f(DualNumber(xp, 1.))
    if isinstance(y, float):
        return 0.
    else:
        return y.tangent

def nth_order_derivative(n, f, x):
    if n == 0:
        return f(x)
    else:
        return derivative(lambda x: nth_order_derivative(n-1, f, x), x)

def foo(x):
    return mul(x, add(x, 3.0))

print("nth_order_derivative(0, foo, 2.0)=", nth_order_derivative(0, foo, 2.0))
print("nth_order_derivative(1, foo, 2.0)=", nth_order_derivative(1, foo, 2.0))
print("nth_order_derivative(2, foo, 2.0)=", nth_order_derivative(2, foo, 2.0))
print("nth_order_derivative(3, foo, 2.0)=", nth_order_derivative(3, foo, 2.0))

def baa(x):
    return mul(x, foo(x))

print("nth_order_derivative(1, baa, 2.0)=", nth_order_derivative(1, baa, 2.0))



          
      
