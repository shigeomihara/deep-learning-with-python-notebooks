from dataclasses import dataclass

@dataclass
class DualNumber:
    primal  : float
    tangent : float
  
def lift(self, x):
    if isinstance(x, DualNumber):
        return x
    else:
        return DualNumber(x, 0.0)

def add(x, y):
    if not (isinstance(x, DualNumber) and isinstance(y, DualNumber)):
        print("Error in add, x, y is not DualNumber.")
        exit()
    p = x.primal + y.primal
    t = x.tangent + y.tangent
    return DualNumber(p, t)

def mul(x, y):
    if not (isinstance(x, DualNumber) and isinstance(y, DualNumber)):
        print("Error in mul, x, y is not DualNumber.")
        exit()
    p = x.primal * y.primal
    t = (x.tangent * y.primal) + (x.primal * y.tangent)
    return DualNumber(p, t)
    

def foo(x):
    if not isinstance(x, DualNumber):
        print("Error in foo, x is not DualNumber.")
        exit()
    return mul(x, add(x, DualNumber(3., 0.)))

print("foo(DualNumber(2.0, 1.))=", foo(DualNumber(2.0, 1.)))

def derivative(f, xp):
    return f(DualNumber(xp, 1.)).tangent

print("derivative(foo, 2.)=", derivative(foo, 2.))

def nth_order_derivative(n, f, xp):
    if n == 0:
        return f(DualNumber(xp, 1.)).primal
    else:
        return derivative(lambda x: nth_order_derivative(n-1, f, x.primal), xp)

# print("nth_order_derivative(0, foo, 2.0)=", nth_order_derivative(0, foo, 2.0))
# print("nth_order_derivative(1, foo, 2.0)=", nth_order_derivative(1, foo, 2.0))

          
      
