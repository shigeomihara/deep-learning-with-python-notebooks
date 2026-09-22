from dataclasses import dataclass

@dataclass
class DualNumber:
    primal : float
    tangent : float
  
def add(x, y):
    if isinstance(x, float) and isinstance(y, float):
        return x + y
    if isinstance(x, float) and not isinstance(y, float):
        #return DualNumber(add(x, y.primal), add(0., y.tangent))
        return DualNumber(add(x, y.primal), y.tangent)
    if not isinstance(x, float) and isinstance(y, float):
        #return DualNumber(add(x.primal, y), add(x.tangent, 0.))
        return DualNumber(add(x.primal, y), x.tangent)
    if not isinstance(x, float) and not isinstance(y, float):
        return DualNumber(add(x.primal, y.primal), add(x.tangent, y.tangent))
        # return DualNumber(
        #     DualNumber(add(x.primal, y.primal), y.tangent), x.tangent )
        #return DualNumber(add(x.primal, y), x.tangent)

def mul(x, y):
    if isinstance(x, float) and isinstance(y, float):
        return x * y
    if isinstance(x, float) and not isinstance(y, float):
        return DualNumber(mul(x, y.primal), mul(x, y.tangent))
    if not isinstance(x, float) and isinstance(y, float):
        return DualNumber(mul(x.primal, y), mul(x.tangent, y))
    if not isinstance(x, float) and not isinstance(y, float):
        return DualNumber(mul(x.primal, y.primal),
                          add(mul(x.tangent, y.primal), mul(x.primal, y.tangent)))
        #     # return DualNumber(
        #     #     DualNumber(mul(x.primal, y.primal), mul(x.primal, y.tangent)),
        #     #     DualNumber(mul(y.primal, x.tangent), mul(x.tangent, y.tangent)))
        #     return DualNumber(mul(x.primal, y), mul(x.tangent, y))
            
def derivative(f, xp):
    f_val = f(DualNumber(xp, 1.))
    if isinstance(f_val, float):
        return 0.
    else:
        return f_val.tangent

def nth_order_derivative2(nx, ny, f, x, y):
    if nx == 0 and ny == 0:
        return f(x, y)
    if nx >= 1:
        return derivative(lambda x: nth_order_derivative2(nx-1, ny, f, x, y), x)
    if ny >= 1:
        return derivative(lambda y: nth_order_derivative2(nx, ny-1, f, x, y), y)

def foo(x, y):
    #return mul(x, add(x, y))
    return mul(x, add(y, x))
    #return mul(x, y)

# print("nth_order_derivative(0, 0, foo, 2.0, 3.0)=",
#       nth_order_derivative(0, 0, foo, 2.0, 3.0))
# print("nth_order_derivative(1, 0, foo, 2.0, 3.0)=",
#       nth_order_derivative(1, 0, foo, 2.0, 3.0))
# print("nth_order_derivative(0, 1, foo, 2.0, 3.0)=",
#       nth_order_derivative(0, 1, foo, 2.0, 3.0))
print("nth_order_derivative2(1, 1, foo, 2.0, 3.0)=",
      nth_order_derivative2(1, 1, foo, 2.0, 3.0))
print("nth_order_derivative2(2, 1, foo, 2.0, 3.0)=",
      nth_order_derivative2(2, 1, foo, 2.0, 3.0))
# print("foo(DualNumber(2.0, 1), DualNumber(3.0, 1))=",
#       foo(DualNumber(2.0, 1.), DualNumber(3.0, 1.)))

# def nth_order_derivative(n, f, x):
#     if n == 0:
#         return f(x)
#     else:
#         return derivative(lambda x: nth_order_derivative(n-1, f, x), x)

# # これはうまく行かない。１変数用のadd, mulに変えないといけないだろう。
# def baa(x):
#     return foo(x, x)
          
# print("nth_order_derivative(0, baa, 2.0)=",
#       nth_order_derivative(0, baa, 2.0))
# print("nth_order_derivative(1, baa, 2.0)=",
#       nth_order_derivative(1, baa, 2.0))
# print("nth_order_derivative(2, baa, 2.0)=",
#       nth_order_derivative(2, baa, 2.0))

# これもうまく行かない。
def baa2(x, y):
    #return foo(add(x, y), add(x, y))
    #return mul(2., mul(add(x, y), add(x, y)))
    return mul(add(x, y), add(x, y))
          
print("nth_order_derivative2(0, 0, baa2, 2.0, 0.0)=",
      nth_order_derivative2(0, 0, baa2, 2.0, 0.0))
print("nth_order_derivative2(1, 0, baa2, 2.0, 0.0)=",
      nth_order_derivative2(1, 0, baa2, 2.0, 0.0))
print("nth_order_derivative2(2, 0, baa2, 2.0, 0.0)=",
      nth_order_derivative2(2, 0, baa2, 2.0, 0.0))
