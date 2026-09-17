def foo(x):
  return mul(x, add(x, 3.0))

from enum import Enum, auto
from contextlib import contextmanager
from typing import Any

# The full (closed) set of primitive operations
class Op(Enum):
  add = auto()  # addition on floats
  mul = auto()  # multiplication on floats

# Interpreters have rules for handling each primitive operation.
class Interpreter:
  def interpret_op(self, op: Op, args: tuple[Any, ...]):
    assert False, "subclass should implement this"

# Our first interpreter is the "evaluating interpreter" which performs ordinary
# concrete evaluation.
class EvalInterpreter:
  def interpret_op(self, op, args):
    assert all(isinstance(arg, float) for arg in args)
    match op:
      case Op.add:
        x, y = args
        return x + y
      case Op.mul:
        x, y = args
        return x * y
      case _:
        raise ValueError(f"Unrecognized primitive op: {op}")

# The current interpreter is initially the evaluating interpreter.
current_interpreter = EvalInterpreter()

# A context manager for temporarily changing the current interpreter
@contextmanager
def set_interpreter(new_interpreter):
  global current_interpreter
  prev_interpreter = current_interpreter
  try:
    current_interpreter = new_interpreter
    yield
  finally:
    current_interpreter = prev_interpreter

from dataclasses import dataclass

# The user-facing functions `mul` and `add` dispatch to the current interpreter.
def add(x, y): return current_interpreter.interpret_op(Op.add, (x, y))
def mul(x, y): return current_interpreter.interpret_op(Op.mul, (x, y))

# This is like DualNumber above except that is also has a pointer to the
# interpreter it belongs to, which is needed to avoid "perturbation confusion"
# in higher order differentiation.
@dataclass
class TaggedDualNumber:
  interpreter : Interpreter
  primal  : float
  tangent : float

class JVPInterpreter(Interpreter):
  def __init__(self, prev_interpreter: Interpreter):
    # We keep a pointer to the interpreter that was current when this
    # interpreter was first invoked. That's the context in which our
    # rules should run.
    self.prev_interpreter = prev_interpreter

  def interpret_op(self, op, args):
    args = tuple(self.lift(arg) for arg in args)
    with set_interpreter(self.prev_interpreter):
      match op:
        case Op.add:
          # Notice that we use `add` and `mul` here, which are the
          # interpreter-dispatching functions defined earlier.
          x, y = args
          return self.dual_number(
              add(x.primal, y.primal),
              add(x.tangent, y.tangent))

        case Op.mul:
          x, y = args
          x = self.lift(x)
          y = self.lift(y)
          return self.dual_number(
              mul(x.primal, y.primal),
              add(mul(x.primal, y.tangent), mul(x.tangent, y.primal)))

  def dual_number(self, primal, tangent):
    return TaggedDualNumber(self, primal, tangent)

  # Lift a constant value (constant with respect to this interpreter) to
  # a TaggedDualNumber.
  def lift(self, x):
    if isinstance(x, TaggedDualNumber) and x.interpreter is self:
      return x
    else:
      return self.dual_number(x, 0.0)

def jvp(f, primal, tangent):
  jvp_interpreter = JVPInterpreter(current_interpreter)
  dual_number_in = jvp_interpreter.dual_number(primal, tangent)
  with set_interpreter(jvp_interpreter):
    result = f(dual_number_in)
  dual_number_out = jvp_interpreter.lift(result)
  return dual_number_out.primal, dual_number_out.tangent

# Let's try it out:
print(jvp(foo, 2.0, 1.0))

# Because we were careful to consider nesting interpreters, higher-order AD
# works out of the box:

def derivative(f, x):
  _, tangent = jvp(f, x, 1.0)
  return tangent

def nth_order_derivative(n, f, x):
  if n == 0:
    return f(x)
  else:
    return derivative(lambda x: nth_order_derivative(n-1, f, x), x)

print(nth_order_derivative(2, foo, 2.0))
