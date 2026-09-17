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

# The user-facing functions `mul` and `add` dispatch to the current interpreter.
def add(x, y): return current_interpreter.interpret_op(Op.add, (x, y))
def mul(x, y): return current_interpreter.interpret_op(Op.mul, (x, y))

from dataclasses import dataclass

# A primal-tangent pair is conventionally called a "dual number"
@dataclass
class DualNumber:
  primal  : float
  tangent : float

def add_dual(x : DualNumber, y: DualNumber) -> DualNumber:
  return DualNumber(x.primal + y.primal, x.tangent + y.tangent)

def mul_dual(x : DualNumber, y: DualNumber) -> DualNumber:
  return DualNumber(x.primal * y.primal, x.primal * y.tangent + x.tangent * y.primal)

def foo_dual(x : DualNumber) -> DualNumber:
  return mul_dual(x, add_dual(x, DualNumber(3.0, 0.0)))

print (foo_dual(DualNumber(2.0, 1.0)))
