
from typing import Generator, Optional, Union
import numpy as np
import sympy as sp
from math import log2
from cmath import log, nan, isnan, \
    sin, cos, tan, sinh, cosh, tanh, \
    asin, acos, atan, asinh, acosh, atanh

from fischer.greek_symbols import symbols_unreserved_in_math


__all__ = ['Constant', 'Operator', 'BinaryOperator', 'Addition', 'Subtraction', 'Multiplication', 'Division', 'Exponentiation', 'Logarithm', 'operator_names', 'possible_scalars', 'Pi', 'Euler', 'Phi', 'all_constants']

operator_names = ['add', 'sub', 'mul', 'div', 'exp', 'log', 'trig_sin', 'trig_cos', 'trig_tan', 'trig_asin', 'trig_acos', 'trig_atan', 'trig_sinh', 'trig_cosh', 'trig_tanh', 'trig_asinh', 'trig_acosh', 'trig_atanh']

# ensure that all trig functions are considered as one of seven functions from which to randomly select but with a random "parameter", defining which trig function to use
operator_name_random_weights = [1/7] * 6 + [1/7 * 1/12] * 12 # This assumes that exponentiation and trig-exponentiation are different ways to exponentiate, so exponentiation is weighted twice as much as the other operations overall.

# Here's a list of possible symbols to be used for scalars in expressions, omitting commonly used symbols such as π and x.
possible_scalars = [chr(65 + n) for n in range(26)] + [chr(97 + n) for n in range(26)]
possible_scalars.remove('e')
possible_scalars.remove('i')
possible_scalars.remove('x')
possible_scalars.remove('y')
possible_scalars.extend(symbols_unreserved_in_math)


class Constant:
    def __init__(self, val: Union[float, int, complex, 'Constant'], symbol: str = None):
        if isinstance(val, Constant):
            if symbol is None:
                symbol = val.symbol
            val = val.val
        if symbol is None:
            symbol = '???'
        self.val = val
        self.symbol = symbol
    def __call__(self):
        return self.val

class Parameter:
    def __init__(self, order: int):
        if order is None:
            order = '???'
        self.order = order
    def __call__(self):
        return

class Operator:
    def __init__(self):
        self.depth: int = 0
        self.parent: Operator = None
    def generate_symbols(self, available_symbols: list[sp.Symbol] = None):
        raise NotImplementedError
    def __call__(self) -> float:
        raise NotImplementedError
    def empty_nodes(self):
        raise NotImplementedError
    def get_head_node(self) -> 'Operator':
        p = self
        while p.parent is not None:
            p = p.parent
        return p
    @staticmethod
    def from_str(s: str):
        if s == 'add':
            return Addition()
        elif s == 'sub':
            return Subtraction()
        elif s == 'mul':
            return Multiplication()
        elif s == 'div':
            return Division()
        elif s == 'exp':
            return Exponentiation()
        elif s == 'log':
            return Logarithm()
        elif s.startswith('trig_'):
            func_name = s[s.find('_')+1:]
            # print(f'Creating new trig exp operator with trig function: {func_name}')
            func = globals()[func_name]
            # print(f'built-in func: {func}')
            # print(f'SymPy func: {getattr(sp, func.__name__)}')
            return TrigonometricExponentiation(func=func)
        else:
            if 'trig_' + s in operator_names:
                return Operator.from_str('trig_' + s)
    def set_empty_child_node(self, i: int, obj: Union[int, 'Operator', Constant]):
        raise NotImplementedError
    def dfs(self) -> Generator[Optional[Union[int, 'Operator', Constant]], None, None]:
        raise NotImplementedError
    def dfs_with_parents(self, parent: Optional['Operator'] = None) -> Generator[tuple[Optional['Operator'], Optional[Union[int, 'Operator', Constant]]], None, None]:
        raise NotImplementedError
    def __index__(self, i):
        raise NotImplementedError
    def symbolic(self):
        raise NotImplementedError
    def update_depths_and_parents(self):
        raise NotImplementedError
    # def next_symbol(self):
    #     if Operator.total_symbols < 26:
    #         s = sp.Symbol()
    #     elif Operator.total_symbols < 52:
    #         s = sp.Symbol(chr(97 + Operator.total_symbols - 26))
    #     else:
    #         s = sp.Symbol('?')
    #     Operator.total_symbols += 1
    #     return s
    def remove_child_node(self, child: Union[int, 'Operator', Constant]):
        raise NotImplementedError
    def has_fast_result(self) -> bool:
        raise NotImplementedError
    def has_empty_children(self) -> bool:
        raise NotImplementedError



class BinaryOperator(Operator):
    def __init__(self, left: Optional[Union[int, Operator, Constant]] = None, right: Optional[Union[int, Operator, Constant]] = None, depth: int = 0):
        super().__init__()
        self.left = left
        self.left_symbol = None
        self.right = right
        self.right_symbol = None
        self.available_symbols = None
    def generate_symbols(self, available_symbols: list[sp.Symbol] = None):
        if available_symbols is None:
            if self.available_symbols is None:
                self.available_symbols = [sp.Symbol(s) for s in possible_scalars]
                # print('created new symbols list')
            available_symbols = self.available_symbols
            # must recursively determine which symbols are already in use
            for node in self.get_head_node().dfs():
                # print(f'Visited node {node}')
                if isinstance(node, BinaryOperator):
                    if node.left_symbol is not None and node.left_symbol in available_symbols:
                        available_symbols.remove(node.left_symbol)
                    if node.right_symbol is not None and node.right_symbol in available_symbols:
                        available_symbols.remove(node.right_symbol)
        # print(f'left: {self.left}')
        if self.left is None:
            if self.left_symbol is None:
                if len(available_symbols) == 0:
                    raise Exception(f'Expression is too long; ran out of symbols for scalars.')
                self.left_symbol = available_symbols.pop(0)
                # print(f'popped symbol for left, new remaining: {available_symbols}')
        if isinstance(self.left, Operator):
            self.left.generate_symbols(available_symbols=available_symbols)
            # print('recursively populated left operator\'s symbols')
        # print(f'right: {self.right}')
        if self.right is None:
            if self.right_symbol is None:
                if len(available_symbols) == 0:
                    raise Exception(f'Expression is too long; ran out of symbols for scalars.')
                self.right_symbol = available_symbols.pop(0)
                # print(f'popped symbol for right, new remaining: {available_symbols}')
        if isinstance(self.right, Operator):
            self.right.generate_symbols(available_symbols=available_symbols)
            # print('recursively populated right operator\'s symbols')
    def update_depths_and_parents(self):
        if isinstance(self.left, Operator):
            self.left.depth = self.depth + 1
            self.left.parent = self
            self.left.update_depths_and_parents()
        if isinstance(self.right, Operator):
            self.right.depth = self.depth + 1
            self.right.parent = self
            self.right.update_depths_and_parents()
    def get_left_right(self):
        left = self.left
        right = self.right
        # if isinstance(left, sp.Symbol) or isinstance(right, sp.Symbol):
        #     raise ValueError(f'Tried to evaluate expression with un-substituted variable. Substitute the variable first by replacing it with a numerical value such as a Constant or int.')
        # if left is not None and not isinstance(left, int):
        #     left = left()
        # if right is not None and not isinstance(right, int):
        #     right = right()
        if left is not None and not isinstance(left, (int, sp.Symbol)):
            left = left()
        if right is not None and not isinstance(right, (int, sp.Symbol)):
            right = right()
        return left, right
    def empty_nodes(self):
        if self.left is None:
            yield self
        elif isinstance(self.left, BinaryOperator):
            yield from self.left.empty_nodes()
        if self.right is None:
            yield self
        elif isinstance(self.right, BinaryOperator):
            yield from self.right.empty_nodes()
    def set_empty_child_node(self, i: int, obj: Union[int, Operator, Constant]):
        if i == 0 and self.left is None:
            self.left = obj
            if isinstance(self.left, Operator):
                self.left.parent = self
        elif i == 1 and self.right is None:
            self.right = obj
            if isinstance(self.right, Operator):
                self.right.parent = self
    def dfs(self) -> Generator[Optional[Union[int, Operator, Constant]], None, None]:
        yield self
        if isinstance(self.left, Operator):
            yield from self.left.dfs()
        else:
            yield self.left
        if isinstance(self.right, Operator):
            yield from self.right.dfs()
        else:
            yield self.right
    def dfs_with_parents(self, parent: Optional[Operator] = None, include_empty_nodes: bool = True) -> Generator[tuple[Optional[Operator], Optional[Union[int, Operator, Constant]]], None, None]:
        yield parent, self
        if include_empty_nodes or self.left is not None:
            if isinstance(self.left, Operator):
                yield from self.left.dfs_with_parents(parent=self, include_empty_nodes=include_empty_nodes)
            else:
                yield self, self.left
        if include_empty_nodes or self.right is not None:
            if isinstance(self.right, Operator):
                yield from self.right.dfs_with_parents(parent=self, include_empty_nodes=include_empty_nodes)
            else:
                yield self, self.right
    def get_left_right_symbols(self):
        # if self.available_symbols is None:
        #     print(f'generate new symbols for expr: {self}')
        #     self.generate_symbols()
        s_left = self.left
        if s_left is None:
            if self.left_symbol is None:
                self.generate_symbols()
            s_left = self.left_symbol
        elif isinstance(s_left, Constant):
            s_left = sp.Symbol(s_left.symbol)
        elif isinstance(s_left, Operator):
            s_left = s_left.symbolic()
        s_right = self.right
        if s_right is None:
            if self.right_symbol is None:
                self.generate_symbols()
            s_right = self.right_symbol
        elif isinstance(s_right, Constant):
            s_right = sp.Symbol(s_right.symbol)
        elif isinstance(s_right, Operator):
            s_right = s_right.symbolic()
        # print(f'Found left/right symbols: {s_left}, {s_right}\n\t(from expression: {self})')
        return s_left, s_right
    def remove_child_node(self, child: Union[int, 'Operator', Constant]):
        if child is None:
            raise ValueError(f'Cannot remove None from node {self}')
        if self.right == child:
            self.right = None
        elif self.left == child:
            self.left = None
    def has_empty_children(self) -> bool:
        return self.left == None and self.right == None
    def __index__(self, i: int) -> Union[int, Operator, Constant]:
        if i == 0:
            return self.left
        if i == 1:
            return self.right
        raise IndexError(f'Index {i} is out of bounds.  BinaryOperator has only two children nodes.')

class Addition(BinaryOperator):
    def __call__(self):
        left, right = self.get_left_right()
        if left is None or right is None:
            return
        if isinstance(left, float) and isnan(left) or isinstance(right, float) and isnan(right):
            return nan
        return left + right
    def symbolic(self):
        s_left, s_right = self.get_left_right_symbols()
        return sp.Add(s_left, s_right)
    def has_fast_result(self) -> bool:
        return False

class Subtraction(BinaryOperator):
    def __call__(self):
        left, right = self.get_left_right()
        if left is None or right is None:
            return
        if isinstance(left, float) and isnan(left) or isinstance(right, float) and isnan(right):
            return nan
        return left - right
    def symbolic(self):
        s_left, s_right = self.get_left_right_symbols()
        return sp.Add(s_left, -s_right)
    def has_fast_result(self) -> bool:
        return False

class Multiplication(BinaryOperator):
    def __call__(self):
        if self.left == 0 or self.right == 0:
            return 0
        if isinstance(self.left, Operator) and self.left.has_fast_result() and self.left() == 0:
            return 0
        if isinstance(self.right, Operator) and self.right.has_fast_result() and self.right() == 0:
            return 0
        left, right = self.get_left_right()
        if left is None or right is None:
            return
        if isinstance(left, float) and isnan(left) or isinstance(right, float) and isnan(right):
            return nan
        if left == 0 or right == 0:
            return 0
        if isinstance(left, float) and isinstance(right, float):
            if log2(abs(left)) + log2(abs(right)) >= 950:
                return nan
        return left * right
    def symbolic(self):
        left, right = self.get_left_right()
        if left is not None and right is not None:
            if right == 0 or left == 0:
                return 0
            if isinstance(left, float) and isinstance(right, float):
                if log2(abs(left)) + log2(abs(right)) >= 950:
                    return nan
        s_left, s_right = self.get_left_right_symbols()
        return sp.Mul(s_left, s_right)
    def has_fast_result(self) -> bool:
        if self.left == 0 or self.right == 0:
            return True
        if isinstance(self.left, Operator) and self.left.has_fast_result() and self.left() == 0:
            return True
        if isinstance(self.right, Operator) and self.right.has_fast_result() and self.right() == 0:
            return True
        return False

class Division(BinaryOperator):
    def __call__(self):
        if isinstance(self.left, Operator) and self.left.has_fast_result():
            v = self.left()
            if v == 0:
                return 0
            if isnan(v):
                return nan
        left, right = self.get_left_right()
        if left is None or right is None:
            return
        if isinstance(left, float) and isnan(left) or isinstance(right, float) and isnan(right):
            return nan
        if left == 0:
            return 0
        if right == 0:
            return nan
        if isinstance(left, float) and isinstance(right, float):
            if log2(abs(left)) - log2(abs(right)) >= 950:
                return nan
        return left / right
    def symbolic(self):
        left, right = self.get_left_right()
        if right is not None:
            if right == 0:
                return nan
            if left is not None:
                if left == 0:
                    return 0
                if isinstance(left, float) and isinstance(right, float):
                    if log2(abs(left)) - log2(abs(right)) >= 950:
                        return nan
        s_left, s_right = self.get_left_right_symbols()
        return sp.Mul(s_left, 1/s_right)
    def has_fast_result(self) -> bool:
        if isinstance(self.left, Operator) and self.left.has_fast_result():
            v = self.left()
            if v == 0 or isnan(v):
                return True
        return False

class Exponentiation(BinaryOperator):
    def __call__(self):
        if self.right == 0:
            return 1
        if isinstance(self.right, Operator) and self.right.has_fast_result() and self.right() == 0:
            return 1
        left, right = self.get_left_right()
        if left is None or right is None:
            return
        if isinstance(left, float) and isnan(left) or isinstance(right, float) and isnan(right):
            return nan
        if right == 0:
            return 1
        if left == 0:
            return 0
        if left == -1: # ??? This was put here for a reason, I can't remember why
            if abs(right) >= 200:
                return nan
        if isinstance(left, float) and isinstance(right, float):
            if abs(log2(abs(left)) * abs(right)) >= 247:
                return nan
            try:
                v = left ** right
            except OverflowError as e:
                print(f'Error in exponentiation {left}**{right}: {e}')
                print(log2(abs(left)), abs(right))
                print(log2(abs(left)) * abs(right))
                # input()
                return nan
            return v
    def symbolic(self):
        left, right = self.get_left_right()
        if left is not None and right is not None:
            if right == 0:
                return 1
            if left == 0:
                return 0
            if isinstance(left, float) and isinstance(right, float):
                if abs(log2(abs(left)) * abs(right)) >= 250:
                    return nan
        s_left, s_right = self.get_left_right_symbols()
        return sp.Pow(s_left, s_right)
    def has_fast_result(self) -> bool:
        if self.right == 0:
            return True
        if isinstance(self.right, Operator) and self.right.has_fast_result() and self.right() == 0:
            return True
        return False

class Logarithm(BinaryOperator):
    # Left node is the base of the logarithm
    def __call__(self):
        if self.right == 1:
            return 0
        if isinstance(self.right, Operator) and self.right.has_fast_result():
            v = self.right()
            if v == 0:
                return nan
            if v == 1:
                return 0
        if self.left == 1:
            return nan
        if isinstance(self.left, Operator) and self.left.has_fast_result():
            v = self.left()
            if v == 0 or v == 1:
                return nan
        left, right = self.get_left_right()
        if left is None or right is None:
            return
        if isinstance(left, float) and isinstance(right, float):
            if isnan(left) or isnan(right):
                return nan
        if left == 0 or left == 1:
            return nan
        if right == 0:
            return nan
        if isinstance(left, float) and isinstance(right, float):
            if log2(abs(right)) / abs(left) >= 500:
                return nan
            v = log(right, left)
            if v.imag == 0:
                return v.real
            return v
        # raise Exception(f'Cannot evaluate Logarithm operator with __call__() because it contains a symbol.')
    def symbolic(self):
        left, right = self.get_left_right()
        if left is not None and right is not None:
            if right == 0 or left == 0 or left == 1:
                return nan
            if isinstance(left, float) and isinstance(right, float):
                if log2(abs(left)) * abs(right) >= 500:
                    return nan
        s_left, s_right = self.get_left_right_symbols()
        return sp.log(s_right, s_left)
    def has_fast_result(self) -> bool:
        if self.right == 1:
            return True
        if isinstance(self.right, Operator) and self.right.has_fast_result() and self.right() == 1:
            return True
        if self.left == 0 or self.left == 1:
            return True
        if isinstance(self.left, Operator) and self.left.has_fast_result():
            v = self.left()
            if v == 0 or v == 1:
                return True
        if isinstance(self.right, Operator) and self.right.has_fast_result():
            v = self.right()
            if v == 0 or v == 1:
                return True
        return False


class TrigonometricExponentiation(BinaryOperator): # This is used for trigonometric functions, which are unary operators, but the code does not need to be change nearly as much if they are treated as special kinds of exponentiation.  Exponentiate by 1 to use only the trigonometric function itself.  Some expressions may involve sin^2, so exponentiation is the binary operator of choice to convert the unary operator to a binary one.
    def __init__(self, func: callable, no_exp: bool = False, *args, **kwargs):
        if no_exp:
            right = 1.
            if 'right' in kwargs:
                del kwargs['right']
            super().__init__(*args, right=right, **kwargs)
        else:
            super().__init__(*args, **kwargs)
        self.func = func
    def __call__(self):
        if self.right == 0:
            return 1
        if isinstance(self.right, Operator) and self.right.has_fast_result() and self.right() == 0:
            return 1
        left, right = self.get_left_right()
        if left is None or right is None:
            return
        if isinstance(left, float) and isnan(left) or isinstance(right, float) and isnan(right):
            return nan
        if right == 0:
            return 1
        f_left = self.func(left)
        if f_left == 0:
            return 0
        if f_left == -1: # ??? This was put here for a reason, I can't remember why
            if abs(right) >= 200:
                return nan
        if isinstance(f_left, float) and isinstance(right, float):
            if abs(log2(abs(f_left)) * abs(right)) >= 247:
                return nan
            try:
                v = f_left ** right
            except OverflowError as e:
                print(f'Error in exponentiation {f_left}**{right}: {e}')
                print(log2(abs(f_left)), abs(right))
                print(log2(abs(f_left)) * abs(right))
                # input()
                return nan
            return v
    def symbolic(self):
        left, right = self.get_left_right()
        if left is not None and right is not None:
            if right == 0:
                return 1
            f_left = self.func(left)
            if f_left == 0:
                return 0
            if isinstance(f_left, float) and isinstance(right, float):
                if abs(log2(abs(f_left)) * abs(right)) >= 250:
                    return nan
        s_left, s_right = self.get_left_right_symbols()
        s_f_left = getattr(sp, self.func.__name__)(s_left)
        return sp.Pow(s_f_left, s_right)
    def has_fast_result(self) -> bool:
        if self.right == 0:
            return True
        if isinstance(self.right, Operator) and self.right.has_fast_result() and self.right() == 0:
            return True
        return False


Pi = Constant(np.pi, 'π')
Euler = Constant(np.e, 'e')
Phi = Constant(5**.5*.5+.5, 'φ')

all_constants = [Pi, Euler, Phi]

