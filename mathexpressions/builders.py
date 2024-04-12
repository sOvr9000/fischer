
from typing import Iterable, Optional, Union
import numpy as np
import sympy as sp

from .me import Operator, Constant, operator_names, operator_name_random_weights, all_constants



__all__ = ['ExpressionBuilder', 'AutomaticExpressionBuilder', 'RandomExpressionBuilder']




class ExpressionBuilder:
    '''
    Binary tree which consists of `Operator`, `Constant`, `int`, `sympy.Symbol`, and `None`.
    
    Facilitates building expressions to be fully computable (not having any `None` or any `sympy.Symbol`) or be representative of some parametrized function where each unique `sympy.Symbol` is a parameter to the function and each `None` is some scalar value denoted with unique upper case letters.

    If max_depth is `None`, then there is no limit to the depth of the operator tree.
    '''
    def __init__(self, max_depth: Optional[int] = 5):
        self.head: Operator = None
        self.max_depth = max_depth
        if self.max_depth is None:
            self.max_depth = np.inf
        self.history = []
    def has_head_operator(self) -> bool:
        return self.head is not None
    def can_evaluate(self) -> bool:
        return self.has_head_operator() and not any(node is None or isinstance(node, sp.Symbol) for node in self.all_nodes())
    def set_head_operator(self, op: str):
        self.head = Operator.from_str(op)
        self.head.depth = 0
        self.history.append(0)
    def get_last_added_node(self) -> Optional[Union[int, Operator, Constant]]:
        if not self.has_head_operator():
            return
        return self.get_node_at(self.history[-1])
    def get_node_at(self, i: int) -> Optional[Union[int, Operator, Constant]]:
        '''
        Return the `i`th node in DFS traversal order where left nodes are searched before right nodes.

        If `i` is too large, then the last node is returned.  If no empty nodes exist at all, then None is returned.
        '''
        node = None
        for k, node in enumerate(self.all_nodes()):
            if k == i:
                break
        return node
    def get_index_of_op(self, op: Operator) -> Optional[int]:
        for i, n in enumerate(self.all_nodes()):
            if n == op:
                return i
    def get_parent_of_empty_node(self, i: int) -> Operator:
        parent = None
        for k, (parent, _) in enumerate(self.all_nodes_with_parents()):
            if k == i:
                break
        return parent
    def set_node_as_op(self, i: int, op: str):
        if not self.has_head_operator():
            self.set_head_operator(op)
        else:
            node = self.get_parent_of_empty_node(i)
            j = self.get_index_of_op(node)
            node.set_empty_child_node(int(i != j + 1), Operator.from_str(op))
            node.update_depths_and_parents()
        self.history.append(i)
    def set_node_as_constant(self, i: int, c: Constant):
        node = self.get_parent_of_empty_node(i)
        j = self.get_index_of_op(node)
        node.set_empty_child_node(int(i != j + 1), c)
        self.history.append(i)
    def set_node_as_int(self, i: int, v: int):
        node = self.get_parent_of_empty_node(i)
        j = self.get_index_of_op(node)
        node.set_empty_child_node(int(i != j + 1), v)
        self.history.append(i)
    def set_node_as_symbol(self, i: int, sym: sp.Symbol):
        node = self.get_parent_of_empty_node(i)
        j = self.get_index_of_op(node)
        node.set_empty_child_node(int(i != j + 1), sym)
        self.history.append(i)
    def can_place_op_at(self, i: int) -> bool:
        if not self.has_head_operator():
            return True
        if i == 0:
            return not self.has_head_operator()
        node = self.get_parent_of_empty_node(i)
        return node.depth + 1 < self.max_depth
    def set_node_value(self, i: int, val: Union[int, Constant]):
        if isinstance(val, int):
            self.set_node_as_int(i, val)
        else:
            self.set_node_as_constant(i, val)
        try:
            if self.has_head_operator() and isinstance(self.head(), complex):
                raise ValueError
        except OverflowError as e:
            self.undo()
        except ZeroDivisionError as e:
            self.undo()
        except ValueError as e: # When complex numbers arise from evaluation
            self.undo()
    def count_empty_nodes(self) -> int:
        return len(list(self.empty_nodes()))
    def all_nodes(self) -> Iterable[Optional[Union[int, Operator, Constant, sp.Symbol]]]:
        if not self.has_head_operator():
            return
        yield from self.head.dfs()
    def all_nodes_with_parents(self) -> Iterable[tuple[Operator, Optional[Union[int, Operator, Constant]]]]:
        if not self.has_head_operator():
            return
        yield from self.head.dfs_with_parents()
    def empty_nodes(self) -> Iterable[int]:
        if not self.has_head_operator():
            yield 0
            return
        for i, node in enumerate(self.head.dfs()):
            if node is None:
                yield i
    def get_total_num_operators(self) -> int:
        if not self.has_head_operator():
            return 0
        return sum(op is Operator for op in self.all_nodes())
    def symbolic(self):
        if not self.has_head_operator():
            return sp.Expr()
        # print(f'Head node: {self.head}')
        return self.head.symbolic()
    def remove_node_at(self, i: int):
        if not self.has_head_operator():
            return
        if self.head.has_empty_children():
            self.head = None
            return
        for k, (parent, node) in enumerate(self.head.dfs_with_parents(include_empty_nodes=False)):
            if k == i:
                parent.remove_child_node(node)
                return
    def undo(self):
        if len(self.history) == 0:
            return
        i = self.history.pop()
        self.remove_node_at(i)
    def __str__(self) -> str:
        return f'Expression: {self.symbolic()}' + (f' = {self.head()}' if self.can_evaluate() else '')
    def __call__(self) -> float:
        return self.head()





class AutomaticExpressionBuilder:
    def __init__(self, expression_builder: Union[ExpressionBuilder, list[ExpressionBuilder]]):
        self.builder = expression_builder
        self.finished = False
    def __next__(self) -> None:
        if self.finished:
            raise StopIteration
        self.step()
    def __iter__(self):
        return self
    def step(self):
        raise NotImplementedError
    def finish(self):
        self.finished = True

class RandomExpressionBuilder(AutomaticExpressionBuilder):
    def __init__(self, expression_builder: Union[ExpressionBuilder, list[ExpressionBuilder]], max_int: int = 16, operators_only: bool = False):
        super().__init__(expression_builder)
        self.max_int = max_int
        self.operators_only = operators_only
    def random_op(self):
        return operator_names[np.random.choice(len(operator_names), p=operator_name_random_weights)]
    def random_constant(self):
        return all_constants[np.random.randint(len(all_constants))]
    def random_int(self):
        return np.random.randint(1, self.max_int+1)
    def step(self):
        if not self.builder.has_head_operator():
            self.builder.set_head_operator(self.random_op())
            return
        empty_nodes = list(self.builder.empty_nodes())
        if self.operators_only:
            op_indices = [i for i in empty_nodes if self.builder.can_place_op_at(i)]
            if len(op_indices) == 0:
                self.finish()
                return
            index = op_indices[np.random.randint(len(op_indices))]
            self.builder.set_node_as_op(index, self.random_op())
        else:
            if len(empty_nodes) == 0:
                self.finish()
                return
            index = empty_nodes[np.random.randint(len(empty_nodes))]
            is_operator = np.random.random() < 0.75 and self.builder.can_place_op_at(index)
            if is_operator:
                self.builder.set_node_as_op(index, self.random_op())
                return
            is_constant = np.random.random() < 0.1
            if is_constant:
                self.builder.set_node_as_constant(index, self.random_constant())
                return
            self.builder.set_node_as_int(index, self.random_int())
    def __str__(self) -> str:
        return f'Random builder: {self.builder}'


