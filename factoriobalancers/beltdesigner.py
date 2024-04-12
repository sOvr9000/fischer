
from .beltgraph import BeltGraph, Evaluation
from .beltturtle import BeltGrid, BeltTurtle



__all__ = ['BeltDesigner']



class BeltDesigner:
    def __init__(self, grid: BeltGrid):
        self.grid = grid
        self.turtles = []







class BeltDesigner:
    '''
    Facilitates the construction of belt balancers.

    Exposes information about where unconnected belts and splitters can be extended with another belt tile.

    Call `BeltDesigner.get_graph()` to construct a `BeltGraph` which is able to calculate flow (propagation and distribution).
    '''
    def __init__(self):
        self.reset()
    def reset(self):
        self.graph = BeltGraph()
        self.graph.add_edge(0, 1)
        self.graph.add_edge(1, 2)
        self.graph.add_edge(1, 3)
        self.graph.set_input(0)
        self.graph.set_output(2)
        self.graph.set_output(3)
        self.is_in_vertex_mode = True
        self.action_stack = ''
    def copy(self) -> 'BeltDesigner':
        des = BeltDesigner()
        des.graph = self.graph.copy_graph()
        des.is_in_vertex_mode = self.is_in_vertex_mode
        des.action_stack = self.action_stack
        return des
    def next_vertex(self):
        deepest = self.deepest_vertex()
        v = self.graph.new_vertex()
        self.graph.add_vertex(v)
        self.graph.disconnect_output(max(self.graph.out_vertices(deepest)))
        self.graph.add_edge(deepest, v)
        out1 = self.graph.new_vertex()
        self.graph.add_vertex(out1)
        out2 = self.graph.new_vertex()
        self.graph.add_vertex(out2)
        self.graph.add_edge(v, out1)
        self.graph.add_edge(v, out2)
        self.graph.set_output(out1)
        self.graph.set_output(out2)
        self.graph = self.graph.rearrange_vertices_by_depth()
    def preceding_vertex(self, amount):
        depths = self.graph.get_vertex_depths()
        preceding_vertex = self.deepest_vertex()
        for _ in range(amount):
            preceding_vertices = self.graph.in_vertices(preceding_vertex)
            if len(preceding_vertices) == 1:
                preceding_vertex = preceding_vertices[0]
            else:
                if depths[preceding_vertices[0]] < depths[preceding_vertices[1]]:
                    preceding_vertex = preceding_vertices[0]
                else:
                    preceding_vertex = preceding_vertices[1]
        return preceding_vertex
    def deepest_vertex(self):
        # if self.graph.num_vertices == 0:
        #     return 0
        depths = self.graph.get_vertex_depths()
        return max(((k, v) for k, v in depths.items() if not self.graph.is_output(k)), key=lambda t: t[1])[0]
    def connect_backward(self, amount_from, amount_to):
        if amount_from == amount_to:
            raise Exception(f'Cannot connect a vertex to itself.')
        pv_from = self.preceding_vertex(amount_from)
        pv_to = self.preceding_vertex(amount_to)
        self.graph.disconnect_output(self.graph.out_vertices(pv_from)[0])
        self.graph.add_edge(pv_from, pv_to)
        self.is_in_vertex_mode = False
    def action(self, a: str, reset: bool = True):
        if reset:
            # print(f'Executing "{a}" from reset state.')
            self.reset()
        else:
            pass
            # print(f'Executing "{a}" from current state.')
        if self.action_stack != '':
            self.action_stack += ' -> '
        self.action_stack += a
        split = a.split(' -> ')
        for _a in split:
            if _a == 'nv':
                self.next_vertex()
            elif _a[:2] == 'cb':
                f, t = map(int, _a.split(' ')[1:])
                self.connect_backward(f, t)
    def open_outputs(self):
        for v in self.graph.outputs:
            if all(self.graph.in_vertices(self.graph.in_vertices(v)[0])): # basically just find outputs that aren't the 1/2 flow output from the very first non-input vertex
                yield self.graph.in_vertices(v)[0]
    def open_inputs(self):
        for v in self.graph.vertices():
            if not self.graph.is_input(v) and not self.graph.is_output(v) and self.graph.in_degree(v) < 2:
                yield v
    def get_cb_amount(self, v: int) -> int:
        depths = self.graph.get_vertex_depths()
        m = max(d for k, d in depths.items() if not self.graph.is_output(k))
        return m - depths.get(v, m)
    def possible_backward_connections(self):
        for u in map(self.get_cb_amount, set(self.open_outputs())):
            for v in map(self.get_cb_amount, set(self.open_inputs())):
                if u < v:
                    yield u, v
    def bfs(self):
        self.action('nv', reset=True)
        visited = [self]
        queue: list[BeltDesigner] = [self]
        while len(queue) > 0:
            v = queue[0]
            del queue[0]

            yield v

            pcba = sorted(v.possible_backward_connections())
            for f, t in pcba:
                d = v.copy()
                d.action(f'cb {f} {t}', reset=False)
                # print(f'Checking new action stack: {d.action_stack}')
                if d not in visited:
                    visited.append(d)
                    queue.append(d)
            if v.is_in_vertex_mode:
                d = v.copy()
                d.action('nv', reset=False)
                # print(f'Checking new action stack: {d.action_stack}')
                if d not in visited:
                    visited.append(d)
                    queue.append(d)
    def get_denominators(self) -> list[int]:
        ev = self.graph.evaluate()
        output_flow = ev['output_flow']
        den_list = []
        for _, flow in output_flow.items():
            for _, frac in flow.items():
                den_list.append(frac.denominator)
        return list(sorted(set(den_list)))
    def __hash__(self):
        return hash(self.graph)
    def __eq__(self, other):
        if not isinstance(other, BeltDesigner):
            return False
        return hash(other) == hash(self)
    def __repr__(self):
        return f'{repr(self.graph.evaluate())}\nOpen outputs: {list(sorted(set(self.open_outputs())))}\n(open outputs depth regression): {list(sorted(set(map(self.get_cb_amount, self.open_outputs()))))}\nOpen inputs: {list(sorted(set(self.open_inputs())))}\n(open inputs depth regression): {list(sorted(set(map(self.get_cb_amount, self.open_inputs()))))}\nPossible cb pairs: {list(sorted(self.possible_backward_connections()))}'


