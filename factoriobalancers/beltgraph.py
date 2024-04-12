
import os
import base64
import zlib
from copy import deepcopy
from graph_tools import Graph
from fischer.factoriobps import load_blueprint_string_from_file, is_blueprint_data, load_blueprint_string, get_entity_map, transpose_balancer_bp
from .metrics import measure_accuracy
from .sparsesolver import calculate_flow_points, pretty_relations, construct_relations

from typing import Optional, Union



__all__ = [
    'Evaluation', 'BeltGraph',
    'list_vertex_pairs', 'compact_string_to_edges_string',
    'edges_string_to_compact_string', 'derive_graph',
    'try_derive_graph', 'add_balancer', 'get_balancer',
    'is_balancer_defined', 'common_balancers',
]


class Evaluation(dict):
    display_flow_points = False
    def __repr__(self):
        s = f'Accuracy: {self["accuracy"]:.5f}\nError: {self["error"]:.5f}\nScore: {self["score"]:.5f}\n'
        s += f'Bottlenecks: {self["bottlenecks"]}\n'
        s += f'Output flow:\n{pretty_relations(self["output_flow"])}\n'
        if self.display_flow_points:
            s += f'Flow points:\n{pretty_relations(self["flow_points"])}\n'
        return s



class BeltGraph(Graph):
    def __init__(self):
        super().__init__(directed=True, multiedged=False)
        self.inputs = []
        self.outputs = []
    @property
    def num_inputs(self):
        return len(self.inputs)
    @property
    def num_outputs(self):
        return len(self.outputs)
    @property
    def num_vertices(self):
        return len(self.vertices())
    @property
    def num_internal_edges(self):
        '''
        The number of belt connections between any two splitters in a Factorio belt balancer that this graph represents.
        '''
        return self.num_edges - self.num_inputs - self.num_outputs
    @property
    def num_internal_vertices(self):
        '''
        The number of splitters in a Factorio belt balancer that this graph represents.
        '''
        return self.num_vertices - self.num_inputs - self.num_outputs
    @property
    def num_edges(self):
        return len(self.edges())
    @property
    def balancer_type(self):
        return f'{self.num_inputs}-{self.num_outputs}'
    @property
    def summary(self):
        return f'{self.balancer_type} [V:{self.num_internal_vertices} E:{max(0,self.num_internal_edges)}]'
    def internal_edges(self):
        for u, v in self.edges():
            if self.is_input(u):
                continue
            if self.is_output(v):
                continue
            yield u, v
    def internal_vertices(self):
        for u in self.vertices():
            if not self.is_input(u) and not self.is_output(u):
                yield u
    def set_input(self, u: int, flag: bool = True) -> None:
        '''
        Define a vertex of the graph to be an input.

        If vertex `u` does not exist yet, then a new vertex is created with that id as an input vertex.
        '''
        if flag:
            if not self.is_input(u):
                self.inputs.append(u)
                self.add_vertex(u)
        else:
            if self.is_input(u):
                self.inputs.remove(u)
                self.add_vertex(u)
    def set_output(self, u: int, flag: bool = True) -> None:
        '''
        Define a vertex of the graph to be an output.

        If vertex `u` does not exist yet, then a new vertex is created with that id as an output vertex.
        '''
        if flag:
            if not self.is_output(u):
                self.outputs.append(u)
                self.add_vertex(u)
        else:
            if self.is_output(u):
                self.outputs.remove(u)
                self.add_vertex(u)
    def set_num_inputs(self, new_num_inputs: int) -> None:
        while self.num_inputs > new_num_inputs:
            self.set_input(self.inputs[0], flag=False)
        while self.num_inputs < new_num_inputs:
            self.set_input(self.new_vertex(), flag=True)
    def set_num_outputs(self, new_num_outputs: int) -> None:
        while self.num_outputs > new_num_outputs:
            self.set_output(self.outputs[0], flag=False)
        while self.num_outputs < new_num_outputs:
            self.set_output(self.new_vertex(), flag=True)
    def get_next_unused_input(self):
        for u in self.inputs:
            if self.out_degree(u) == 0:
                return u
    def get_next_unused_output(self):
        for u in self.outputs:
            if self.in_degree(u) == 0:
                return u
    def is_input(self, u):
        return u in self.inputs
    def is_output(self, u):
        return u in self.outputs
    def get_outbound_limit(self, u):
        if self.is_input(u):
            return 1
        if self.is_output(u):
            return 0
        return 2
    def get_inbound_limit(self, u):
        if self.is_input(u):
            return 0
        if self.is_output(u):
            return 1
        return 2
    def is_functional(self, u):
        '''
        Return whether vertex `u` is a balancer, splitter, or merger.  Returns False when the vertex has one or fewer inbound edges and outbound edges, which would either cause jams in the flow or doesn't propagate flow.
        '''
        return self.vertex_degree(u) >= 3
    def is_disconnected(self, u):
        '''
        Return whether vertex `u` has no inbound nor outbound edges.
        '''
        return self.vertex_degree(u) == 0
    def is_partially_disconnected(self, u):
        '''
        Return whether vertex `u` has either no inbound, no outbound edges, or both.
        '''
        return self.in_degree(u) == 0 or self.out_degree(u) == 0
    def is_splitter(self, u):
        '''
        Return whether vertex `u` represents a 1-2 balancer.
        '''
        return self.out_degree(u) == 2 and self.in_degree(u) == 1
    def is_merger(self, u):
        '''
        Return whether vertex `u` represents a 2-1 balancer.
        '''
        return self.out_degree(u) == 1 and self.in_degree(u) == 2
    def is_balancer(self, u):
        '''
        Return whether vertex `u` represents a 2-2 balancer.
        '''
        return self.vertex_degree(u) == 4
    def is_identity(self, u):
        '''
        Return whether vertex `u` represents a 1-1 "balancer", which has no effect on flow in the graph.
        '''
        return self.in_degree(u) == 1 and self.out_degree(u) == 1
    def is_empty(self) -> bool:
        '''
        Return whether the graph has no inputs or outputs.

        Returns True when there is at least one input or at least one output.
        '''
        return self.num_inputs == 0 and self.num_outputs == 0
    def copy_graph(self):
        graph = deepcopy(self)
        # graph.inputs = self.inputs[:]
        # graph.outputs = self.outputs[:]
        return graph
    # def permute_vertices(self):
    # 	vertices = self.vertices()
    # 	perm = vertices[:]
    # 	shuffle(perm)
    # 	perm_map = dict(zip(vertices, perm))
    # 	self.inputs = list(map(perm_map.__getitem__, self.inputs))
    # 	self.outputs = list(map(perm_map.__getitem__, self.outputs))
    # 	self.edges = [
    # 		(perm_map[u], perm_map[v])
    # 		for u, v in self.edges
    # 	]
    #	self.vertices = perm
    def can_add_edge_or_combine(self, u, v):
        if u == v:
            return False
        return True
    def add_edge_or_combine(self, u, v):
        if u == v:
            raise Exception('Cannot connect a vertex to itself.')
        if self.has_edge(u, v):
            self.combine_vertices(u, v)
        else:
            self.add_edge(u, v)
    def can_add_edge(self, u, v):
        if u == v:
            return False
        if self.out_degree(u) >= self.get_outbound_limit(u):
            return False
        if self.in_degree(v) >= self.get_inbound_limit(v):
            return False
        if self.has_edge(u, v):
            return False
        return True
    def add_edge(self, u, v):
        if u == v:
            raise Exception('Cannot connect a vertex to itself.')
        if self.out_degree(u) >= self.get_outbound_limit(u):
            raise Exception(f'Vertex {self.vertex_to_str(u)} has too many outbound edges.')
        if self.in_degree(v) >= self.get_inbound_limit(v):
            raise Exception(f'Vertex {self.vertex_to_str(v)} has too many inbound edges.')
        if self.has_edge(u, v):
            raise Exception('Cannot create duplicate edges.')
        super().add_edge(u, v)
    def can_insert_vertex(self, u, v):
        if not self.has_edge(u, v):
            return False
        return True
    def insert_vertex(self, u, v):
        if not self.has_edge(u, v):
            raise Exception(f'There is no edge from vertex {self.vertex_to_str(u)} to vertex {self.vertex_to_str(v)}.')
        nv = self.new_vertex()
        self.delete_edge(u, v)
        self.add_edge(u, nv)
        self.add_edge(nv, v)
    def insert_graph(self, graph: 'BeltGraph') -> list[tuple, tuple]:
        # print(f'Inserting {graph}')
        nv_map = [None] * (max(graph.vertices()) + 1)
        inputs = []
        outputs = []
        for u in graph.vertices():
            if graph.is_input(u) or graph.is_output(u):
                continue
            v = self.new_vertex()
            nv_map[u] = v
            # print(u, '->', v)
            self.add_vertex(v)
        for u in graph.inputs:
            inputs.append(nv_map[graph.out_vertices(u)[0]])
        for v in graph.outputs:
            outputs.append(nv_map[graph.in_vertices(v)[0]])
        for u, v in graph.edges():
            if graph.is_input(u) or graph.is_output(v):
                continue
            self.add_edge(nv_map[u], nv_map[v])
        # print(f'Returning {inputs}, {outputs}')
        return [tuple(inputs), tuple(outputs)]
    def out_vertices(self, u: int) -> Union[tuple, tuple[int], tuple[int, int]]:
        out_edges = self.out_edges(u)
        if len(out_edges) == 0 or out_edges[0][1] is None:
            return tuple()
        if len(out_edges) == 1:
            return out_edges[0][1],
        if out_edges[0][1] is None:
            if out_edges[1][1] is None:
                return tuple()
            return out_edges[1][1],
        if out_edges[1][1] is None:
            return out_edges[0][1],
        return out_edges[0][1], out_edges[1][1]
    def in_vertices(self, u: int) -> Union[tuple, tuple[int], tuple[int, int]]:
        in_edges = self.in_edges(u)
        if len(in_edges) == 0 or in_edges[0][0] is None:
            return tuple()
        if len(in_edges) == 1:
            return in_edges[0][0],
        if in_edges[0][0] is None:
            if in_edges[1][0] is None:
                return tuple()
            return in_edges[1][0],
        if in_edges[1][0] is None:
            return in_edges[0][0],
        return in_edges[0][0], in_edges[1][0]
    def doubled(self) -> 'BeltGraph':
        '''
        Return a similar graph with double the inputs and double the outputs by substituting all 2-2 balancers (single vertices in current graph) with 4-4 balancers.
        '''
        r = self.rearrange_vertices_by_depth()
        r.simplify()
        double_balancer = BeltGraph()
        double_balancer.load_blueprint_string('0eNqlltuOgjAQhl/FzDUYWw6lJLsvYszGQ7NpgoW0xWgM775F3KPMKg43hNJ+8zPTv8wZNlWrGquNh/IMelsbB+XyDE6/m3XVj/lTo6CEg7a+DSMRmPW+HxhmxCl0EWizU0coWRc9uZJ3qwiU8dprNQi4PJzeTLvfKBvQX6vVsbHKubgNK+27rcM93qjKB35TuwCoTR88QHMxzyI4QRkzPs9CsKuguvVN66EX+ycKv4ni7dq4prYei5H/ijHCTCjMZJyZTmeKe8xsOlPe+/b8maoVU6smpisv7mWjoGQDYUpK1dglFztt1XaYkI9EYAvKZmPjstmt91xTae/Dy5HMXmELBMYpicUUJpQdcJNZPhaCYjmJyM4I1ZIPqc5Jx2bx03/aIPZjgvAZBZIaiv/kQ1aRhD2DFJQvCKqRTHBGOkMfKyHnT1hcIIIpPzyMmU7RJwZWjrAoPzpMXz5Fn/xfH8VOGcIsCKcXxqQYCGEmFANdmKGN1F7tA+C7sY2gWofFYWx57UXjoQV9+exEV7NZHK7XGTohQA7KuuGkLVgqJBciETIr0q77AA3PvFE=')
        # double_balancer.load_blueprint_string('0eNqllu1ugyAUhu/l/KaNgBb1VpZl6QdpSCwawKVN470PdVvW6Zmd55fBj+e8vOdFuMOhanXjjA1Q3sEca+uhfLmDN2e7r/p74dZoKMEEfQEGdn/pR/raOO39Jri99U3twuagqwAdA2NP+gol714ZaBtMMHokDoPbm20vB+3iCxOWbyoTQnzGoKl9/LC2ff0I23AhGdygFEJts65jE5r4H43/TZNL85xVuM0+qfk8NaVRi3lqtoYqlrTuaFREq5pQ25gWd3Z1vOLc5EEt+0pk3Yam7TM3qZOvUc+XPCloVMQTnqzBJg9YBifj9HF8ZTdXhJOyJxNEu6BhOYKVpPShalOS0yg2o2ExE3akuA1ql3OxckXyB/VPrEie01oqfs9GzBUpaEUksrcktJQLBEtck5haQfzJDuDvlhqLdFRIWuwxV1Ja7J8KisiosX/SI9o2KlPEI0VLDobNaR3FsLR9c8TGw+RwAC1/nFcZvGvnxybnPFWFUEqqIsvTrvsAHuiNhQ==')
        double_balancer = double_balancer.rearrange_vertices_by_depth()
        double_splitter = BeltGraph()
        double_splitter.load_blueprint_string('0eNqlVtuOgyAU/JfzjE0BFfFXNpuNtqQhUSSAmxrjvy/apmm2klp5MhiYmTNnuIxQN73QRioH5Qjy1CkL5dcIVl5U1cz/3KAFlCCdaAGBqtp5JK7aCGsTZypldWdcUovGwYRAqrO4QokntBnE6kY6J8zTcjJ9IxDKSSfFTdAyGH5U39Z+ZonfSUGgO+uXd2rm95AJxsUhQzB4dJIdsmlW+A+V7EFl71BpnFaP7405SyNOtyn5Ckf6wtF7J83FdP67hSVfWO596nqn+7mdLzxZnEMvtZAVjnxfLezTWtieWviDhW+ppQinfbUfd+xiPUk8Lkl8HRUf98DmYSPWAopxXHJC2klUE+kxAEt39Y3iAFzs9lx0PiItVSDROItzIyQ/jwpI0GQWBxtSW0SeH1vNjtuNlAbuoONn0eN3OBKAi9t3QZXkM5XsWaW/25eXQPn0+kDwK4y9naMFThknjFHGsyKdpj/P3Nc5')
        double_splitter = double_splitter.rearrange_vertices_by_depth()
        double_merger = BeltGraph()
        double_merger.load_blueprint_string('0eNqllt2OgyAQhd9lrrERRBFfpdls+kMaEosEcFNjfPeFtrtpVtm2cmUw8M3xzJngCPu2F9pI5aAZQR46ZaHZjmDlSe3a8M4NWkAD0okzIFC7c1iJizbC2syZnbK6My7bi9bBhECqo7hAgyf0MsTqVjonzMNxMn0gEMpJJ8VN0HUxfKr+vPc7G/xMCgLdWX+8U6G+R2aY8E2JYPD0gmzKKSj8QyVrqPUzahH/4kWVd1qxTKMrNBb5r8ZymVq+pbHI7zS6TKtmtN731ZxM55+vOBlUop/UdL3TfQjXrA57z9n6f9X1OtX8XdV8TcrYsw7iPC0YVQSL02aiunpylEYcblvIUhGSNs6zItVSkZVzyCLG0LQ2xvwuU2eHP6ZQqkgIcZXW1zoin6V1MoZNns0XXeFJrlAcuVfyJFeiWLwq0TQPOH/BXq/j5uEXAMGXMPY2pjWmjBPGCsbLmk7TN4FgrxE=')
        double_merger = double_merger.rearrange_vertices_by_depth()
        new_graph = BeltGraph()
        new_graph.set_num_inputs(r.num_inputs * 2)
        new_graph.set_num_outputs(r.num_outputs * 2)

        sub_graphs = {}
        def get_sub_graph(u: int):
            if u not in sub_graphs:
                if r.is_balancer(u):
                    the_graph = double_balancer
                elif r.is_merger(u):
                    the_graph = double_merger
                elif r.is_splitter(u):
                    the_graph = double_splitter
                elif r.is_partially_disconnected(u):
                    return None
                else:
                    raise Exception(f'Cannot get a subgraph for vertex {r.vertex_to_str(u)}\n{r.balancer_type}\n{r.in_degree(u)} {r.out_degree(u)}')
                sub_graphs[u] = new_graph.insert_graph(the_graph)
            return sub_graphs[u]
        
        def connectable_subgraph_outputs(sg: tuple[tuple, tuple]):
            for out_v in sg[1]:
                if new_graph.out_degree(out_v) < new_graph.get_outbound_limit(out_v):
                    yield out_v

        def connectable_subgraph_inputs(sg: tuple[tuple, tuple]):
            for in_v in sg[0]:
                if new_graph.in_degree(in_v) < new_graph.get_inbound_limit(in_v):
                    yield in_v
        
        combined_vertices = []

        def connect_subgraphs(from_sg: list[tuple, tuple], to_sg: list[tuple, tuple]):
            k = False
            for g in range(len(from_sg[1])):
                i = from_sg[1][g]
                if i in combined_vertices:
                    continue
                for j in to_sg[0]:
                    if j in combined_vertices:
                        continue
                    if new_graph.has_edge(i, j) and new_graph.can_combine_vertices(i, j):
                        # print(new_graph)
                        new_graph.combine_vertices(i, j)
                        # print(f'combined {i} and {j}')
                        from_sg[1] = tuple(j if u == i else u for u in from_sg[1])
                        # sub_graphs[i] = get_sub_graph(j)
                        # print(new_graph)
                        combined_vertices.append(i)
                        # combined_vertices.append(j)
                        if k:
                            return
                        k = True
                    else:
                        if new_graph.can_add_edge(i, j):
                            new_graph.add_edge(i, j)
                            if k:
                                # print('double connection')
                                return
                            k = True

        visited_edges = []

        # print(r)
        for u in r.vertices():
            # print(u)
            if not r.is_input(u) and not r.is_output(u):
                # if new_graph.is_output(u):
                # 	print(f'new graph output {u}')
                # 	continue
                sg = get_sub_graph(u)
                # if sg is None:
                # 	continue
                # if r.is_partially_disconnected(u):
                # 	print(u)
                # 	print(r)

                for v in r.in_vertices(u):
                    # if new_graph.is_output(v): # This can happen on (m+3)-(n+2) balancers being converted to m-n balancers, for example 
                    # 	continue
                    if r.is_input(v):
                        vs = list(connectable_subgraph_inputs(sg))
                        for _ in range(2):
                            # if len(vs) == 0:
                            # 	break
                            _u = new_graph.get_next_unused_input()
                            _v = vs.pop()
                            new_graph.add_edge(_u, _v)
                    else:
                        if (v, u) not in visited_edges:
                            visited_edges.append((v, u))
                            from_sg = get_sub_graph(v)
                            # if from_sg is None:
                            # 	continue
                            connect_subgraphs(from_sg, sg)

                # print(r.out_vertices(u))
                for v in r.out_vertices(u):
                    # print(u, v)
                    # if new_graph.is_input(v): # This can happen on (m+3)-(n+2) balancers being converted to m-n balancers, for example 
                    # 	continue
                    if r.is_output(v):
                        # print(v)
                        vs = list(connectable_subgraph_outputs(sg))
                        # print('DOUBLING: len(vs) = ' + str(len(vs)))
                        for _ in range(2):
                            _u = vs.pop()
                            _v = new_graph.get_next_unused_output()
                            new_graph.add_edge(_u, _v)
                            # print(f'DOUBLING: Connected {_u} to output {_v}')
                            # new_graph.debug_vertex(_v)
                    else:
                        # if new_graph.is_output(v):
                        # 	continue
                        if (u, v) not in visited_edges:
                            visited_edges.append((u, v))
                            to_sg = get_sub_graph(v)
                            # if to_sg is None:
                            # 	continue
                            # print(sg, to_sg)
                            connect_subgraphs(sg, to_sg)
                            # print(sg, to_sg)
        
        # for v in r.outputs:
        # 	if r.in_degree(v) == 1:
        # 		continue
        # 	for u in r.in_vertices(v):
        # 		print(u)
        
        return new_graph
    def transposed(self) -> 'BeltGraph':
        '''
        Return a new graph with all edges reversed, where the inputs and outputs are swapped.
        '''
        graph = self.copy_graph()
        graph.inputs, graph.outputs = graph.outputs, graph.inputs
        edges = list(graph.edges())
        for u, v in edges:
            graph.delete_edge(u, v)
        # Separate the loops because edges can be added too soon, causing an error where a vertex has either three inputs or three outputs.
        for u, v in edges:
            graph.add_edge(v, u)
        graph = graph.rearrange_vertices_by_depth()
        return graph
    def debug_vertex(self, u: int) -> None:
        print(f'{self.vertex_to_str(u)}: [{", ".join(map(self.vertex_to_str, self.in_vertices(u)))}] -> {self.vertex_to_str(u)} -> [{", ".join(map(self.vertex_to_str, self.out_vertices(u)))}]')
    def can_combine_vertices(self, u, v) -> bool:
        return not self.is_input(u) and not self.is_output(v) and self.out_degree(v) > 0 and ((self.out_degree(u) < 2 and self.in_degree(v) < 2) or self.out_vertices(u) == (v, v))
    def combine_vertices(self, u, v):
        '''
        Destroy certain edges and create new ones such that the result is equivalent to there being two identical edges from u to v.

        In the context of belts, this is the act of connecting the two outputs of one splitter to the two inputs of another splitter,
        so that effectively nothing is changed about the flow of the graph.

        One would prefer to delete one of those splitters and connect the belts accordingly so that the same flow is achieved with fewer splitters ("combining" the splitters into one).
        '''
        assert self.can_combine_vertices(u, v), f'Cannot combine vertices {self.vertex_to_str(u)} and {self.vertex_to_str(v)}.' + ('' if self.out_degree(v) > 0 else f' Vertex {self.vertex_to_str(v)} has no outbound edges.  Are you combining a series of vertices such that vertices are being removed and then referenced later for combination?')
        if self.has_edge(u, v):
            self.delete_edge(u, v)
        in_vertices = self.in_vertices(u)
        for _u in in_vertices:
            self.delete_edge(_u, u)
            if _u == v:
                continue
            self.add_edge(_u, v)
    def insert_graph_between(self, graph: 'BeltGraph', from_vertices, to_vertices):
        if graph.num_inputs != len(from_vertices):
            raise Exception(f'Graph has incorrect number of input vertices. Received {graph.num_inputs}, expected {len(from_vertices)}')
        if graph.num_outputs != len(to_vertices):
            raise Exception(f'Graph has incorrect number of output vertices. Received {graph.num_outputs}, expected {len(to_vertices)}')
        # for u in from_vertices:
        # 	for v in to_vertices:
        # 		if self.has_edge(u, v):
        # 			self.delete_edge(u, v)
        input_vertices, output_vertices = self.insert_graph(graph)
        # print(f'Input vertices added by new graph: {input_vertices}')
        # print(f'Output vertices added by new graph: {output_vertices}')
        # print(f'Trying to connect from vertices: {from_vertices}')
        # print(f'Trying to connect to vertices: {to_vertices}')
        for u, v in zip(from_vertices + list(output_vertices), list(input_vertices) + to_vertices):
            if self.can_add_edge(u, v):
                self.add_edge(u, v)
    def disconnect_input(self, u: int) -> int:
        '''
        Delete the input from the graph, delete its edge, and return the vertex to which the input is connected.

        If it is not connected, then return `-1`.
        '''
        v = -1
        if self.out_degree(u) == 1:
            v = self.out_vertices(u)[0]
            self.delete_edge(u, v)
        self.set_input(u, False)
        self.delete_vertex(u)
        return v
    def disconnect_output(self, u: int) -> int:
        '''
        Delete the output from the graph, delete its edge, and return the vertex that is connected to it.

        If it is not connected, then return `-1`.
        '''
        v = -1
        if self.in_degree(u) == 1:
            v = self.in_vertices(u)[0]
            self.delete_edge(v, u)
        self.set_output(u, False)
        self.delete_vertex(u)
        return v
    def disconnect_num_inputs(self, amount: int) -> list[int]:
        assert self.num_inputs >= amount, f'Cannot disconnect {amount} inputs from the current graph with {self.num_inputs} total inputs'
        vs = []
        for _ in range(amount):
            v = self.disconnect_input(self.inputs[-1])
            if v == -1:
                continue
            vs.append(v)
        return vs
    def disconnect_num_outputs(self, amount: int) -> list[int]:
        assert self.num_outputs >= amount, f'Cannot disconnect {amount} outputs from the current graph with {self.num_outputs} total outputs'
        vs = []
        for _ in range(amount):
            v = self.disconnect_output(self.outputs[-1])
            if v == -1:
                continue
            vs.append(v)
        return vs
    def disconnect_vertex(self, u: int) -> Union[int, tuple[tuple, tuple]]:
        '''
        Call `disconnect_input(u)` or `disconnect_output(u)` if appropriate.
        
        Otherwise, preform the same operations on an intermediate vertex and return all relevant vertices as a tuple of tuples, which removes the vertex and all of its edges.

        The returned tuple consists of two tuples which either contain zero, one, or two integers, based on how the vertex was connected before removal.
        '''
        if self.is_input(u):
            return self.disconnect_input(u)
        if self.is_output(u):
            return self.disconnect_output(u)
        us = self.in_vertices(u)
        for _u in us:
            self.delete_edge(_u, u)
        vs = self.out_vertices(u)
        for _v in vs:
            self.delete_edge(u, _v)
        self.delete_vertex(u)
        return us, vs
    def delete_identity_vertex(self, u: int) -> None:
        '''
        Delete a vertex that has exactly one inbound edge and one outbound edge.  Replace the vertex with an edge that connects its input to its output.
        '''
        assert self.is_identity(u), f'The vertex {u} is not an identity type of vertex.  It does not have exactly one inbound edge and one outbound edge.'
        i = self.in_vertices(u)[0]
        j = self.out_vertices(u)[0]
        # self.delete_edge(i, u)
        # self.delete_edge(u, j)
        self.delete_vertex(u)
        if self.has_edge(i, j):
            self.combine_vertices(i, j)
        else:
            self.add_edge(i, j)
    def delete_balancer_vertex(self, u: int, swap: bool = False) -> None:
        '''
        Delete a vertex which represents a 2-2 balancer (having two inbound edges and two outbound edges). Delete and reconnect edges accordingly to simplify the graph.

        If `swap=True`, then reconnect edges in the other possible way.
        '''
        assert self.is_balancer(u), f'The vertex {u} is not a balancer type of vertex.  It does not have exactly two inbound edges and two outbound edges.'
        (in1, _), (in2, _) = self.in_edges(u)
        (_, out1), (_, out2) = self.out_edges(u)
        if swap or in1 == out1 or in2 == out2:
            out1, out2 = out2, out1
        if in1 == in2:
            raise Exception(f'Unhandled case')
        self.delete_vertex(u)
        self.add_edge_or_combine(in1, out1)
        self.add_edge_or_combine(in2, out2)
    def simplify(self):
        while True:
            v = next(self.removable_vertices(), -1)
            if v == -1:
                break
            if self.is_identity(v):
                self.delete_identity_vertex(v)
            elif self.is_balancer(v):
                self.delete_balancer_vertex(v)
    def new_vertex(self):
        vertices = self.vertices()
        for u in range(len(vertices)+1):
            if u not in vertices:
                return u
    def rearrange_vertices(self, vertex_map: list[int]) -> 'BeltGraph':
        graph = BeltGraph()
        for u in self.inputs:
            graph.set_input(vertex_map[u])
        for v in self.outputs:
            graph.set_output(vertex_map[v])
        for u, v in self.edges():
            graph.add_edge(vertex_map[u], vertex_map[v])
        return graph
    def rearrange_vertices_by_depth(self, depths: dict = None) -> 'BeltGraph':
        if depths is None:
            depths = self.get_vertex_depths()
        vmap = [None] * (max(self.vertices()) + 1)
        for i, (_, u) in enumerate(sorted((d, _u) for _u, d in depths.items())):
            vmap[u] = i
        for u in self.vertices():
            if vmap[u] is None:
                vmap[u] = max(v for v in vmap if v is not None) + 1
        # print(list(sorted(v for v in vmap if v is not None)))
        return self.rearrange_vertices(vmap)
    def removable_vertices(self):
        for u in self.vertices():
            if self.is_identity(u):
                yield u
            if self.is_balancer(u):
                graph = self.copy_graph()
                graph.delete_balancer_vertex(u)
                if graph.is_solved():
                    yield u
    def possible_new_edges(self):
        '''Generator which yields a pair of vertices if an edge can be created between the two vertices, along with pairs between existing vertices and a possible new vertex.'''
        vertices = self.vertices()
        av_vertices = len(vertices) - self.num_inputs - self.num_outputs
        nv = self.new_vertex()
        if av_vertices == 0:
            yield nv, nv+1
            return
        if av_vertices == 1:
            yield vertices[0], nv
            return
        aug_vertices = vertices + [nv]
        for i, u in enumerate(aug_vertices[:-1]):
            for v in aug_vertices[i+1:]:
                if self.can_add_edge(u, v):
                    yield u, v
                if self.can_add_edge(v, u):
                    yield v, u
    def likely_new_edges(self):
        '''
        Generator which yields a pair of vertices only if the edge created between the two vertices would make progress toward functionalizing one or both of the two vertices.
        If no such pairs exist, then yield from possible_new_edges().
        
        This is useful for constructing belt graphs where the vertices are prioritized to be either splitters, balancers, or mergers, rather than letting them be nonfunctional vertices in the graph.
        '''
        ne = list(self.possible_new_edges())
        vertices = self.vertices()
        nonfunctional_vertices = [u for u in vertices if not self.is_functional(u)]
        new_ne = []
        if len(nonfunctional_vertices) > 0:
            for i in range(len(ne)-1,-1,-1):
                u, v = ne[i]
                if u in nonfunctional_vertices or v in nonfunctional_vertices:
                    new_ne.append((u, v))
        # if len(new_ne) > 4:
        # 	ne = new_ne
        yield from ne
    def possible_actions(self):
        for edge in self.possible_new_edges():
            yield 'add_edge', edge
        for edge in self.edges():
            yield 'delete_edge', tuple(edge)
    def do_action(self, action):
        name, args = action
        if name == 'add_edge':
            self.add_edge(*args)
        elif name == 'delete_edge':
            self.delete_edge(*args)
    def evaluate(self):
        '''
        Compute the flow of the graph and see if it's accurate to the balancer type (number of inputs and number of outputs), computing and returning multiple metrics.
        '''
        flow_points = self.calculate_flow_points()
        output_flow = {k: v for k, v in flow_points.items() if self.is_output(k)}
        bottlenecks = [k for k, v in sorted(flow_points.items()) if sum(v.values()) > 1]
        accuracy, error, score = measure_accuracy(output_flow, self.inputs, self.outputs)
        evaluation = Evaluation()
        evaluation.update({
            'flow_points': flow_points,
            'output_flow': output_flow,
            'bottlenecks': bottlenecks,
            'accuracy': accuracy,
            'error': error,
            'score': score,
        })
        return evaluation
    def calculate_flow_points(self) -> dict[int, list]:
        return calculate_flow_points(*construct_relations(self))
    def is_solved(self, ev: Evaluation = None):
        '''
        Return whether the graph correctly distributes flow based on the given number of inputs and outputs and without bottlenecks if it doesn't have fewer outputs than inputs.
        '''
        if ev is None:
            ev = self.evaluate()
        return ev['accuracy'] == 1 and (self.num_outputs < self.num_inputs or len(ev['bottlenecks']) == 0)
    def get_vertex_depths(self):
        def dfs(u, d):
            # if self.in_degree(u) == 0:
            # 	dfs.depth[u] = 0
            # 	return 0
            if u in dfs.depth:
                old = dfs.depth[u]
                dfs.depth[u] = min(dfs.depth[u], d)
                if d < old:
                    for v in self.out_vertices(u):
                        dfs(v, d + 1)
            else:
                dfs.depth[u] = d
                for v in self.out_vertices(u):
                    dfs(v, d + 1)
        dfs.depth = {}
        for u in self.inputs:
            dfs.visited = [False] * self.num_vertices
            dfs(u, 0)
        return dfs.depth
    def get_vertices_by_depth(self, depth: dict = None) -> list[list[int]]:
        if depth is None:
            depth = self.get_vertex_depths()
        if len(depth) == 0:
            return []
        transpose = [[] for _ in range(max(depth.values()) + 1)]
        for u, d in depth.items():
            transpose[d].append(u)
        return transpose
    def separated(self) -> list['BeltGraph']:
        '''
        Return a list of subgraphs within this graph which are disjoint.  For example, if there's a 1-2 graph and a 4-4 graph contained within this graph, then this function would return a list of two graphs, one for each of the two subgraphs.
        '''
        graphs = []
        def get_connected_graph(u: int) -> BeltGraph:
            # Use DFS to extract a connected subgraph containing vertex u.
            graph = BeltGraph()
            visited = [False] * (self.num_vertices * 2)
            def dfs(v: int):
                if visited[v]:
                    return
                visited[v] = True
                graph.add_vertex(v)
                for w in self.out_vertices(v):
                    if graph.can_add_edge(v, w):
                        graph.add_edge(v, w)
                    dfs(w)
                for w in self.in_vertices(v):
                    if graph.can_add_edge(w, v):
                        graph.add_edge(w, v)
                    dfs(w)
            dfs(u)
            # Mark all vertices with no inbound or outbound edges as inputs or outputs.
            for v in graph.vertices():
                if graph.in_degree(v) == 0:
                    graph.set_input(v, True)
                if graph.out_degree(v) == 0:
                    graph.set_output(v, True)
            return graph
        for u in self.vertices():
            if self.in_degree(u) == 0:
                graphs.append(get_connected_graph(u))
        return graphs
    @property
    def edges_string(self):
        return '{};{};{}'.format(
            ",".join(map(str,self.inputs)),
            ",".join(map(str,self.outputs)),
            ' '.join(
                '{}:{}'.format(
                    u,
                    ','.join(
                        str(v)
                        for _, v in self.out_edges(u)
                    )
                )
                for u in self.vertices()
                if self.out_degree(u) > 0
            )
        )
    @property
    def compact_string(self):
        return edges_string_to_compact_string(self.edges_string)
    def clear(self):
        for v in self.vertices():
            self.delete_vertex(v)
        self.inputs.clear()
        self.outputs.clear()
    def load_edges_string(self, edges_string):
        self.clear()
        inputs_str, outputs_str, neighbors_str_list = edges_string.split(';')
        for u_str in inputs_str.split(','):
            self.set_input(int(u_str), True)
        for v_str in outputs_str.split(','):
            self.set_output(int(v_str), True)
        for neighbors_str in neighbors_str_list.split(' '):
            v_str, u_str = neighbors_str.split(':')
            v = int(v_str)
            if ',' in u_str:
                u1, u2 = u_str.split(',')
                self.add_edge(v, int(u1))
                self.add_edge(v, int(u2))
            else:
                self.add_edge(v, int(u_str))
    def load_compact_string(self, compact_string):
        self.load_edges_string(compact_string_to_edges_string(compact_string))
    def load_blueprint_string(self, blueprint_string, verbose=True):
        try_derive_graph(load_blueprint_string(blueprint_string), self, verbose=verbose)
    def load_blueprint_string_from_file(self, fpath, verbose=True):
        try_derive_graph(load_blueprint_string_from_file(fpath), self, verbose=verbose)
    def load_common_balancer(self, balancer_type):
        if balancer_type not in common_balancers:
            raise Exception(f'Common balancers does not contain a definition for balancer type {balancer_type}.')
        self.load_compact_string(common_balancers[balancer_type])
    def load_factorio_sat_network(self, path: str):
        '''
        Load a BeltGraph from a file that can be read by the Factorio-SAT solver by R-O-C-K-E-T:
        '''
        with open(path) as f:
            s_arr = [list(map(int, line.strip().split())) for line in f]
        for u, (i1, i2, o1, o2) in enumerate(s_arr):
            if i1 == 0:
                self.set_input(u, True)
            if i2 == 0:
                self.set_input(u, True)
            if o1 == 1:
                self.set_output(u, True)
            if o2 == 1:
                self.set_output(u, True)
            if i1 > 1:
                if o1 > 1:
                    self.add_edge(i1, o1)
                if o2 > 1:
                    self.add_edge(i1, o2)
            if i2 > 1:
                if o1 > 1:
                    self.add_edge(i2, o1)
                if o2 > 1:
                    self.add_edge(i2, o2)
    @classmethod
    def from_blueprint_string(cls, blueprint_string, verbose=True):
        graph = cls()
        graph.load_blueprint_string(blueprint_string, verbose=verbose)
        return graph
    @classmethod
    def from_blueprint_string_file(cls, fpath, verbose=True):
        graph = cls()
        graph.load_blueprint_string_from_file(fpath, verbose=verbose)
        return graph
    @classmethod
    def from_edges_string(cls, edges_string):
        graph = cls()
        graph.load_edges_string(edges_string)
        return graph
    @classmethod
    def from_compact_string(cls, compact_string):
        graph = cls()
        graph.load_compact_string(compact_string)
        return graph
    @classmethod
    def from_common_balancers(cls, balancer_type):
        graph = cls()
        graph.load_common_balancer(balancer_type)
        return graph
    def save_as_factorio_sat_network(self, path: str = None):
        '''
        Save the BeltGraph to a file that can be read by the Factorio-SAT solver by R-O-C-K-E-T:

        https://github.com/R-O-C-K-E-T/Factorio-SAT/tree/main
        '''
        if path is None:
            path = '.'
        if not os.path.isdir(path):
            os.makedirs(path)
        fname = f'{self.num_inputs}x{self.num_outputs}'
        r = self.rearrange_vertices_by_depth()
        elist = list(r.edges())
        with open(f'{path}/{fname}', 'w') as f:
            s_arr = [[-1] * 4 for _ in range(r.num_vertices+1)]
            for t, (u, v) in enumerate(elist):
                k = 1
                if not r.is_output(v):
                    k = t+2
                if s_arr[u][3] == -1:
                    s_arr[u][3] = k
                else:
                    s_arr[u][2] = k
                k = 0
                if not r.is_input(u):
                    k = t+2
                if s_arr[v][1] == -1:
                    s_arr[v][1] = k
                else:
                    s_arr[v][0] = k
            print('\n'.join(' '.join(map(str,a)) for u, a in enumerate(s_arr) if (a[3] != -1 or a[2] != -1) and (a[1] != -1 or a[0] != -1)), file=f)
    def vertex_to_str(self, u):
        if self.is_input(u):
            return f'{u}*'
        if self.is_output(u):
            return f'*{u}'
        return str(u)
    def __str__(self) -> str:
        l = 32
        d = self.get_vertex_depths()
        vd = self.get_vertices_by_depth(d)
        s = '=' * l
        s += '\n' + 'BeltGraph ' + self.summary
        for d, us in enumerate(vd):
            s += '\n' + '-' * l
            s += f'\nDepth: {d}\n'
            for k, u in enumerate(sorted(us)):
                if k > 0:
                    s += '\n'
                if self.is_input(u):
                    s += '(input) '
                else:
                    s += f'[{", ".join(map(self.vertex_to_str, self.in_vertices(u)))}] -> '
                s += f'{self.vertex_to_str(u)}'
                if self.is_output(u):
                    s += ' (output)'
                else:
                    s += f' -> [{", ".join(map(self.vertex_to_str, self.out_vertices(u)))}]'
        s += '\n' + '=' * l
        return s
        # return '='*l + f'\nBeltGraph {self.summary}\n' + '-'*l + f'\nInputs: {", ".join(map(self.vertex_to_str,self.inputs))}\nOutputs: {", ".join(map(self.vertex_to_str,self.outputs))}\n' + '-'*l + f'\nOutbound Edges:\n' + '\n'.join(
        # 	f'Depth {d}:\n' + '\n'.join(
        # 		'\t{}{}: {}'.format(
        # 			'(input) ' if self.is_input(u) else '',
        # 			self.vertex_to_str(u),
        # 			', '.join(self.vertex_to_str(v) for _,v in sorted(self.out_edges(u)))
        # 		)
        # 		for u in sorted(us)
        # 		if not self.is_output(u)
        # 	)
        # 	for d, us in enumerate(vd)
        # ) + '\n' + '-'*l + f'\nInbound Edges:\n' + '\n'.join(
        # 	f'Depth {d}:\n' + '\n'.join(
        # 		'\t{}{}: {}'.format(
        # 			'(output) ' if self.is_output(u) else '',
        # 			self.vertex_to_str(u),
        # 			', '.join(self.vertex_to_str(v) for v,_ in sorted(self.in_edges(u)))
        # 		)
        # 		for u in sorted(us)
        # 		if not self.is_input(u)
        # 	)
        # 	for d, us in enumerate(vd)
        # 	if d > 0
        # ) + '\n' + '='*l
    def __hash__(self):
        return hash(tuple(map(tuple,self.edges())))
        # return hash(tuple(sorted(self.laplacian_matrix_eigvals(), key=lambda v: (abs(v), v.real))))



def list_vertex_pairs(iterable):
    for u, v in iterable:
        print(f'{u} --> {v}')

def compact_string_to_edges_string(compact_string):
    return zlib.decompress(base64.b64decode(compact_string.encode('utf-8'))).decode('utf-8')

def edges_string_to_compact_string(edges_string):
    return base64.b64encode(zlib.compress(edges_string.encode('utf-8'))).decode('utf-8')



def derive_graph_from_entity_map(entity_map: dict, graph: BeltGraph, max_underground_length: int) -> None:
    OFFSETS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    def get_offset(d):
        return OFFSETS[d]

    def step_pos(x, y, d):
        _x, _y = get_offset(d)
        return x+_x, y+_y

    # trace map and calculate output positions
    output_positions = []
    trace = {}
    inv_trace = {}
    for (x, y), (name, info) in entity_map.items():
        if name == 'belt':
            d, = info
            forward_pos = step_pos(x, y, d)
            trace[(x, y)] = forward_pos
            inv_trace[trace[(x, y)]] = (x, y)
            if forward_pos not in entity_map:
                output_positions.append((x, y))
        elif name == 'underground':
            d, io_type = info
            _x, _y = get_offset(d)
            # if io_type == 'input':
            for i in range(1, max_underground_length+2):
                nx = x + i * _x
                ny = y + i * _y
                if (nx, ny) in entity_map:
                    nname, ninfo = entity_map[(nx, ny)]
                    if nname == 'underground' and ninfo[0] == d:
                        break
            else:
                continue
            # else:
            # 	nx = x
            # 	ny = y
            trace[(x, y)] = step_pos(nx, ny, d)
            inv_trace[trace[(x, y)]] = (x, y)
        elif name == 'splitter':
            d, _, (ox, oy), f = info
            next_positions = []
            for i, p in enumerate((step_pos(x, y, d), step_pos(ox, oy, d))):
                # Don't consider connections if they are where a splitter outputs to the side of another splitter (where they wouldn't be connected).
                # ... or if there's a filter to deliberately prevent a usual connection
                if f[i]:
                    continue
                if p in entity_map:
                    emp = entity_map[p]
                    if emp[0] == 'splitter' and emp[1][0] != d:
                        continue
                    if emp[0] == 'underground':
                        nd, io_type = emp[1]
                        if io_type == 'output' and nd == d:
                            continue
                        if io_type == 'input' and nd == (d + 2) % 4:
                            continue
                next_positions.append(p)
            trace[(x, y)] = next_positions
            trace[(ox, oy)] = next_positions
            for next_position in next_positions:
                inv_trace[next_position] = (x, y)

    def dfs(x, y, px, py):
        if (x, y) not in trace:
            return
        name, info = entity_map[(x, y)]
        if name == 'splitter':
            _, _, (ox, oy), _ = info
            dfs.connections.append(((px, py), (x, y)))
            if (x, y) in dfs.visited:
                return
            dfs.visited.append((x, y))
            dfs.visited.append((ox, oy))
            for next_position in trace[(x, y)]:
                dfs(*next_position, x, y)
        elif name == 'belt' or name == 'underground':
            if (x, y) in dfs.visited:
                raise Exception('Cannot interpret a graph from a blueprint with sideloaded belts.  Bypass this by rearranging the splitters to accomplish the same effect without sideloading.')
            dfs.visited.append((x, y))
            next_position = trace[(x, y)]
            dfs(*next_position, px, py)
            if next_position not in trace:
                dfs.connections.append(((px, py), (x, y)))

    input_positions = [p for p in trace.keys() if (p not in trace.values() and not any(p in v for v in trace.values())) and entity_map[p][0] == 'belt']
    if len(input_positions) == 0:
        raise Exception(f'No input belts found')

    if len(output_positions) == 0:
        raise Exception(f'No output belts found')

    dfs.connections = []
    dfs.visited = []
    for x, y in input_positions:
        dfs(x, y, x, y)

    num_splitters = sum(1 for name, _ in entity_map.values() if name == 'splitter') // 2
    # print(f'num splitters: {num_splitters}')

    graph.clear()
    for i in range(len(input_positions)):
        graph.set_input(num_splitters + i, True)
    for i in range(len(output_positions)):
        graph.set_output(num_splitters + graph.num_inputs + i, True)

    for p1, p2 in dfs.connections:
        name1, info1 = entity_map[p1]
        name2, info2 = entity_map[p2]
        if name1 == 'splitter': # from node
            _, id1, _, _ = info1
            if name2 == 'splitter': # to node
                _, id2, _, _ = info2
            elif name2 == 'belt': # to output
                id2 = graph.get_next_unused_output()
        elif name1 == 'belt': # from input
            id1 = graph.get_next_unused_input()
            if name2 == 'splitter': # to node
                _, id2, _, _ = info2
            elif name2 == 'belt': # to output (why???)
                id2 = graph.get_next_unused_output()
        graph.add_edge(id1, id2)

def derive_graph(bp_data: dict, graph: BeltGraph, max_underground_length: int = 8) -> None:
    entity_map = get_entity_map(bp_data)
    derive_graph_from_entity_map(entity_map, graph, max_underground_length=max_underground_length)




def try_derive_graph(bp_data: dict, graph: BeltGraph, verbose: bool = True):
    if not is_blueprint_data(bp_data):
        if verbose:
            print(f'The provided blueprint data does not represent a blueprint.')
        return False
    try:
        derive_graph(bp_data, graph)
        return True
    except Exception as e:
        if verbose:
            label = bp_data['blueprint']['label'] if 'label' in bp_data['blueprint'] else 'Blueprint'
            print(f'Could not derive a graph from the provided blueprint (label: "{label}")\nError that occurred: {e}')
        return False








common_balancers = {"1-1": "eJwzsDa0NrAyBAAFzwFz", "1-2": "eJwzsDbSMbY2sDJUMLQCsgAXxwLv", "1-3": "eJwzsDbRMdUxszawMlQwtDLSMVYwsgKKKBhbGeqYAQBUsAWW", "1-4": "eJwzsDbRMdUx0zG3NrAyVDC0MtYxUjC2AoopGFkBRQFgnwX/", "1-5": "eJwdyjkRADAMAzAqBuAhTn+XP6/muuoUd/NQQYnKGxbkZEN6sKN5cWK4Erp/w6ozy5UPa0ALrQ==", "2-1": "eJwz0DG0NrY2sDJSMARiIytjAB31A08=", "2-2": "eJwz0DG0NtYxsTawMlIwBGIjKyAPACzjBA8=", "2-3": "eJwz0DG0NtMx17GwNrAyUjAEYiMrEx1jBRMroKiCsZWFjqmCKUgEAKbcB+g=", "2-4": "eJwz0DG0NtUx0zHXsbA2sDJSMARiIysTHWMFEyuguIKxFVAGAIg3Byc=", "2-5": "eJwdyrkRwDAMA7BVOAALUY/t0PvvlUtQI6j7UEGJSqpuuCAnys1Bu3gw3lw4/ia2/4tlFRPpYb/o4Q2N", "2-6": "eJwdy7kNADEMA8FWWAADUfJL99/X+RxsssAEdTYVlKikimonnNAt3VhoXhwod06sezaGH0D3I5j+2QcgDw5O", "2-7": "eJwdy7kBwCAMA8BVNIAKZGMesf9eIRRXXqOOGiUqqKQ6VdQ4zQFd4c5E9+RAurgw/QKGX0H5JSzvG7H/8AHYlxCa", "3-1": "eJwz0DHUMbI2tTawMlYwtDJRMAJiYytTBRMrYwBJEgU2", "3-2": "eJwz0DHUMbI21TGzNrAyUTC0MlYwAtImQNrYCigKAF++Bfo=", "3-3": "eJwNybEBACAIA7BXekAHQUCp//8lQ6YsGv0dXvZb2rDhCmwVE6UZpIKNULI+2nsJGQ==", "3-4": "eJwVybEBACAIA7BXekAHEEEt//+lDplidI5e3Dx0a1PAn6GJUDFReonUb0wl6wL/nQnT", "3-5": "eJwdjLERACEMw1bxAC5iIAn499/rOSo1koLi+DSpRSVVVH/hCXlhXE4nC+lDBcqbjePrNhTWoITlYmL7XdB+H2jcZkOygv0DFYoT5g==", "3-6": "eJwdy7cRwDAQA7BVOAAL8YMCtf9e9qlAiUExrgYlKqikiuo7nNAvXEg3J9qbC9PFg+03sPwOypON43c/l+0P2Q==", "3-7": "eJwdy7EBwCAMA7BXfICHGEgA8/9fpQwaFRTbkahGdWpQSRU1T7hDV/NAd7KQXpwoK7ix/Bam34PCf8b26xgu5gdmZBIp", "4-1": "eJwz0DHUMdIxtja3NrAyUTC0MlUwAtLGQNrEykzBzMpcwdTKDACHfAco", "4-2": "eJwz0DHUMdIxtjbXsbA2sDJRMLQyVTAC0sZA2sTKTMHMCiijYGplBgCmJAfw", "4-3": "eJwdibcNADAMw17RARqsVEf5/6+UgQBBBsXCuhcVlHa4Q24ol3q9e3Jg+n0MS0zkb815AGpOC7s=", "4-4": "eJwdyrkRwDAMA7BVOAAL0Y8U0/vvZV0KdAiKg/N+PFRQuuEFteGN2ZaThXQPlP+D7WI+U3gLVQ==", "4-6": "eJwdjLsBRCEAg1bJACnEv3n773WeBR1QjKvbRzfDTLPMNucr6eJSM9QuPctTKxSDZo63KHmhIFTTRP0LR7TANU7eUzvvqpHp9QMppxZt", "4-7": "eJwdjckNADEMAluhAB7GuUn/fa03DySEGAiKyXbVqE4NalKL2tS54Q6V0gOt1D25MK3gwXJ1BIUfi+NHY/vnE5LfCpTOOglkJaLKhA83hhfnB0UuGKk=", "4-8": "eJwdzLsRADEMAtFWKIBA+CPZXP99nUfBZvCC4uD8NKhJLWpTSRV1qPuFF/Qa3piv5WQhreBF+W0EhfuO6wZw3AQkt4LtYv5RvxSF", "5-1": "eJwNw8ERACAIA7BVOkAfVEWh7r+X5i5BcXByXemGN/QPH8x/ObHdaCugcLKQLpQlHPcDa0QLxA==", "5-2": "eJwz0DHUMdIx1jGxttQxNLA2sDJVMLQyUzACYmMrcwUTIDa1slCwsAJJK5gBmeZWpgATSQo/", "5-3": "eJwdiskNAEEIw1qhgDwIzAGZ/vtatC9blh1EILEeCQaYz3WN2hbD1LE1fodHdKOrUdb6dysxMXmr0B8vMg6P", "5-5": "eJwtjbsNRDEMw1bRACpM5++3/14XBFewEkiFcbq5f0yzzDbHGV/UFJespXbpNTTreOsU3TTtIs0Qd7qGaPVsjSIMIophUvBXGK9A1nvTKjDxA60uGW8=", "5-7": "eJw1jtsJAzAUQldxAD+ied3Y/fdqGuiHiCBHG0Wzc3xsutODnvSiN12flg1dORP9amRh57BwIlMDFXVqQo4OVdCIGy3o5DGhysPCLRdcsH4VHazo+q22P0XRoAz1aFMLmm9OO6qXV947zEhU+wIDjiTv", "6-1": "eJwNx7kRADAQwsBWKIDg8G/cf18m2BmpKDZ2Ds4nvfKConmjx/DBjOWLawnbKqiyJ/UBd0wL8A==", "6-2": "eJwdyrkNADEMA8FWWAADUX5F99/X2RdMsMAGxWRj5zgSlSc8oSu90K7ujXFNF8oKKPyfWK+26wPBQw0O", "6-3": "eJwtyrkNwDAMQ9FVOAALUbZ8MPvvFSFI8YoPMigmByfr0aAmVU94QS29Mdr0QbXli2slJSj9/SFZxYttBRT/fLpeM58RlQ==", "6-4": "eJwdi8kNADEMhFpxAfMwuTPbf19r5QVCIoWauobmRxNdDDG/9A28onlHLx8+Mctv8VRfBpEBfluQfmdsk4If/4QRBA==", "6-5": "eJw1jMkNBEEMAlNxADy63DeTf17rGWk/gBBUE0p1Dc2HI66yKavLp3kF3pHl3SdG5Vm+DLoBposV10wxgu6XsINalG5fQTD9UYPhDxzbpKgi/89mluhxTBP5A1yaHHc=", "6-6": "eJwtjMkNADEQwlqZAnjEk5vtv6+NorxACLsIpaqa+scUS2xlUZ45v+IReEaerF7RTu8nhynaQTFdtNimivPtvpKg+XqC6qsKHjNNCoJ8LJghaiyDyB8GIxvh", "6-7": "eJw1zMsNAzEQAtBWKIDDgL/L9t9XLCc5gdDMK4pmY+d4tamHLvpsphvd38qEsuCTLRv99HFy5qGEJ5rUgBR1qkEz14FGLgX1XA1q8blZ0Po+r8hUQf4r9VN2VJQ/Y2IeKQ==", "6-8": "eJw1jsENBCEMA1txAX7EBhbw9t/XIbT3cpSMximKZmPneN3oTg/6oSe96M1Wb+WBMuGTLQv9zOPkExU3VNGgOnbUqMOOuKhz6bFpwZUrh3auH3ZuBazLFNSiRU3o713RvouZ+whmZEqQo04NSF/lQUX5B/ZuJ2s=", "7-1": "eJwdyLkRwDAMA8FWUAACgrQ+qP++TDu62QuKyeLDwXlVN7whH2S3rMDTGF6YH7aVUDaXJUhW4fwzerxO3A7t", "7-2": "eJwdiskRADEMwlqhAB7GuZ3++1qyL40EQTHZ2Dk4rzo1btSG6iDNVgp0y6iF+WSXEsr6v5DcV6lBvorejgmF2wdFThHF", "7-3": "eJw1i7kRACEQw1pxAQ4Q38LSf18HwWXSyC7G1c3dw/MwzDRxSi6RoZpb7XJPisaV+WAlCJJumrjx/URLwqF4eSdV1H9Trn5oMhSm", "7-5": "eJw1jckNxEAMw1pxAXoMPbfSf1/rBNinBIlsQqmuoan1ZFNWTmVXjqd5B76RpkWvMHxiVrHeYpsZTNNFBt1csYJaH1HPa4YgWP7AwTBL3ACzxQm2y/Zyjj9n3D+vlesYxPgBOFYfig==", "7-7": "eJw1zMkNwEAIQ9FWKMCH+cwap/++QiLlBBj5NaFU19DUurmUTVlRKrtyKOfdvAOfSF/Raw7TYtax3mUbRA8wS8ygmyOqU//XC6aLrOD4c4Ptj45juiAuM0QG4+/mrzWTYjyZHyAY", "8-1": "eJwdyMkRADEMAsFUCEAPocMHm39ea/tBVTNutLC0srZh82N/rgWehTZSdNRBi8S4b14tMcAQCyyxsV9wMcF8mcc/VwQR9Q==", "8-2": "eJwdi8kNADEMAluhAB7Gzsn239cmeSANgwiKyWJj5+D81KnxhRd0kt4oK9AOdEsYt81Ly0oorQY1vx/2c2EVVG/R4R/G+xMb", "8-3": "eJw1i8kNwDAQAluhAB4e397031fWkfIAgQaKcXVz9/D0elhmm/OU2CJV46gFRT3DCNC8bd20gyaSTjNEgvsWOTumi/6jGllPUEX5nLy+0HsXwg==", "8-4": "eJw1jMsNACEUAluhAA6Of5/997VqsieYTCAZZxdXN3ePTTfDTLN2iiliKZ8sQVI90ALUL43bZpBFDpqp4th7IGq8D62giPL79JAz+gCI3Bct", "8-5": "eJw9jskNA0EQAlMhAB4Dc+P883LvSva3BBSNotk5OLm4Pzbd6UFPen1a1KAcOBJ6ZIwHzVysB+0HqWIb2gVvNKkOzVjUgXrcqAsrWtSATl4LtH6Z2rx0mW5KXbaW148TdWpC/3Id2V9IsSKW", "8-6": "eJw1jcsNRDEMAluhAA6G/P3672udSHtkBExQNBs7ByfX54qmG93pQc8v8kC54ZTQUkYvMlKBedG66FRhpwbVoJEOqlbV3tSCIzWpDp18Aqimhw6opxa1oZXPCu18Ykj/Q1+Z6qRT8wfmpCIF", "8-7": "eJwljcERRSEMAluhAA6CUWNe/3199V8yYWZZGkWzMzg4uT6faLrTQQ960utrJUGVcMnopYa4Z9w4a2M9QOfTYTs1oYNtKqFZbtSCdj07lPUG4FZvA1rlxUSWghpQvM7RnYm85V0aVED++3+wQiMv", "8-8": "eJw1jcsNwDAMQlfxABwCsfOh++/VNFIPlgDBcwMhdCQKA/NRg04iqEMJFTSg+TSv4Dl5Rzdb5BFlMsbn5qeWKTCDMic4gmlu8CynLzg4fNnB7YsPLt8Psc0EFWxmgadS/7r/PJodrBdMOCKf", "1-6": "eJwdyrcBADAMArBXOIDBOJ38/1eKZsUenFxUUKJyhwW5MFHc2ZAWK7pvQ/OLqP71AFRFC0g=", "1-7": "eJwdy7EBwCAMA7BXfICHOAnQmv//KmWXYj98qaBEJVVU77AgFxPlxYn0YGP5UExfjOE/oH3LB+mlDYw=", "1-8": "eJwdy7kRwDAMA7BVOAALU4+d0PvvlZx6YN2HL7UoUUElVVTfZUFOBtKHG+Fm4fjX2B6P9gyU53wh0w5T", "2-8": "eJwdy7EBwCAMA7BXfIAHnJBAzf9/tWXQqEGdhxqUqKCSmlRRfYYD+oQnE9ObC+lmYfsvWL4J7dtQvvEFmU4P2Q==", "3-8": "eJwdy7ERACEMA7BVPICLGEgAs/9e/5dKlYLieBI1qEktKqmiNnVeeEJeGL/TxURZwYv04YbCPXHdF8e9sd0fy8n6ALEkEvM=", "5-4": "eJwdyrkRADAIA8FWKEABAj8g99+XPY42uHMQgcQ4dJBggHlc06ht8czn0LKpRlnrf1b6p+1Xlgp9AQMwDgA=", "5-6": "eJw1jckNwEAMAluhAB7Ge5P++4qzUl6AYERQTDb2R5Na1KYOM5h6wgPyRJY2L/TywxI3JKtzYVuNSqj78lg+VOB8tQqPf1Bg8EDN9wJK3xdMV9YL9j8Z+w==", "5-8": "eJwtjrkNA0AQAluhAIKD+3H/fXl9coA2WAZNo2h2jo87PehJL3rTh77s7dOyoIqz0SsjEyuXBzcaVMeJTE2oXtSGetyogxnVFbTztqAWVcWQ/qzmm5KjRV1o5YlAN88FbnE5FnPyvOCC72/fRW1qYUei2hfPnydD", "7-4": "eJwljMkRACEQAlMhAB4y44n557Xq/qhuoFAMJisb+1alGtWpsYsn5IXwQFoF9YB2Qz90WgnlccMKSlD4zSH5PUDlltbvPyoIFBM=", "7-6": "eJw1zMkNxDAMQ9FWVAAP/vIapv++RmMgN4qEXhNKdQ1NrZdH2ZRVpbIrx9t8Au9IP9ErD9Ni1rH+4ZguCLrZYgWYKepn+2LB8vWC6UsGp7yaET0ek6LK/ITxCc0MkT/bTh7y", "7-8": "eJw1zMkNxDAMQ9FWWAAPpmR5YfrvaxwPchP4hdcoBpOdxfFoMxrjTMFIRmcUYzzNG/JCWA3pif4edZZxyraSEpTWoiYka1AFLV8Smr4qNHxhqHxtnBdRiWkFdUJ8Sv+U9k8/HFQg3Q==", "16-8": "eJw1j8mBAyAMA1txAXogyVxO/30tkOxvkK+hgRCMRMfAxMIGG0hQoMEE+ycbkkghT2ciO3Ig56cVR7A4Q5d8KYsrenHHuDRLLdZ97ktsJQZZUlCPXXIwX9Ifj/KxyjDLPZRlh1v5JIc3vMK7nlR41fMK919J603t26+zSmGVJ3wEZz398Kj3g7B/Jc17Rf3280hvqAdXacBH+8hP6CxvL2GI/8lZ3q6t/J36Ay/UQTU=", "1-9": "eJwdzLkNADAMw8BVNIAKy/mV/fdK4JrgxVVSjerUoCa1qE0dZtywICcb0oMdzb9jWOJB9+aC5AJwXAS2C8FyMZj/V0Dx34wHphcVOA==", "2-9": "eJwdzLkBwDAQAsFWKIBAcHpx/33ZVrDhTqMeDWpSi9rUoRst2nQ9LYa+nM5Cz+ZCZXJgp9ixok4VZmRKGFHjgZyLQsp1oZZL4+Ti0IdB9f+uF8CvGZM=", "3-9": "eJwdzbkRADEQAsFUCABD7KOHyz+vU8nAGaMZFOPTpBa1qcMYjNuCkYz6hhO6CxfSzYn24sZ08WBZQQnbGlRB4WdB8uOg4SdC5X1ZlCcbx0qqofT7gdqH6wewFRtV", "4-5": "eJwty7cRwEAMxMBWWMAFxHue+u9LZhQgw6ZQU78YYooltjhXegRPzTP60/DWim1SFcugE6Q/FeXPBZguWhxzBEE3JTJorlfUP0wv7Ru0sheO", "4-12": "eJwdjsmBRTEMwlqhAA4BZ3v8/vsaT65eJA2KZv186Y81WGL1oFiTtVibdViX9f1GJpQN56CyMPPx4os2PXAzqQvtuGjDIz50n1ceHXaeAD55DnjnaaAbL3rCK08JzzwrdlSUocaIOpCzqcbpCXpz4kkX9MXdsnCiRXXtynkvM/poYUXNGJD+q9vbkaOz/wAXAjJ7", "5-9": "eJwtjsmNBUEMQlMhAA6Fqa35+ec17tYckCyQeQyKRXP+qlhmTdZibdZhXdZDj9/IgrJRLedgtlY0eKERLWriRkUZWnlef6btjec/1s4HgCq61IEcPawB3XxU6OQDQ08+NmrEgyWUYtEF92F6whUvugd1z3lRnm+hLrzSL97xoRfctQM7EtU0RaYKp3dKf6BKMQo=", "6-9": "eJwdjrsBREEIAluxAALB/XL993W+DYhgHBOEUBiYPy1oQwe6qEQR1VWhxi+9gh15R3WGT8zOMglmkGaBI5jmBHtd5gVPcFi92cHrZwgeP0mI1oQquP08oWkJYqjx81GS+xd11XcSbCr9XgtO3ydfD8q45gArtimQQZkLnHG+of7rMS0h", "7-9": "eJwlj0uSRTEIQrfiAhgEND/e/vfV3vRAyzJIOAOEkChMrF8OJJG9SGQhJ3IhN/L8hnfQJ9SVvlFd0xyxvrbNBBlMc4MnSHOBM7gtQRU83tAMye+bUFkH2qFjzU+if20/TetCK3SdPWRoWQO8kddF1Ails89PFJ+mIw1rv6vefAyh4YcRbJ+PJNg+TTyCHf45H5NgxjULVLDM85LrsazgMAXWH9tcOY8=", "8-10": "eJw1T8uVBDAIaoUCOAQ0P6f/vtbJvD0pCoKDohlMTi7ujy9jMMToaTCSMRmLsRnnM+pAdeGuURrIBrMkrC/a3+6UkgooS5u6UDMX1cpdNi3oloPesOsZwqrnCUc9W/hHPvCpuMyBaNWiJ3KU91utctIBz3L/0XeyXl541IsMNWd+VzqPvHBLQSU0SpMy1B8cakH+j6ySqfkHwb821w==", "9-1": "eJwdjMkRACAMAluhAB7ikWjsvy8Tfyww2yh2Dk4uGp37ym8LNSgOekgYRTPUsQotNOCFu5LyvKCcJpQvx/lRv06BQZYqpcYejSoVCA==", "9-2": "eJwdjMkRADEIw1pxAX5gQi62/74W8tEwaCyj6BwMTi5unk+Xbp/lhVIGb4yUEI2ZcqzGTg2cxk0FVPpA9eg9VDpYDUtNaD6pd1dxQdXYUIXXD3Q0GRM=", "9-3": "eJwdjLkBADEIw1bxAC5i8kC4/fc6SKMCGQ2KxsnFzUNnfLq0QdM3UgNqWEqYjZUy7MZJTXgj8kI13lCZA5VyKqDSl32JfEmocguqSlAO2fuaFbglfln9GsU=", "9-5": "eJwdjssVRDEIQluhABaB/Jn++xrzVgoeuDaKZufg5OLm+XnTh77sjV2/FgmKDEcNPeoYT85oYD25c3GekuIGt3jCM16shFeZfCVVNakD1cl0OSfu9ICdDwtXwyPDPTUFF6tqNm60qQXteHzh9YUN1WMXuo+q+u9QtVTu/gFLbCpE", "9-6": "eJwtjssNRDEMAluhAA6BfJz49d/XOtHerBGDaRTNzsHJxeD+POhJLzroTZ+vpQTlgVNGr2OkOubFK9UQl+/LpNSG9qUqL6gJRbrTRWbatOCeWtSAW74f0HoZQyOtG7bzTYGVbw3OX1LtmFRJRQKqsv0DBV8mRQ==", "9-7": "eJw1jssNA0EMQluhAA6G+Xv77yveSXKwhC0/ICiajZ2Dk4v7cacHPelFb/qwxRN5oBqnjJYSeqphvOt81XpvOxU4qUkNSKkN7XSD4uoCO7Wgng6qLFfatODImwudvNGwC6xoWHlrwF+8fsaP0vz7VK9FzQ/zmCds", "9-8": "eJw1jskNRDEMQluhAA6GLE78++9rkkhzQRYCnoOi2dg5OJlcn003utODnnTSi95flARdcamh1UYvBcaVWTLy+uteOrmkBhQntkuT6tAsi1pQLwe1YdXDQaseEY56UGjX40J5Ww7okPart//g+WRQ+QORtSW2", "9-9": "eJw1j8uNRTEMQluhABYB58vrv6/JjTQbJNsgjhtFs9g5OLm4f3XHYnXWYE3WYm3WYW+/FjXoE+egIqF/MiJjfrKiwv5E1zypBc140oJWPGjDMyVqwIoPvVHKrXWHRtx54P4sd9Hjom9jpRp9Ty3e9EK1PFD45LHCOw8XXnnEOFF/TeOlC3bcvrTuH+fxnX8+R5s60I4vzYDqbeYfNyw4Kw==", "10-8": "eJwtjssNAzAMQldhAA4B5+vuv1edtBcLCQyvUTSDnYOTi5vn404PetKL3vRhNIY+LRVQqsFXRaqjp4xxz0wJKzWwr3GuUmUmtSFfV+9Z1bOoA6100IZOujCqN/KNw863DysfAtzyUUDzH9ZOmy6kgjjUgsZv7gsDhCi0", "10-10": "eJw1j8uNRTEMQltxASwCOD+//vsa50qzQU5MwmGAEIzExMLGwf054QkveMMHvsiBJFJI/0aRweIIPfE7ZlExn6yiYz85xYz7hO3e4Anu0oF28JQ616FTHjBDuyxohUd9+WHWhxBWfRShVe5XM9RRC5zBVVrvhrOUkEJZNtyD3s+6YdeHH7r1NQixpLeSn1lN2KUmuILd6X6o9x/VpQG2eZTa3N3z1dH4A0UJPFk=", "12-4": "eJwdjskBBEEIAlMhAB4N9unmn9fq/ESxYFA0g5OLm4eXjxqUfr70YwyGfiMVUMpwaiJaztTCarlTG6flTR28lhrtlnqhSAe94BoKsdITdnrDO78ceKYPfPILhOpaxQZchLoW7UKvPbr9qPJXzQc1ja56VaNQ9XsqpeL+/yQp7w==", "12-12": "eJwtkMsVRCEMQltJASwE4i/Tf1+jnreDBOI9NhCCkegYmFjYYAP5y4YkUsizTWRHDuRELuRGb+j8taKCRYeucjEjr+3FHuPaWRyxrt3FGWzXk0+rJHCFVCbcgqs0oRVmqd2VWznhkzmiwQxnPbTwqkcXmuUN99AqT3iEdz3kcK9HHZ71wMOjHnvQJUMKHTGgGRpfnXlfl4O9lBBDZ7KhEWLZ8Gntcr9g9iM8l88f7BvmLi1oB0epQxnqX4vzZfof2xlIHw==", "24-24": "eJwtk8uxZDEIQ1MhAC3MH/Pyz2vUnruhbnVZQhzoA4XBEUgUGoMLPVCFGtShAU1oQRs60As7MGoM5n/XcB03cBO3cBuXBj8HWhx6HJocuhzaHPocGh06HVodvnut+O7X7Nft107z76yF6FqK/YqvlcSv5FpLrY302pX5lbt+RM+ruq6itm6ivu6i8Wqu07Be7fUUnfUSpbbFzqvUjpi9Su0Vi41BHInZLDBSnK2DHMnauqiRujsHw1CzczElc/Zhkel9ZGTuPjgytY+P1NketErO9kG3NOWBMWndcYzKxH4YZWw/kjK+H0wZ3Y+nNHsyRUqzp2JG7tkPskzux1kuJf9Ry8x+tMVy4yA4dm1cBOe8mxdJ6KTeKIK/X2LrjUBwhvqiG+lQnpJns5FXgmuap+oPk92NRB5xvlEEJ/+tCtkS+jF13eAvKm6bhjBJ21Jwl8EP3qdL6bajickfL5W2x8ukcpsXnFK+HegrXQ9GSefDw6OJt6QjzbX1D5j7hiFNPDb5N2Bm3wqUSvBc2JSrjc/ZCaqQjNFfC69N0mjJ2DIU0fG86qnsi+okRiwhTlCJcEmeI7FwLv/mcvJxZP4DhiuxUQ==", "24-12": "eJwtksmR5TAMQ1NhADiI+zL55zVo17+wbBUBEU98UBgcgUShMVjogyrUoA4NaEIL2tCBLuzBqDGY/1vDOjawiS1sY2nw50CLR49Hk+f/3lmInqXYX/GzkvgredZSZyN9tjJ/Zc+f6PuqnquonZuon7tofDXPaVhf7fMUnfMSpbbF3lepHTH7KrUrFheDeBJzWeBI8a4eciTralEjtTcPw6HmZjEl825Dpm/5vcdoMnWbUu960Co51w/d0tQExqT1xjEqExSYjN26jPOHZ3rLFtryopSms2JGlicjk2xyWTaRx9wySV48BGPUxSI4914ukhBJsVEEub9hrC8CsVL1m8qYlvKUfJeNXAlin0/Vv9i2F4l84uxRBEP9oUe2hP4YuV7wRMXt0hAmaVcKvk3wg/vjUnrtaMbzD4VK24fCpPKaG0Zufh3ola4veUnnx4JLEB/0J81n6D867heGNPG45JpyZr8KlErw+Xkpnyp+zk5QheQY/bvC65I0WjKuDEV0XJf6VPYb1UmMWEKcoBLhklwvYmEu/+Vy8nFk/geyt58h", "32-32": "eJw1U8mRBDEIS4UAeJj72PzzWrm756OiXELGEj4srGzsHJxc3Dwsh0VYlMVYnCVYkqVYmmVYDyt6lNVYnTVYk7VYm3XYDpv8yW2+3bf99l+Bq3AloCEQEb13gAcdgZBASSAl0BKICdQEcmLg2R0GPAPPwDPwDDwDz8Bz8Bw8B8/t76wpyQVdM7ILvuYUF3ItqC70WtJcELQUiTyIpiaxB9E2JPFgrh+SerDXhWQu6lnHTfKgrhuprTupP3U8NXqDtB5Eb5LORUNvkcmDupHsTZGbiGPIe8s5mzK2m7soZ9u4D3WvHLhxlLpQwpCDgY7vZz5q3S8A1LlfCKhlvyCoDa3w8xj1wSksPHjcif0iQg3KGxPsQecbFc7Bf+Oi8h0MlZS9rTxKc3WHp6lzB0Gde+/sFywNpn+zpcH0b7xg3PqJmFp3EKwcGjzjYC+RFATe5DEKrn/TRxvU3g2gwTvfJSBDCM0Bj32j2IeiNoczyGezuZwM4QwHvM8NkA8FLD+cWJWzhR+BTTmf5fl7ZuWXRsWOcitZbRyOJusNCAoFrnAuCMpWcGGM2RAOpHo2QVZKFMWlFLqFQihrR3gOlW6jcBpYfuw14jym3oRavrDwYn+MOk1V2849VLIdPDDRn82Yop4naJjYyPbok8hgfFh4nFw2lPOQ6yaGNEqMjZOisC3hKnLbME4h903lwEfC2DhJCt8aLoxg39gJi4bbqeb3ot4JbpgfG1hqJceaG0dQ2iasNorYQi6wCH7iB9g/AH3lnw==", "64-64": "eJxNlssBJCEIRFMxAA4C/jf/vPaV3Tu9F8axQbEoSqu5haU16zZs2rJtXs3dPMzTvJl382E+zZf5tqgWxIRFWjSLbjEspsWy2JbV0i1ZMi2bZbccltNyWW5r1ZpbC2vs2Kx1a8PatLasbevVulsP62mdhLr1YX1aX9a3jWrDbYSN/BN8CL4En4Jvwcfga/A5BnkNEhvKDL+B38Bv4Dfxm/hN/CZ+E7+J39QR8Jv4TfwWfgu/hd/Cb+G38Fv4LZ0Vv4Xfxm/jt/Hb+G38Nn4bv43fFihCBVgquFSAqSBTgaaCTQWcCjoVeCp+LvjwA/wE/QT+BP+kAEkFkhKk7z/1jFZcJs7oJWXaGaN0mXHGLFNmnbHKlnFCdnG/Ns6sxfPadqYX79eOM6P4vHadmcW3bNQz2cmvJbaXyGuJHSX6tcTOEvNaYleJLZvE7pJ+bZxVS+a17Swv2a8dZ0XJee06K0tu2VbPaqX5tcRyxryW2FFav5bYWdq8lthV2pbtxO7S/do4u5ae17azvfR+7Tg7Sp/XrrOz9C076tmtDL+W2F5GXtuOU5g9wDGOw3bPWfY4Dn89mebMDs1cgHMeh44+AHnsE+onFuTbCcod7kAaJ2B6JP67naAbIiuQMn4oDtT4PzQH8HFeqjNfz0t3CoQP6USjKNQ4Uq3BPPm8rUBsP287EDvP2xLsledtC3LjWNDXN/mAQkDNcPLfxNLW0ch/sxetHF00YK+nlVifnJ92Yp59n5ZifWKftiJP9qXdI6AHjIzUGYNY5p+WI2fWf9qOdTjv03qsA1ZP+0HadXxJlsAcHx8SJ3JbnfyJDQg8iKW9okL4LZxZP0Vgxk0+oi45PK3MXuT/tDP54PO0NP6s87Q1+TOW4DV8KmdMSSCx9MDb8sSCydP2rJnnbX3m93nbn9zgBnLhotKCVZKKEM7UQlLaNGYv5DASfOirVzJYp55XNsjfzysd7BXnlQ9iwTOVJzjTvYHcRnB2+u2VFtZZ55UXxuT5SAz+7bwyUwZtgvJ7FMB09Ijz8XPgNtfDLmwD/NwRq3AiDsHHAr+ddUUhxMURfqfcsMapgIMWU19nxH+VpBq/7gEtkGE8C80DnixWJg6o4ZaQUelQ17E0rtwehJdJzzLiSBMm69aq7IGuOex1oVpZAlQ9lYcfh7EuoWvam4pcAWy3W+kqxvl2gbojv+7Q+mKFuoMq+NINKVElGS4YR2qY+9g41U37qTJZipkuNHf9MYGtSI/sGUppgBPe8MvxmE7GlZ1SR/UyAYAbhc9lqjhgUdkH3fam8+ms5MhV6nQYc9/5UI5fvnT/D4OuM7ETSjDZFLKyXJmqpx4FeHAfEA2SyoVI7mpWKJNpbjU+l8XavBC80o0wz1NPCCIrJ0LpvGl/7Um+U53MeKjOrMhOcR8YGsNgF75Siv2qJOwEgZ9KQq/Qq+Oq3nrVRN3FfFX9OmP8IW6I2eDxqKTG/VMxGP1P3fBjvK4qkS84KU/WQREcRXOpz1Keqh8cogyh55C6yMdPZfj2qQw3Z+ilJNXw/namYqVQ+AvZxb50eITUvL4qo7Fyi6eTwfAqTpdKfkqxKC2hwF2WriWqr5OzK+jCBKEPWZpq1coSyyBxeJHy8I7zqpuLSF5kjtrwn1pxvq5zt68+KMwPD+7OXw1RePJnPIoEWNSG8mI/T0RHp/iFCeyk+4M7gAhWx1ssk5RkAVMHI1fPhWQDj0ZeVbqBCzVmjn1YcYnPUn18uDGYe3jjugHirbdusP7xYKnn/PKA9VgHRnMjsS5j+Lq1zv6vrhr3Ww++PzUWp7f/arbEeqlbK+piKua6a0JYs0qya5U06aw8E1QlNo1etqokXZX+6fXA+cQZlzapu6V/8ztfti/fvj8MeLOQD+NauDhcz3botlVVSTGRIYFW10m3SYB3BCsUrnO/L/z5F45cVVk=", "11-11": "eJw1kLlxADEMA1thAQhEgPrg/vsydWMnCPjNLgcShFCYWNg4uMjxo4saqEQR1d1CTdRCbdRB3Z/hZOQLOhV6Uc6K+WI5Z+wXx7nivsheyUiaA+y1YQq8wa5saARlTahb10poh6YbhH2e/nBC6Y8otM0mquD8G+a2BK3QsAo6IfmjDi2zDXqm/mfKn0fomAcNxmMdqCLbqis38muxK9dc4AyuP8KUSXBEtvMG22s/C65gWuN55TQTuSP7G/3U8wuQ2UXN", "13-13": "eJwtkMkRAzAIA1uhAD2MBD5I/30FO/loPObQigEHIQQSEwsbBz7gDucnApGIiViIjTjIgXQkkUL2TCLnZ5SH+RWWp+lKlE/LK7N82bqyy7edK94jx7xnZM73juIAp3GU+nGMsxTgNvVPm7vxlBpkmHY9NAuvR2da9QAtRj1GU1R0LBp3aUJdUrEXugVvKdurTQ/U3F7cUG+e9cKZTr185lmc4DCfxb5VJ11FgsvIUt9OxlVyqKPzel1CPdTG8Avf7upjPB5vDIFp7B5BnSvfnvba/81+fj1fnthPsA==", "14-14": "eJw1kMuRAzAIQ1uhAB2MEP6Q/vta7GQvDGYQevKAgwgIiYmFjQMfcIcTHh8lNKEFbeggB9KRRAayNYmcyIXcn1E+zcuX8XZRLtN9Zvm2ecsqP7ZvOcVhPl71opu3KM3j9bNIcBtZ4Qgad8VEDAsvCXEsWHJomFSP0eLUwzT1ziU1jXqwFu0uKCxGReeguYoBTlPUwzdGBa8F5zPtg7teShPrBTVfPzDfP7l3EIHHqIoGS+OpWIj20vPqJisOJIvOn5c5okSok2Yxwf6y3smr4np3ejL+L/t35w9nDFIY", "15-15": "eJw1kMsRAyAIRFuhgD0Iu/gh/fcVdJKLI4zCezvgCBBCYmJh48AH3OEBJ1wfLWhDBzmQjgwkkf0hkRO5kBt5MAemf0b5NL9HlC9j+TbdW5Yfm7dcFcP2LU+Fm49bu1eEebxO/0pzvc6sIGJZsOggLVYxwWn0kqBhZIngMakerGnU4zWxHrLx1KM2ZikhN87ShMKU9VRMXs/G1Jwdh5uinpl5VhxEWpziAmWRxQ5uG9fd3nOoP0/cyc3DXXKoXdp5IWi+iwPR8KPYb3rgLDZ8W3ca3RkWHYgQ20JFge2+f7uiU9qI3t5BJeJ8AVrnVjQ=", "16-16": "eJw1kduNRSEMA1tJAf4gscPDt/++loO0PyMkEjIOA4kCITQmFjYOciATWUgiheyfNnTQA53oQhN9Gxo90Qu90QdzYCZmYfI3nDPyQzlX8IOcO/rDdJ5YH7ZrxPmQw5WR+ViuiuSjXIzsx+kSakfJJFhR2xTYQVoJ3UNZgkYo/axD7Scekp97aPjpB2UVNINt3R1cyfLLFJp+sUL0SxZafuEil2ujFHn1G3Wi2mxQUce827w1N9RBddTNNcERNc0F3lnD3OAJLquhDM43XcH973Os8flUmgN193h3ctszapkH3MF7dV++NTTvJ64/anVYjQ==", "20-20": "eJwtksuNRTEIQ1uhAC/CnzD99zW+0dsgFIKJDzlQGByBRKExuNADVahBHRrQhBa0oQO9fxOYxBSGtwdzcQ+u4hqu4wZu4hZu41LsU6Pcod6h4PG/sxaia0fsy3wtJb6QayX1hV5rmTWV+2XKlhHVF9lkouy6ovFirh/RepGdLjrrKnq/aLFecBOvzUK2uG0aYiTrV8reNvSRtK1BlsRsFSqlZuegQ+bsMy4d+7xL1TYTlcodtrd07QMio/uYyNg+LNK9j4y07Th6pM92onnZ9+GSnn3EpHN/0KTv/riJnY1AHInYbJBf8ITPNbH8law2FEGgusmlXglSVSTd+ub9LMV93nhCQA138f5KvOO+wcQlZ5u2qaw/Qet1h7cY4Q6ciyOjg1Jxvoff50idX6l0myZVMrY4lGTPVqBI7W6TkUnF53/Y1TuK5tZ9Oz4irY8RZ9lGIUyitvhPmXB5Bxli91dyQqDTlqDBRKrE9wlQ3Gtu+Tc09U2/Ur5+4Sl+X4ntuUk+3LRt8xnk7D9B1/WE33+CeJyF", "4-9": "eJwdjssNwDAMQldhAA6BfEv336tuDkhgZPEaRbO/bnQ5050e9KQXvenztgyo5Ez00sjmwo4aD1YeSlCLDrVxok4ZKlPNgvbf+EBPJKpB/09lRYt6UPdJ1cbMpYBGLgjUc1kg5+JgZnF/Lqkiew==", "7-10": "eJw1jsuNRTEMQluhABYB58vrv6+JojsLJAsLOI2iWewcnL9qLLGuUazOGqzJWqzNOr+WBV05G3XVczCuZtSwIlEdUjSoDfVoURMa908XtONOD6hFnSq48qqh+k+t2LSgGTfqwM7DgpVHBrc8OOjk8cE9nvQ9RnzoDd94PWd9oz5fs/fXvCNTd8of6YmK8h+i5DTu", "8-9": "eJw1j8uNRTEIQ1uhAC9iE/Lx67+v4UaaBQhZYB8GCCExUVjYvxxIIltK5EQWciE38vyGT7BLvpFd0xxRX1smY3/tmAIzKPOCK5jmBvvyWhtqZVkJKbTdabqhVi7Uzm8n276Hghgqa0IZoiVohKYfYCj9GEPywwwNP9LQcd6XtS1+VzzPp+Ka9QDrH3CYE6xg/3PAHeRT9AcjhDZQ", "8-12": "eJw1j8sNAzAIQ1fxAD7Ehvzo/nuVRuoBBAibx6BoBpOTi/sT3QQjGZOxGJtxGJc5mGKaGZ9RB+pwXURHlgbmL62SsH/plIIyFKVLtcClTS3olie9oVNe9IBnvbPwrhgMIcbbSYTKm15w1oOCV/nQF265nrwNgzasesjwqEcNRz1w2PXYoV1Ouov2MS3cUlI9yD/gKE0qof7pUBfSm/gLD3E5rw==", "9-10": "eJw1kLt1QDEMQlfRABQWyJ9H9t8rsk/SUMjI4jKQIITCxMLG+alCTdRCbdRBfZgDMzGJqZ/hHJFX6C/kzKgr08lYV7ZTca5kmzdyRW5zgRm5zAIVXFYiZzDNA+5QWhuckdOc6L+3X5TgfN6KLJNgn6Y1wBMcZi+1eVgf1JNjdfQR+qxOz9Dxg4miq0E7sFzN/EXlNUuhdY9rhmQV1FvTDz5Ufvwh+lUQNf7+4bYmVMG+pbv1OetR1QvIoMxxA2Z39r0uvv8u6DzIfjqmwObTm+xfZvxNqw==", "12-10": "eJwtj0uSRTEIQrfiAhgEMJ/r2/++2qR6YilichggBCMxsbBx8IED5M8H/pADSaSQ7UnkRC7k/o3iDN6iosLFFXnLvOMqnthFxylmfHdkn+wgX52l/lfBdffMa6VLhhhyuZm6YWlDJ6xywjPM8oJHOOsBhmc9xvCqhxke5Ql9oa/UUbppZVyFrSyooVd5wys038t9tevlC7texNAuzbvSuWa1Wf9g7NAfNIKnlFB7snSgjrafoj+41EJE", "11-10": "eJwtj9GRRSEMQltJAXwIRK/m9d/XRmd/nIxAOBkgBCMxsfBh44Dj5wV/8IYPciCJFLJtiZy/UWSwmKGiwvfJ4oxZdKwrfHfaxRXnChx3ZGc+cAe/UkIO7tKGvlCWJ9wbXW4ghmc9inBLFySseixh1sMJ7bJfvPs22GbXYwzmg3SpDxwhlgd0QqM0oRUer6iXnHLD7NAs+VZrXRhlsO87N87zH2d7BLVHpQV10bo/PH/uCD9J", "15-10": "eJw1j8sRBDAIQluxAA4i5uf239eazO6NEGF4DiIgJAYmFjYO6CDBAAXmJxM5kBO5kBt5MByDGIGhjxeHsTgtisdU3JbXGxVus4K27se+z1MRRr8mWSFjPKdTy5jP6dxGLItd2lCLVRoQTbvSkS1WZY+ell5vnSXrDbSMehtNs5JQmrp2ItJiluZ1IkuCwqTKRm8Rt1nHUvXgTKcen2ncG7VgKe4N96Xl/Neei8xGOxZe8jte/h/Ph7Ye1LAYP5Zo8BZuoQ5+ATvbSws=", "20-15": "eJwtkMsVRDEIQluhABbxnzj99zUm5y3VA3JZFCqNzmCyuHkoiyIUpRjFKUFJSlE25fyiGJtxmIspTGUa05ljkMxibuZhLZb8VqtAWhe0VWF39FZHtBqyNVB33K2Jc0cZSUHkLmREBzKqDfG2BYm3ybaxrbfZbQo5d6Nyn6i2Oc1g3h70DbP2QZzX0TGkBt8dyQiE9wNCWD8mRPbDQkQ/MviclDFy6Vj0g9B+uAjpR4xY/aDhp2sqHeDVtl+M3b7pCc/vu/qlV2sLmsPixVO4fv4abUlfsGyvd1qPZcLUl9z9S6V5G9NqX7SEnluXTi1FC1g9h4RNnkMr2GmXa2VyqzN9Oc8f7mtk/g==", "1-11": "eJwtjsERBCEQAlMhAB6COq5s/nmda92X6uqmvVrUQ2260aJNd3rQky56vS2Ccjb0LBacyYEVDaqjokk1aOSaoI/ShmauEWopCjMSZYxsHko5JRXk3B5Uf2DnpvHknFg/vfcewA==", "1-13": "eJwdjrsRBDEUwlqhAALD85ftv6/zOWMIJLXPnR70pBe96cNqLLHMKlZnDdb8WgSlaFQGOxxxYuTc3bO5cKJ+L/XYtDCjxgM7TwErzwK1XDZ2NKjCikxd+sjihiq+mgZdxaHuo2hRF33yCqGdFwmtvE5o5qXCdbFw+0Nq/gCwcSqw", "9-4": "eJwdjssVADEIAluhAA6B/N3++1rNZZ6KIo2i2Tk4ubh5Pg960oveXws1qOCQ0AsjZMzCCnXswokL5fKEZrhRF25hQTd0aMMKd+a4x3uAFAY0StaGTrav2LVXB46XAdLzzQALWmla7soo6wdA1yPz", "12-1": "eJwdjMkRwDAMAluhAD0Esh2b9N9XIn8YZjkyGIqKETNWPLHjBDPIV/Wm+YBmQeZEmRujZTZb5sBjHuzuHVNgNuS/WeDNuK9fVkJpCZJV4LEIsQlH33M2Yd2fv5MfKIEd0g==", "12-2": "eJwVjskBADEIAluhAB6BxBxu/32tPke5BkVzcjG4eXj5qEHp86XfN1ITShlOLczGlQpE405tnMabOniNGq2W+qCZXvSEV7oiCgN2esM7feFIH/ikH1T3an7wSAuqtBKoBbptURU/9qtcQS+oBtTMAVWCq+gHG8QnsQ==", "24-16": "eJw1ksuRZTEIQ1MhAC3MH9P55zW6rjcbyoWNLA4cKAyOQKLQGFzogSrUoA4NaEIL2tCBXtiBscZg/ncN13EDN3ELt3Ep8ClQ4lDjUORQ5VDmUOdQ6PTfWWvRtRH7gq9diS/k+pFaV+l1k/nCXXfR86Kuh6itp6ivl2i8mGvM1Bpv+51njfn7RTtf3ljbYvYia/l3b1xEStwtNtYSue1olbo7FzMyd7NRLTNbgzqSvU1SLdVbJV37KEj3PhBSunNkznajS6p2UiZ3DKMyto+UjO6DJXV2FGNSs8M8ZXm2ncC4TOyPpozvD6i071XcI62bNEpbd3vQV3r2h1v67o+4FP8vmdocydm+6JGKLZI+Wxx+yNUtB59dOuIqsBXfYYbwZiMRV4w8DlIlz8eMqJJcDzrFz6Yi6Sl/5pzmHGmSvm1ouiR+/R63/QC7bRrSxX0zECXJkQbaJOpNgzqxUcgQz03aaEkuCg9HgtNgVUi8LrLFOUxu80izyhAqwTYL5RL+Zsg5GxsHMQdbZjtXvDYamWJUvp+g/S83Xvn3l/dmIQiQGzpw7uy832mefAZZ/wD4oaU1", "128-128": "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"}


def add_balancer(balancer_type, compact_string):
    common_balancers[balancer_type] = compact_string

def get_balancer(balancer_type) -> BeltGraph:
    if balancer_type in common_balancers:
        return common_balancers[balancer_type]

def is_balancer_defined(balancer_type):
    return balancer_type in common_balancers






# graph = BeltGraph()


# for t in range(100):
# 	ne = list(graph.possible_new_edges())
# 	if len(ne) == 0:
# 		print(f'No more possible edges.  Ended at {t}')
# 		break
# 	graph.add_edge(*choice(ne))
# 	print('\n')
# 	print(graph.compact_string)

if __name__ == '__main__':
    # bp_data = load_blueprint_string('0eNqlW0tv4zYQ/iuETi0gZ8WXRAZoD7310h6aPS2Cwo6ZWKgiGRK1u0Hg/17ayjseaTgM9hB7xW8enJlvSE0es00zun1ftz67fMy2brjp672vuza7zK5cy7pb5neODfum9t71A1v3jm067xvXupv/Bla3pwf8ru/Gu91+9BeM/enZtnMD29aD7+vN6B3jX+wRy61vdmFJeIz5bvrUjT58zNm63R6RAvxRRNu9k/JjF3RZN820dtLidmyaBzaM+6Cb2waxfxxRXxDeKXmbs2H9kDP/o3s26Q3SM8ZJh64NqKfnTopNT4zDScI/XVhaB7whKOjZ16uvF1me1TddO2SX3x6zob5r183Rk/5h74ILv9e9H8M3edau749fTE+sbHYI69qt+5ld8kNOXCkO13nmWl/72k0KnD48/NuO9xvXB+iX1e7nvnfDsPL9uh32Xe9XG9f4gL7vhnra78csQK7khc6zh/BLdaEPR80+QIp4SLEAKeMh+QKkiocsFiB1NOQSYhmNuGR2FY24tDcmGnEpgGw0olpA5EX8br9g6oCZZ9u6dzfT/4tzEj4n0nNFPJdCE3IJaCsSklIDmDJGPzGPpWKw+LytOgKrmFerjIBa0KqiJx6k3ec8GUON7u8CK7bbpdwrTzH4VPgn2snOCbH09EaFuSgifKxmfSw4vXAAPhaCXjkgSJlQORSAGZU/T16UAJZOqBUKteVlAp9DHqgSCB3CNAmYkHctIWlf+wSFy1pZJPQi8uMeluckcIId0WYIev8jMZEoYzhMzAaLVPQCD8SK1PTaC0GW9BKJi4uKXjAhzxo6JE7neI7TYME7J0AVdAHARipK/r3yiMAloBIJXMUxGahkAtsIwDcqAZMDmDqBs8RHT6hzEkrKjgrQ2+plf0+3Dme3N4U0BSryDaGxh7bAxjf2QIToIr6xB7TSnF73BSZDtKCzAE6ATDlH8LeVBIw0rei8g8oerekCOEpAzFlwvj7piuBwddbhcOnWhs45UKjbBDYogPuohHsUCDLlBrLAZEwpKKVavlEbsYGlTOKDz4YsZ2mpUkQWMAPNWKkpIvl7K5ellHSiw0VEFc9NUPga+tkNgrTRdAcgVUX8gQmC4mRegxAFnQMKDAdUkiygwHRLlUohhpmMhzOj0nSagLahpJd0gHmqKuW0I1AtSmVSivqnHqLEON+miMSZZQr6wQV1gjOcLgDVnxpBL95APJmE13+odtSolGs+XE9t4l90AA2pKcn3bJCDKQnLz5kPZ48xZALB7aElV3vU2dgWCdV+pt6AAWN5gkBctbGCTCcc5TNJxgdi3xJenQB3clbTiQ+4+7Ql/SwDQVYpTb7E5aY1dFJAXalaS6/gqFcDvCjoElAm8ILTO30JjSUIOiZSa0kmC1BplfAySaEKEy80+Z5O4vxSJlAe1gj6sA3SCDqnKlxWUVpeGVd8OGE4R0ZZETWbMx/7nD5hoCBIOkmCkIpObdD4Ck8YNNCoaI6a33kaoYLmd3gSbWpcfpNGel5Yp0QKsXSKgPYyaoqnmHW04NGXVaBWIvq2CtRKkgsjqJ1KKIXIgBI6eroK9EBJLlS4eUfCCI9ecrGhlxho8FNYemsLYcqEXhPETOguQUz6QA4IKcl9GQipyLkKQtJfL4KQ9Hw6QV7nWe3dfVj++ucUedasw9Lw3benWf7VNML/2/Mk/zVjq/DzO4Mf+OWvv6/e/G0FG9umvg+itr8G/O+uH6YUNlxVVlS6DP+UORz+B0Bzmvo=')
    # pprint(bp_data)
    # graph = BeltGraph.from_blueprint_string('0eNqlW0tv4zYQ/iuETi0gZ8WXRAZoD7310h6aPS2Cwo6ZWKgiGRK1u0Hg/17ayjseaTgM9hB7xW8enJlvSE0es00zun1ftz67fMy2brjp672vuza7zK5cy7pb5neODfum9t71A1v3jm067xvXupv/Bla3pwf8ru/Gu91+9BeM/enZtnMD29aD7+vN6B3jX+wRy61vdmFJeIz5bvrUjT58zNm63R6RAvxRRNu9k/JjF3RZN820dtLidmyaBzaM+6Cb2waxfxxRXxDeKXmbs2H9kDP/o3s26Q3SM8ZJh64NqKfnTopNT4zDScI/XVhaB7whKOjZ16uvF1me1TddO2SX3x6zob5r183Rk/5h74ILv9e9H8M3edau749fTE+sbHYI69qt+5ld8kNOXCkO13nmWl/72k0KnD48/NuO9xvXB+iX1e7nvnfDsPL9uh32Xe9XG9f4gL7vhnra78csQK7khc6zh/BLdaEPR80+QIp4SLEAKeMh+QKkiocsFiB1NOQSYhmNuGR2FY24tDcmGnEpgGw0olpA5EX8br9g6oCZZ9u6dzfT/4tzEj4n0nNFPJdCE3IJaCsSklIDmDJGPzGPpWKw+LytOgKrmFerjIBa0KqiJx6k3ec8GUON7u8CK7bbpdwrTzH4VPgn2snOCbH09EaFuSgifKxmfSw4vXAAPhaCXjkgSJlQORSAGZU/T16UAJZOqBUKteVlAp9DHqgSCB3CNAmYkHctIWlf+wSFy1pZJPQi8uMeluckcIId0WYIev8jMZEoYzhMzAaLVPQCD8SK1PTaC0GW9BKJi4uKXjAhzxo6JE7neI7TYME7J0AVdAHARipK/r3yiMAloBIJXMUxGahkAtsIwDcqAZMDmDqBs8RHT6hzEkrKjgrQ2+plf0+3Dme3N4U0BSryDaGxh7bAxjf2QIToIr6xB7TSnF73BSZDtKCzAE6ATDlH8LeVBIw0rei8g8oerekCOEpAzFlwvj7piuBwddbhcOnWhs45UKjbBDYogPuohHsUCDLlBrLAZEwpKKVavlEbsYGlTOKDz4YsZ2mpUkQWMAPNWKkpIvl7K5ellHSiw0VEFc9NUPga+tkNgrTRdAcgVUX8gQmC4mRegxAFnQMKDAdUkiygwHRLlUohhpmMhzOj0nSagLahpJd0gHmqKuW0I1AtSmVSivqnHqLEON+miMSZZQr6wQV1gjOcLgDVnxpBL95APJmE13+odtSolGs+XE9t4l90AA2pKcn3bJCDKQnLz5kPZ48xZALB7aElV3vU2dgWCdV+pt6AAWN5gkBctbGCTCcc5TNJxgdi3xJenQB3clbTiQ+4+7Ql/SwDQVYpTb7E5aY1dFJAXalaS6/gqFcDvCjoElAm8ILTO30JjSUIOiZSa0kmC1BplfAySaEKEy80+Z5O4vxSJlAe1gj6sA3SCDqnKlxWUVpeGVd8OGE4R0ZZETWbMx/7nD5hoCBIOkmCkIpObdD4Ck8YNNCoaI6a33kaoYLmd3gSbWpcfpNGel5Yp0QKsXSKgPYyaoqnmHW04NGXVaBWIvq2CtRKkgsjqJ1KKIXIgBI6eroK9EBJLlS4eUfCCI9ecrGhlxho8FNYemsLYcqEXhPETOguQUz6QA4IKcl9GQipyLkKQtJfL4KQ9Hw6QV7nWe3dfVj++ucUedasw9Lw3benWf7VNML/2/Mk/zVjq/DzO4Mf+OWvv6/e/G0FG9umvg+itr8G/O+uH6YUNlxVVlS6DP+UORz+B0Bzmvo=')
    # print(graph)
    # graph.delete_edge(14, 32) # turns 9-9 into 9-8
    # pprint(graph.evaluate())

    # graph = BeltGraph.from_compact_string('eJwtT8uVRTEIasUCWAian6//vkbvmQ2JCgoKKKEFbehAF3oIRxDJXwgRiEQsxEYcxEU8pP8UlTJl8ZpWNeziM50Pb4mmNxhekgUHkyNKjYi3uMBttx4oY7N9arHYTtyk+XAZV7FtGPsEse1Vt409dvRh/5+e2mD3o5izsR2i97IWjrUUbszqTHTrdyKZ6gtl9BE9O01Oa7NNXvWFN68v8h+0Fzk8')
    graph = BeltGraph.from_blueprint_string('0eNql0+FqwyAQB/B3uc+mRBsb46uUMZLtGEJiRK+jIfjuMxmjpTiyZuAHFe7nXzln6PoLOm8sgZ7BvI02gD7PEMyHbftljyaHoMEQDsDAtsOywqvzGEIRXG+I0ENkYOw7XkHzyP5cTr61wY2eig57ukNEfGGAlgwZ/A60LqZXexm6dJzmWxYDN4ZUPtolRSILqZqDZDClKS+rg4xL0AdW7GCb8saKPHv8/eGy4A93zHPVU5xqNtLJf74lz7OnPay6sWWerZ+7vNpIqfakrB9SpmZdG1zffScGn+jDWiYUr+pG1PKURqVi/AIjDSPW')
    print(graph)
    print(graph.get_vertex_depths())


    from fischer.factoriobps import blueprints_in_book
    from pprint import pprint

    bp_data = load_blueprint_string_from_file('belt_balancer_book.txt')

    graph = BeltGraph()
    for bp in blueprints_in_book(bp_data):
        if try_derive_graph(bp, graph):
            if graph.balancer_type != '1-1':
                print(graph.balancer_type)
                # if graph.balancer_type == '8-8':
                # 	print(graph)
                evaluation = graph.evaluate()
                if evaluation['accuracy'] == 1:
                    print('Accurate')
                    common_balancers[graph.balancer_type] = graph.compact_string
                else:
                    print('Inaccurate')

    pprint(common_balancers)

