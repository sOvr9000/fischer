
import os
import sys
from fischer.factoriobalancers import BeltGraph, is_balancer_defined, add_balancer, BeltDesigner
from multiprocessing import Pool



sys.setrecursionlimit(2000)

splitter1to2 = BeltGraph()
splitter1to2.set_input(0)
splitter1to2.set_output(2)
splitter1to2.set_output(3)
splitter1to2.add_edge(0, 1)
splitter1to2.add_edge(1, 2)
splitter1to2.add_edge(1, 3)

splitter2to2 = splitter1to2.copy_graph()
splitter2to2.set_input(4)
splitter2to2.add_edge(4, 1)
splitter2to2 = splitter2to2.rearrange_vertices_by_depth()



def construct_m_n_alt2(m: int, n: int, action_stacks_txt: str, m_split: int = None) -> BeltGraph:
    '''
    Algorithm:

    If m is 1:
        If n is even:
            1. Construct the graph for 1-(n//2).
            2. Split each output into two outputs.
        else:
            1. Refer to action_stacks.txt and construct the partial graph for 1-n.
            2. Construct the necessary 1-k splitters to split the outputs into 1/n of the input.
    else:
        1. Construct the graphs for j-n and k-n where j=floor(m/2) and k=ceil(m/2).
        2. Substitute functionally equivalent groups of vertices from one graph to the other, in either order.
    
    Substitution of functionally equivalent groups of vertices:
    1. TODO


    If `m_split` is None, then `m_split` is set to `m // 2`.  This is used to construct two separate graphs `m_split`-n and `m - m_split`-n in order to construct the final graph for m-n.
    '''
    def split_output(graph: BeltGraph, output: int, splitter: BeltGraph) -> None:
        u = graph.disconnect_output(output)
        (v,), outputs = graph.insert_graph(splitter)
        graph.add_edge(u, v)
        for r in outputs:
            s = graph.new_vertex()
            graph.add_edge(r, s)
            graph.set_output(s)
    
    def substitute(graph_composite: BeltGraph, u: int, v: int) -> None:
        # graph_composite is a single BeltGraph object that contains two separate graphs.
        # u and v are vertices of equal depth in the two separate graphs.
        # Both vertices must have exactly one input and two outputs.
        _u, = graph_composite.in_vertices(u)
        graph_composite.delete_edge(_u, u)
        graph_composite.add_edge(_u, v)
        to_remove = []
        for i, w in enumerate(graph_composite.dfs(u)):
            if i == 0:
                continue
            if graph_composite.in_degree(w) == 2:
                break
            to_remove.append(w)
        for w in to_remove:
            if w in graph_composite.outputs:
                graph_composite.set_output(w, False)
            graph_composite.delete_vertex(w)
        # Clean up disconnected subgraphs
        vertices = set(graph_composite.dfs())
        for v in graph_composite.vertices():
            if v not in vertices:
                graph_composite.delete_vertex(v)
                graph_composite.set_output(v, False)

    balancer_type = f'{m}-{n}'
    if is_balancer_defined(balancer_type):
        # print(f'Loaded balancer {balancer_type}')
        return BeltGraph.from_common_balancers(balancer_type)

    graph = BeltGraph()
    if m == 1:
        if n == 2:
            return splitter1to2.copy_graph()
        if n % 2 == 0:
            # Construct the graph for 1-(n//2)
            graph = construct_m_n_alt2(1, n//2, action_stacks_txt)

            # Attach a splitter at the end of each output.
            for u in reversed(graph.outputs):
                split_output(graph, u, splitter1to2)
        else:
            # Find the action stack from action_stacks.txt
            with open(action_stacks_txt, 'r') as f:
                found = False
                for line in f:
                    if found:
                        # The value of `line` will be the action stack for 1-n
                        break
                    if line.startswith(f'1/{n}'):
                        found = True
                else:
                    raise ValueError(f'No action stack found for 1/{n}')
            des = BeltDesigner()
            des.action(line)
            graph = des.graph
            ev = graph.evaluate()
            flow = ev['output_flow']
            # First ensure that all denominators are not greater than n.
            if any(flow[u][0].denominator > n for u in graph.outputs):
                bad = [u for u in graph.outputs if flow[u][0].denominator != n]
                # print(ev)
                # print(f'Bad outputs: {bad}')
                for vi in range(0, len(bad), 2):
                    v1 = bad[vi]
                    v2 = bad[vi + 1]
                    u1 = graph.in_vertices(v1)[0]
                    u2 = graph.in_vertices(v2)[0]
                    graph.delete_edge(u1, v1)
                    graph.delete_edge(u2, v2)
                    graph.insert_graph_between(splitter2to2, [u1, u2], [v1, v2])
                ev = graph.evaluate()
                flow = ev['output_flow']
                
            for u in reversed(graph.outputs):
                fraction = flow[u][0]
                num = fraction.numerator
                if fraction.denominator < n:
                    num *= n // fraction.denominator
                if num == 1:
                    continue
                splitter = construct_m_n_alt2(1, num, action_stacks_txt)
                split_output(graph, u, splitter)
            
            # print(f'Constructed 1-{n} from action stack {line}')
    else:
        if m_split is None:
            m_split = m // 2
        j = m_split
        k = m - j
        graph_j = construct_m_n_alt2(j, n, action_stacks_txt)
        graph_k = construct_m_n_alt2(k, n, action_stacks_txt)
        if graph_j is None:
            print(f'Could not construct subgraph {j}-{n} (j={j})')
            return
        if graph_k is None:
            print(f'Could not construct subgraph {k}-{n} (k={k})')
            return
        # print(f'j={j}, k={k}, m={m}, n={n}')
        # print(f'Graph j:\n{graph_j}')
        # print(graph_j.evaluate())
        # print(f'Graph k:\n{graph_k}')
        # print(graph_k.evaluate())
        # input()
        # Substitute functionally equivalent groups of vertices.
        # It can be done expensively by checking graph isomorphicity.
        # But this algorithm is simpler and may even be more efficient as it tests substitutions between equal-depth vertices and picks the one that works at the lowest depth.
        graph_composite = BeltGraph()
        for g in (graph_j, graph_k):
            ins, outs = graph_composite.insert_graph(g)
            for v in ins:
                u = graph_composite.new_vertex()
                graph_composite.set_input(u)
                graph_composite.add_edge(u, v)
            for u in outs:
                v = graph_composite.new_vertex()
                graph_composite.set_output(v)
                graph_composite.add_edge(u, v)
        vertices = graph_composite.get_vertices_by_depth()
        V_j, V_k = graph_composite.disjoint_vertices()
        # print(V_j, V_k)
        # print(vertices)
        # print(graph_composite)
        # input()

        def dfs(g: BeltGraph, d: int, v0i: int, v1i: int) -> BeltGraph:
            # Search first along v1i, then along v0i, and then along d.
            # At the end of each branch (where d exceeds the number of lists in `vertices`), the current graph is evaluated and returned only if it is solved.
            # os.system('cls')
            # print(f'Searching with parameters {d}, {v0i}, {v1i}')
            # input()
            if d >= len(vertices):
                # Reached end of a branch in DFS
                # print(f'End of branch for search m={m}, n={n}.  Evaluating graph {g.advanced_summary}:')
                # print(g.evaluate())
                if g.is_solved():
                    return g
                return

            # The two bounds below are extremely important for reducing the search space exponentially.  We search v0 between 0 and len(vertices[d])//2, and v1 between len(vertices[d])//2+1 and len(vertices[d]).
            # if v0i >= len(vertices[d]) // 2:
            #     return dfs(g, d=d+1, v0i=0, v1i=len(vertices[d])//2+1)
            # if v1i >= len(vertices[d]):
            #     return dfs(g, d=d, v0i=v0i+1, v1i=len(vertices[d])//2+1)
            # This version of the check allows v0 to start from the same position in v0's interval as v1 in v1's interval.  This is where substitutions are more likely to be correct, especially if j == k.
            if v0i >= len(vertices[d]) // 2:
                return dfs(g, d=d+1, v0i=0, v1i=len(vertices[d])//2)
            if (v0i == 0 and v1i >= len(vertices[d])) or (v0i > 0 and v1i == len(vertices[d])//2+v0i-1):
                return dfs(g, d=d, v0i=v0i+1, v1i=len(vertices[d])//2+v0i+1)
            if v1i >= len(vertices[d]):
                return dfs(g, d=d, v0i=v0i, v1i=len(vertices[d])//2)

            # Also check for substitutions that likely make use of feedback loops when necessary.  For example, if a vertex in graph_j does not precede a feedback loop, then it is likely that the correct corresponding vertex to be substituted in graph_k does not precede a feedback loop.
            # TODO

            v0 = vertices[d][v0i]
            v1 = vertices[d][v1i]
            if not g.has_vertex(v0) or not g.has_vertex(v1) or g.out_degree(v0) != 2 or g.out_degree(v1) != 2 or g.in_degree(v0) != 1 or g.in_degree(v1)  != 1 or (v0 in V_j) != (v1 in V_k):
                # These parameters do not match substitution conditions.  Skip onto the next set of parameters.
                # TODO Also check if the resulting graph after substitution is one that's already been searched.  If so, skip onto the next set of parameters.  This should be accomplished by storing the connected vertex pairs as a sorted tuple in a set, making it so that it's less likely that any two searched graphs are not isomorphic.
                return dfs(g, d=d, v0i=v0i, v1i=v1i+1)
            # Now check substitution.  This is where the branches split because the substitution may not be correct.
            _g = None
            if not g.has_edge(g.in_vertices(v0)[0], v1):
                _g = g.copy_graph()
                substitute(_g, v0, v1)
                # if _g is not None:
                #     solved_g = dfs(_g, d=d, v0i=v0i, v1i=v1i+1)
                #     if solved_g is not None:
                #         return solved_g
            elif not g.has_edge(g.in_vertices(v1)[0], v0):
                _g = g.copy_graph()
                substitute(_g, v1, v0)
                # if _g is not None:
                #     solved_g = dfs(_g, d=d, v0i=v0i, v1i=v1i+1)
                #     if solved_g is not None:
                #         return solved_g
            if _g is not None:
                solved_g = dfs(_g, d=d, v0i=v0i, v1i=v1i+1)
                if solved_g is not None:
                    return solved_g
            # No branches preceding this current part of the search have succeeded.  Try again without substitution between v0 and v1.
            return dfs(g, d=d, v0i=v0i, v1i=v1i+1)

        graph = dfs(graph_composite, d=1, v0i=0, v1i=len(vertices[1])//2)
        if graph is None:
            return
        graph = graph.rearrange_vertices_by_depth()
    if not graph.is_solved():
        raise ValueError('Graph is not solved')
    add_balancer(balancer_type, graph.compact_string)
    return graph



def construct_m_n_alt2_mp(m: int, n: int, action_stacks_txt: str, workers: int = 8) -> BeltGraph:
    '''
    Use multiprocessing to try to construct the graph for m-n using different splits of m.
    '''
    # Async map, returning the first successful graph.
    with Pool(workers) as p:
        results = p.starmap_async(construct_m_n_alt2, [(m, n, action_stacks_txt, m_split) for m_split in range(1, m)])
        for graph in results.get():
            if graph is not None:
                return graph

def construct_m_n_alt2_mp_precendent(m: int, n: int, action_stacks_txt: str, workers: int = 8) -> BeltGraph:
    '''
    Try to construct all graphs k-n where `1 <= k <= m`, and save whicher graphs are solved.  Return the graph for m-n if it is solved.
    '''
    for _m in range(1, m+1):
        graph = construct_m_n_alt2_mp(_m, n, action_stacks_txt, workers=workers)
        if graph is not None:
            print(f'Solved {graph.advanced_summary}')
        else:
            print(f'Failed {_m}-{n}')
    return graph




def main():
    action_stacks_txt = 'factoriobalancers/action_stacks.txt'
    # for m in range(2, 129):
    #     for n in range(m + 1, 129):
    #         os.system('cls')
    #         print(f'CONSTRUCTING {m}-{n}')
    #         graph = construct_m_n_alt2(m, n, action_stacks_txt)
    #         print(graph.advanced_summary)
    #         print(graph.evaluate())
    #         input()
    graph = construct_m_n_alt2(2, 37, action_stacks_txt, m_split=3)
    if graph is None:
        print('No graph solution found')
        return
    print(graph.advanced_summary)
    print(graph.evaluate())

    # from fischer.factoriobalancers import BeltGrid
    # from fischer.factoriobps import get_blueprint_string
    # grid = BeltGrid(size=(13, graph.num_internal_vertices * 6 - 2), max_inputs=graph.num_inputs, max_outputs=graph.num_outputs, max_splitters=graph.num_internal_vertices, max_turtles=1)
    # if grid.generate_from_graph(graph):
    #     print(grid)
    #     print(get_blueprint_string(grid.to_blueprint_data()))
    # else:
    #     print('No grid solution found')



if __name__ == '__main__':
    main()


