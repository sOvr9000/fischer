
from fischer.factoriobalancers import BeltGraph, is_balancer_defined
from math import log2





def construct_m_n(m: int, n: int) -> BeltGraph:
    print(f'Constructing {m}-{n}')
    l = log2(max(m, n))
    pow2 = int(2 ** int(1+l))

    if log2(m) == int(log2(m)) or log2(n) == int(log2(n)):
        graph = BeltGraph()
        graph.load_common_balancer(f'{m}-{n}')
        return graph

    graph = BeltGraph()
    graph.load_common_balancer(f'{pow2}-{pow2}')

    if is_balancer_defined(f'{pow2-n}-{pow2-m}'):
        graph2 = BeltGraph()
        graph2.load_common_balancer(f'{pow2-n}-{pow2-m}')
    else:
        graph2 = construct_m_n(pow2-n, pow2-m)

    graph.insert_graph_between(graph2, [graph.disconnect_vertex(graph.outputs[-1]) for _ in range(graph2.num_inputs)], [graph.disconnect_vertex(graph.inputs[-1]) for _ in range(graph2.num_outputs)])
    graph = graph.rearrange_vertices_by_depth()

    return graph



def main():
    graph = construct_m_n(10, 13)
    with open(f'./{graph.balancer_type}_edges.txt', 'w') as f:
        f.write(repr(graph))
    with open(f'./{graph.balancer_type}_output_flow.txt', 'w') as f:
        f.write(repr(graph.evaluate()))


if __name__ == '__main__':
    main()







