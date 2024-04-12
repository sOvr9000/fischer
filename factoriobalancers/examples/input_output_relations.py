
from fischer.factoriobalancers import BeltGraph, common_balancers

from sympy.ntheory import primefactors





def input_output_relations() -> BeltGraph:
    # For each common balancer, compare the prime factorizations of the number of inputs and the number of outputs to the structure of the graph.
    view_first = ['5-7', '7-5']
    for balancer_type in view_first:
        print(balancer_type)
        graph = BeltGraph()
        graph.load_common_balancer(balancer_type)
        graph = graph.rearrange_vertices_by_depth()
        print(graph)
        print(hash(graph))
        print(primefactors(graph.num_inputs), primefactors(graph.num_outputs))
        input()
    for balancer_type, compact_str in common_balancers.items():
        if balancer_type in view_first: continue
        print(balancer_type)
        graph = BeltGraph()
        graph.load_compact_string(compact_str)
        graph = graph.rearrange_vertices_by_depth()
        print(graph)
        print(hash(graph))
        print(primefactors(graph.num_inputs), primefactors(graph.num_outputs))
        input()


def main():
    input_output_relations()


if __name__ == '__main__':
    main()







