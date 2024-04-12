
from pprint import pprint
from fischer.factoriobalancers import BeltGraph, common_balancers

from sympy.ntheory import primefactors





def vertex_depths() -> BeltGraph:
    graph = BeltGraph()

    graph.load_common_balancer('5-7')

    print(graph)
    pprint(graph.get_vertex_depths())


def main():
    vertex_depths()


if __name__ == '__main__':
    main()







