
from fischer.factoriobalancers import BeltGraph





def construct_3_2() -> BeltGraph:
    # Create a BeltGraph
    graph = BeltGraph()

    # Define vertices 0, 1, and 2 as inputs.
    graph.set_input(0)
    graph.set_input(1)
    graph.set_input(2)
    # Equivalent to below:
    # graph.set_num_inputs(3)

    # Define vertices 3 and 4 as outputs.
    graph.set_output(3)
    graph.set_output(4)
    # Equivalent to below:
    # graph.set_num_outputs(2)

    # Connect a new vertex to the two outputs.
    graph.add_edge(5, 3)
    graph.add_edge(5, 4)

    # Connect inputs 0 and 1 to a new vertex.
    graph.add_edge(0, 6)
    graph.add_edge(1, 6)

    # Connect input 2 to a new vertex.
    graph.add_edge(2, 7)

    # Connect everything else to make this graph represent a 3-2 balancer.
    graph.add_edge(6, 5)
    graph.add_edge(6, 8)
    graph.add_edge(7, 5)
    graph.add_edge(7, 8)
    graph.add_edge(8, 7)

    return graph



def main():
    graph = construct_3_2()
    print(graph)
    print(graph.evaluate())


if __name__ == '__main__':
    main()







