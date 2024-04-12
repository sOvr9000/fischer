
from fischer.factoriobalancers import BeltGraph





def convert_4_4_to_3_2() -> BeltGraph:
    # Create a BeltGraph and load a 4-4 balancer.
    graph = BeltGraph()
    graph.load_common_balancer('4-4')

    # To convert the 4-4 graph into a 3-2 graph, the simple and naive solution is to insert a 2-1 balancer graph that connects two outputs to one input.
    # Load the 2-1 graph.
    merger = BeltGraph()
    merger.load_common_balancer('2-1')

    # Remove the last two outputs and their edges, and retrieve the vertices that connect to them.
    from1 = graph.disconnect_vertex(graph.outputs[-2])
    from2 = graph.disconnect_vertex(graph.outputs[-1])

    # Remove the last input and its edge, and retrieve the vertex to which it is connected.
    to = graph.disconnect_vertex(graph.inputs[-1])

    # Insert the 2-1 merger.
    graph.insert_graph_between(merger, [from1, from2], [to])

    graph.simplify()

    # Rearrange the vertices so that it's easier to follow in the display.
    graph = graph.rearrange_vertices_by_depth()
    
    # The graph is now a 3-2 balancer.
    return graph



def main():
    graph = convert_4_4_to_3_2()
    print(graph)
    print(graph.evaluate())

    # Try to simplify the 

    # Compare to simplified 3-2 balancer.
    graph.load_common_balancer('3-2')
    print(graph)
    print(graph.evaluate())


if __name__ == '__main__':
    main()







