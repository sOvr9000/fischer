
from fischer.factoriobalancers import BeltGraph, BeltGrid



def generate_bp() -> str:
    graph = BeltGraph()

    # Can load this graph with `graph.load_common_balancer('4-4')`, but for the sake of demonstration, this is how it is constructed.
    # Vertices 0-3 are inputs and vertices 4-7 are outputs.
    graph.set_num_inputs(4)
    graph.set_num_outputs(4)

    # Add 4 internal vertices, although 6 should be used for unbottlenecked throughput in the case of partial input.
    graph.add_vertices(8, 9, 10, 11)

    # Add edges for inputs.
    graph.add_edge(0, 8)
    graph.add_edge(1, 8)
    graph.add_edge(2, 9)
    graph.add_edge(3, 9)

    # Add edges for internal vertices (between splitters).
    graph.add_edge(8, 10)
    graph.add_edge(9, 10)
    graph.add_edge(8, 11)
    graph.add_edge(9, 11)

    # Add edges for outputs.
    graph.add_edge(10, 4)
    graph.add_edge(10, 5)
    graph.add_edge(11, 6)
    graph.add_edge(11, 7)

    # Evaluate the graph to validate the throughput.
    print(graph)
    print(graph.evaluate())
    print(graph.advanced_summary)

    # Construct a grid from the graph
    grid = BeltGrid((6, 22), max_inputs=4, max_outputs=4, max_splitters=graph.num_internal_vertices, max_turtles=1)
    grid.generate_from_graph(graph, max_underground_length=8)

    print(grid)




def main():
    generate_bp()



if __name__ == '__main__':
    main()


