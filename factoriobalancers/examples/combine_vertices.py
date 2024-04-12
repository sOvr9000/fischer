

from fischer.factoriobalancers import BeltGraph


def main():
    # Set up a 2-2 balancer with more vertices than necessary.
    graph = BeltGraph()
    graph.set_input(0)
    graph.set_input(1)
    graph.set_output(5)
    graph.set_output(6)

    # Set up a graph that essentially represents two lines of belts with three 2-2 splitters balancing them,
    # although there is only a single belt connecting between the first and second splitters as well as the second and third splitters.
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    graph.add_edge(3, 4)
    graph.add_edge(4, 5)
    graph.add_edge(4, 6)

    # View the bottlenecked flow.
    print(graph.evaluate())

    # To remove the bottleneck, we'd add belts (a second edge) between each splitter (vertex).
    # However, we are not allowed to add duplicate edges, so we must (more efficiently) remove one vertex and adjust the edges accordingly,
    # so it's like combining the vertices into one.
    # Combine vertices 3 and 4.
    graph.combine_vertices(3, 4)
    print(graph)

    # Notice that the graph no longer has the vertex 3.  So instead of combining vertices 2 and 3, we must combine 2 and 4.
    graph.combine_vertices(2, 3)
    print(graph)

    # View the non-bottlenecked flow.
    print(graph.evaluate())

if __name__ == '__main__':
    main()
