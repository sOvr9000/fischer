
from fischer.factoriobalancers import BeltGraph





def convert_128() -> BeltGraph:
    graph = BeltGraph()
    graph.load_common_balancer('128-128')

    graph2 = BeltGraph()
    graph2.load_common_balancer('1-8')

    from_vertices = []
    for _ in range(graph2.num_inputs):
        from_vertices.append(graph.disconnect_vertex(graph.outputs[-1]))

    to_vertices = []
    for _ in range(graph2.num_outputs):
        to_vertices.append(graph.disconnect_vertex(graph.inputs[-1]))

    graph.insert_graph_between(graph2, from_vertices, to_vertices)

    # graph = graph.rearrange_vertices_by_depth()
    
    return graph



def main():
    graph = convert_128()
    with open(f'./{graph.balancer_type}_edges.txt', 'w') as f:
        f.write(repr(graph))
    with open(f'./{graph.balancer_type}_output_flow.txt', 'w') as f:
        f.write(repr(graph.evaluate()))


if __name__ == '__main__':
    main()







