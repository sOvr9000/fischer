
from fischer.factoriobalancers import BeltGraph





def double_5_7() -> BeltGraph:
    # Create a BeltGraph
    graph = BeltGraph()

    # Load a 4-4 balancer.
    graph.load_common_balancer('5-7')

    # Double the number of inputs and outputs.
    graph = graph.doubled()

    # Rearrange vertices so that it's easier to follow in the display.
    graph = graph.rearrange_vertices_by_depth()

    # View and evaluate the new graph.
    print(graph)
    print(graph.evaluate())



def main():
    double_5_7()


if __name__ == '__main__':
    main()







