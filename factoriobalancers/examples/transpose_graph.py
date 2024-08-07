
from fischer.factoriobalancers import BeltGraph
from fischer.factoriobps import get_blueprint_string
from construct_m_n_alt import construct_m_n_alt
import json
from clipboard import paste, copy




def transpose_graph():
    graph = BeltGraph()

    # Load a 12-10 balancer from a blueprint string.
    # graph.load_blueprint_string('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')
    graph.load_blueprint_string(paste())

    # Transpose the graph.
    graph = graph.transposed()
    print(graph)
    print(graph.evaluate())

    print('Graph summary:')
    print(graph.summary)
    print('Graph is solved:')
    print(graph.is_solved())

    copy(get_blueprint_string(graph))



def main():
    transpose_graph()



if __name__ == '__main__':
    main()







