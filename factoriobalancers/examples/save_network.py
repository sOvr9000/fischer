
from construct_m_n_alt import construct_m_n_alt
from fischer.factoriobalancers import BeltGraph
import json





def save_network() -> BeltGraph:
    graph = construct_m_n_alt(7, 12)

    # View and evaluate the new graph.
    print(graph)
    print(graph.evaluate())

    graph.save_as_factorio_sat_network('./examples/networks')

    # d = json.load(open('examples/construct_m_n_alt_output.json', 'r'))
    # for balancer_type, compact_string in d.items():
    #     print(balancer_type)
    #     graph = BeltGraph()
    #     graph.load_compact_string(compact_string)
    #     graph.save_as_factorio_sat_network('examples/networks')



def main():
    save_network()


if __name__ == '__main__':
    main()







