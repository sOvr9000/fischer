


from fischer.factoriobps import blueprints_in_book, load_blueprint_string_from_file
from fischer.factoriobalancers import try_derive_graph, BeltGraph, Evaluation
import json



def main():
    bp_book_data = load_blueprint_string_from_file('./examples/read_blueprint_book_string.txt')
    output: dict[str, tuple[BeltGraph, str, Evaluation]] = {}
    for bp in blueprints_in_book(bp_book_data):
        graph = BeltGraph()
        if try_derive_graph(bp, graph, verbose=False):
            print(graph.summary)
            graph.simplify()
            graph = graph.rearrange_vertices_by_depth()
            ev = graph.evaluate()
            if graph.is_solved(ev):
                overwrite = False
                if graph.balancer_type in output:
                    if len(ev['bottlenecks']) < len(output[graph.balancer_type][2]['bottlenecks']):
                        overwrite = True
                    elif graph.num_vertices < output[graph.balancer_type][0].num_vertices:
                        overwrite = True
                else:
                    overwrite = True
                if overwrite:
                    output[graph.balancer_type] = graph, graph.compact_string, ev
                d = {bt: cs for bt, (_, cs, _) in output.items()}
                json.dump(d, open('./examples/read_blueprint_book_output.json', 'w'))

if __name__ == '__main__':
    main()




