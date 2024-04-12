
from fischer.factoriobalancers import BeltGraph





def simplify_4_4() -> BeltGraph:
    # Create a BeltGraph
    graph = BeltGraph()

    # Load a 4-4 balancer that has two unnecessary splitters at the end.
    graph.load_blueprint_string('0eNqdlm1vgjAQx7+K6WswbSkP8lWWZUHtTBMspBxmxvDdVyTbdOMGvVfq4f24h/+1d2P7utetMxZYeWPm0NiOlS831pmTrerRBtdWs5JdjIPeWyJmq/NomP4RKzZEzNij/mClGCKipwzydPr44JsMrxHTFgwYPQV//3F9s/15r50P69v/veogBlfZrm0cxHtdg4e3Ted9Gzu+2fNivk0jdvVf1DYdxsB+8eQzr2trA+Af/CWJiZPMc5LAuMRCXCqMt4RL16b5f5ZZWFRLxc+fcb3XgDu5xn8utlN6YvSlraaHtgc284aCKhdx5x+N04fpeTZD31GbLufrITgVKBCgIMoIC1ASeVh8SagCQgUgFFGyfwQg5+ir54pPVI7UISPqFOPlRCHxVVkXNBXwNTMldrSGIZWQPKxDiPClIBYU40kiDxkkmZCPUvE4SMYicySJFxKWf0rDYeln1HNkZfY5cT6xraMIXDuQ+1hSbyAkroTT2oLhRNjage1WknYk3KPy+6QBffa+P9txxOrKu3qb2kCzUd5w0a6bDrxCqHwn8zzJd2mhhuETvc25kQ==')

    # Verify that the graph is already solved.
    print('\n' * 2)
    print(f'Solved: {graph.is_solved()}')

    # Check all 2-2 splitters to see if they can be removed without invalidating the balancer.
    r = list(graph.removable_vertices())
    print('Removable vertices:')
    print(r)

    # Remove one of the removable vertices.
    graph.delete_balancer_vertex(r[0])

    # Remove another one because it's possible with this unsimplified 4-4 balancer.
    # NOTE: We cannot safely reuse the previously generated list of removable vertices here.
    graph.delete_balancer_vertex(next(graph.removable_vertices()))

    # Rearrange vertices so that it's easier to follow in the display.
    graph = graph.rearrange_vertices_by_depth()

    # View and evaluate the new graph.
    print('\n' * 2)
    print(graph)
    print(graph.evaluate())

    # Show that there are no more removable vertices from the simplified graph.
    r = list(graph.removable_vertices())
    print('Removable vertices:')
    print(r)



def main():
    simplify_4_4()


if __name__ == '__main__':
    main()







