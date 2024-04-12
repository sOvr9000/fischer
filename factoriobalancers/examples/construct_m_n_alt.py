
from fischer.factoriobalancers import BeltGraph, is_balancer_defined
# from math import log2
import json


from multiprocessing import Pool


graph_lru_cache_memo = {}
def graph_lru_cache(func):
    def wrapper(*args):
        if args in graph_lru_cache_memo:
            return graph_lru_cache_memo[args].copy_graph()
        g = func(*args)
        graph_lru_cache_memo[args] = g
        return g.copy_graph()
    return wrapper


# @graph_lru_cache
def construct_m_n_alt(m: int, n: int) -> BeltGraph:
    graph = BeltGraph()

    if is_balancer_defined(f'{m}-{n}'):
        graph.load_common_balancer(f'{m}-{n}')
        return graph

    if m % 2 == 0 and n % 2 == 0:
        graph = construct_m_n_alt(m // 2, n // 2)
        graph = graph.doubled()
        graph.simplify()
        graph = graph.rearrange_vertices_by_depth()
        return graph

    if m % 2 == 1 and n % 2 == 0:
        graph = construct_m_n_alt(m+3, n+2)
        us = graph.disconnect_num_outputs(2)
        vs = graph.disconnect_num_inputs(3)
        splitter = construct_m_n_alt(2, 3)
        graph.insert_graph_between(splitter, us, vs)
        graph.simplify()
        graph = graph.rearrange_vertices_by_depth()
        return graph
    
    if m % 2 == 0 and n % 2 == 1:
        graph = construct_m_n_alt(m+2, n+1)
        us = graph.disconnect_num_outputs(1)
        vs = graph.disconnect_num_inputs(2)
        splitter = construct_m_n_alt(1, 2)
        graph.insert_graph_between(splitter, us, vs)
        graph.simplify()
        graph = graph.rearrange_vertices_by_depth()
        return graph

    if m % 2 == 1 and n % 2 == 1:
        graph = construct_m_n_alt(m+3, n+1)
        us = graph.disconnect_num_outputs(1)
        vs = graph.disconnect_num_inputs(3)
        splitter = construct_m_n_alt(1, 3)
        graph.insert_graph_between(splitter, us, vs)
        graph.simplify()
        graph = graph.rearrange_vertices_by_depth()
        return graph


# @graph_lru_cache
def try_construct(m, n):
    try:
        graph = construct_m_n_alt(m, n)
        if graph.is_solved():
            return graph
    except Exception:
        pass
    return None


def main():
    solved = json.load(open('./examples/construct_m_n_alt_output.json', 'r'))

    B = 64
    order = list(sorted(((m, n) for m in range(1,B+1) for n in range(1,B+1) if f'{m}-{n}' not in solved), key=lambda t: (t[0]+t[1], -t[0]*t[1])))
    failed = []

    def process_batch(n):
        batch = []
        while len(batch) < n:
            if len(order) == 0:
                break
            t = order.pop(0)
            if f'{t[0]}-{t[1]}' in solved:
                continue
            batch.append(t)
        print(batch)
        with Pool(
            processes=12,
        ) as pool:
            graphs = pool.starmap(try_construct, batch)
            for g in graphs:
                if g is None:
                    continue
                print(g.balancer_type)
                solved[g.balancer_type] = g.compact_string
                g.save_as_factorio_sat_network('./examples/construct_m_n_alt_networks')
            json.dump(solved, open('./examples/construct_m_n_alt_output.json', 'w'), indent=4)
            for m, n in batch:
                if f'{m}-{n}' not in batch:
                    failed.append(f'{m}-{n}')
        print(f'Remaining balancers: {len(order)}')
        print(f'Failed balancers: {len(failed)} -- {failed}')
        
    while len(order) > 0:
        process_batch(24)




if __name__ == '__main__':
    main()







