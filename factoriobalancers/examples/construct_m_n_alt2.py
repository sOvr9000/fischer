
from fischer.factoriobalancers import BeltGraph, is_balancer_defined



def construct_m_n_alt2(m: int, n: int) -> BeltGraph:
    graph = BeltGraph()
    '''
    Algorithm:

    If m is 1:
        If n is even:
            1. Construct the graph for 1-(n//2).
            2. Split each output into two outputs.
        else:
            1. Refer to action_stacks.txt and construct the partial graph for 1-n.
            2. Construct the necessary 1-k splitters to split the outputs into 1/n of the input.
    else:
        1. Construct the graphs for j-n and k-n where j=floor(m/2) and k=ceil(m/2).
        2. Substitute functionally equivalent groups of vertices from one graph to the other, in either order.
    
    Substitution of functionally equivalent groups of vertices:
    1. TODO
    '''




def main():
    pass



if __name__ == '__main__':
    main()


