
from fractions import Fraction



def pretty_fraction(fraction):
	if isinstance(fraction, Fraction):
		return f'{fraction.numerator}/{fraction.denominator}'
	return str(fraction)



def sparse_solve(
	# parameters:list[int],
	# variables:list[int],
	transitives: list[int],
	relations: dict[int, dict[int, float]]
) -> dict[int, dict[int, float]]:
	'''
	Parameters are the free variables (representing inputs).
	Variables are the variables that are supposed to be defined in terms of the parameters (representing outputs).
	Transitives are the variables which get substituted and cancelled out (representing intermediate vertices).

	These lists guide the deduction of relations such that variables are represented in terms of parameters.
	`relations` is a dict that maps vertices (including outputs but excluding inputs) to a dict that represents the linear combination which represents the flow at that vertex.
	For example:
	```
	{
		0: {2: h, 1: h},
		1: {4: h, 6: h},
		2: {3: h, 5: h},
		3: {1: 1, 2: 1},
		7: {0: h},
		8: {0: h},
	}
	```
	This is an unsimplified 3-2 balancer.
	This function returns the dict which maps each output to the dict that represents the linear combination of the inputs, based on the resulting flow.
	```
	sparse_solve(
		[1, 3, 2, 0],
		{
			0: {2: h, 1: h},
			1: {4: h, 6: h},
			2: {3: h, 5: h},
			3: {1: 1, 2: 1},
			7: {0: h},
			8: {0: h},
		}
	)
	```
	Returns:
	```
	{
		7: {4: 0.5, 5: 0.5, 6: 0.5},
		8: {4: 0.5, 5: 0.5, 6: 0.5},
	}
	```
	'''
	while len(transitives) > 0:
		transitive = transitives.pop()
		sub_linear_combination = relations.pop(transitive) if transitive in relations else {}
		substitute(relations, transitive, sub_linear_combination)
	return relations

def construct_relations(graph):
	fractions = [None, 1, Fraction(1, 2)]
		# graph.inputs[:], \
		# graph.outputs[:], \
	return \
		[
			u
			for u in graph.vertices()
			if not graph.is_input(u) and not graph.is_output(u)
		], \
		{
			u: {
				v: 1 if graph.is_output(u) else fractions[graph.out_degree(u)]
				for v, _ in graph.in_edges(u)
			}
			for u in graph.vertices()
			if not graph.is_input(u) and (graph.out_degree(u) > 0 or (graph.is_output(u) and graph.in_degree(u) > 0))
		}


'''
(0) = 1/2 * (1) + 1/2 * (2)
(1) = 1/2 * (4) + 1/2 * (6)
(2) = 1/2 * (3) + 1/2 * (5)
(3) = 1/1 * (1) + 1/1 * (2)
(7) = 1/1 * (0)
(8) = 1/1 * (0)
Solve in terms of (4), (5), and (6).
Eliminate one non-output variable at a time.
Pick any non-output variable: (0)
Substitute where possible:
(1) = 1/2 * (4) + 1/2 * (6)
(2) = 1/2 * (3) + 1/2 * (5)
(3) = 1/1 * (1) + 1/1 * (2)
(7) = 1/1 * (1/2 * (1) + 1/2 * (2))
	= 1/2 * (1) + 1/2 * (2)
(8) = 1/1 * (1/2 * (1) + 1/2 * (2))
	= 1/2 * (1) + 1/2 * (2)
This is now a simpler version of the same problem.
(Recursion)
Pick any non-output variable: (1)
Substitute where possible:
(2) = 1/2 * (3) + 1/2 * (5)
(3) = 1/1 * (1/2 * (4) + 1/2 * (6)) + 1/1 * (2)
	= 1/2 * (4) + 1/2 * (6) + 1/1 * (2)
(7) = 1/2 * (1/2 * (4) + 1/2 * (6)) + 1/2 * (2)
	= 1/4 * (4) + 1/4 * (6) + 1/2 * (2)
(8) = 1/2 * (1/2 * (4) + 1/2 * (6)) + 1/2 * (2)
	= 1/4 * (4) + 1/4 * (6) + 1/2 * (2)
(Recursion)
NOTE: Make sure to simplify expressions such as that for (3) = ...
Pick any non-output variable: (2)
Substitute where possible:
(3) = 1/2 * (4) + 1/2 * (6) + 1/1 * (1/2 * (3) + 1/2 * (5))
	= 1/2 * (4) + 1/2 * (6) + 1/2 * (3) + 1/2 * (5)
	= 1/1 * (4) + 1/1 * (6) + 1/1 * (5)
(7) = 1/4 * (4) + 1/4 * (6) + 1/2 * (1/2 * (3) + 1/2 * (5))
	= 1/4 * (4) + 1/4 * (6) + 1/4 * (3) + 1/4 * (5)
(8) = 1/4 * (4) + 1/4 * (6) + 1/2 * (1/2 * (3) + 1/2 * (5))
	= 1/4 * (4) + 1/4 * (6) + 1/4 * (3) + 1/4 * (5)
(Recursion)
Pick any non-output variable: (3)
Substitute where possible:
(7) = 1/4 * (4) + 1/4 * (6) + 1/4 * (1/1 * (4) + 1/1 * (6) + 1/1 * (5)) + 1/4 * (5)
	= 1/2 * (4) + 1/2 * (6) + 1/2 * (5)
(8) = 1/4 * (4) + 1/4 * (6) + 1/4 * (1/1 * (4) + 1/1 * (6) + 1/1 * (5)) + 1/4 * (5)
	= 1/2 * (4) + 1/2 * (6) + 1/2 * (5)
This shows that the 3-2 balancer distributes 1/2 of each input to each of the outputs.
'''

def substitute(relations, transitive, sub_linear_combination):
	for vertex, linear_combination in relations.items():
		if transitive in linear_combination:
			scalar = linear_combination.pop(transitive)
			for sub_vertex, sub_scalar in sub_linear_combination.items():
				if sub_vertex in linear_combination:
					linear_combination[sub_vertex] += scalar * sub_scalar
				else:
					linear_combination[sub_vertex] = scalar * sub_scalar
			if vertex in linear_combination:
				flow = linear_combination.pop(vertex)
				if flow == 1:
					del relations[transitive]
					return
				scale = 1 / (1 - flow)
				for flow_vertex in linear_combination:
					linear_combination[flow_vertex] *= scale

def calculate_flow_points(transitives, relations):
	for transitive in transitives:
		sub_linear_combination = relations[transitive] if transitive in relations else {}
		substitute(relations, transitive, sub_linear_combination)
	return relations


def pretty_relations(relations):
	return '\n'.join(
		'{}:\t{}'.format(
			u,
			'\t'.join(
				f'{v}:{pretty_fraction(fraction)}'
				for v, fraction in lc.items()
			)
		)
		for u, lc in relations.items()
	)


if __name__ == '__main__':
	from beltgraph import BeltGraph

	# 3-2
	# graph = BeltGraph.from_blueprint_string('0eNqlluuOgjAQhV/FzG8wtFwl2X0RYzaojWmChbTFaAzvvoO4rllblxZCArTM16E9nM4VtnXHWsmFhvIKfNcIBeX6CoofRFUPbfrSMijhxKXusCUAUR2HhvGNMIY+AC727Awl6QOHSPoUSftNAExorjkbE7g9XL5Ed9wyiehHNDu3kikValkJ1TZSh1tWa6S3jcLwRgxDIzIkyzSAC97QZdoPmf1B0hekamuuNfYZYCOKmFGxe3bRP9klzsgHcUgygD2XbDd2JwZ+6swnVn5m4GczFiwyT0nuj7QQixnrZkGu/NctmjKvJPIe4IVPTXz3P43a+CbdEeotvGn5v/6LHXqMPMgGr/9JexziblxNp9tOg2mQxF+KFgshqbsdWayDZA6o96TcgRS9/bzCW1Z0kqxW3vxJdkmjGbIiz6riwiIqOmOPiy1bHPU2CxtxxlZnQybeSxdPcUzqv9XdMsbKhGt2xPDfWimAusJQbFvfy5twrGo+fsqizWIR4vG5sLxANwg5MalGJytIkq9onmZ4JkXffwMLMTKJ')
	# graph.delete_edge(0, 8)

	# 13-13
	# graph = BeltGraph.from_blueprint_string('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')

	# 11-11
	graph = BeltGraph.from_blueprint_string('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')

	# print(construct_relations(graph))
	rel = construct_relations(graph)
	print(pretty_relations(calculate_flow_points(*rel)))
	print(pretty_relations(sparse_solve(*rel)))


'''
DIFFERENCE BETWEEN CALCULATE_FLOW_POINTS AND SPARSE_SOLVE:
calculate_flow_points fully reveals the flow of the graph but is O(n^2)
	- Useful for detecting critical points where flow is too concentrated/dense
sparse_solve reveals the flow of the graph only at its outputs but is O(nlogn)
	- Useful for quickly testing the effectivity of a balancer graph
'''
