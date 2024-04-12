
from fractions import Fraction
from math import log1p
from functools import lru_cache
import sympy as sp


@lru_cache(maxsize=4096)
def factorization_distance(n1, n2):
	n1 = abs(n1)
	n2 = abs(n2)
	if n1 == n2:
		return 0. # this only exists to save on computation
		# the formulas below still return 0 without this conditional
	if n1 == 0:
		v = factorization_distance(1, n2)
		return (1 + v) * (1 + v)
	if n2 == 0:
		v = factorization_distance(n1, 1)
		return (1 + v) * (1 + v)
	pfs1 = sp.factorint(n1)
	pfs2 = sp.factorint(n2)
	s = 0
	for pf in set(list(pfs1.keys()) + list(pfs2.keys())):
		if pf not in pfs1:
			pfs1[pf] = 0
		elif pf not in pfs2:
			pfs2[pf] = 0
		v = pfs1[pf] - pfs2[pf]
		s += log1p(v * v) / ((abs(pfs1[pf])+.5) * (abs(pfs2[pf])+.5))
		if (pfs1[pf]==0) ^ (pfs2[pf]==0):
			s += pf ** .5
	# v = sum(pfs1.keys()) / max(1, len(pfs1)) - sum(pfs2.keys()) / max(1, len(pfs2))
	# s += log1p(v * v)
	return s

def measure_accuracy(output_flow: dict[int, dict[int, any]], inputs: list[int], outputs: list[int]) -> tuple[float, float, float]:
	num_inputs = len(inputs)
	num_outputs = len(outputs)
	if num_inputs == 0 or num_outputs == 0:
		accuracy = 0
	else:
		accuracy = sum(
			sum(
				0.5 + 0.5 / Fraction(val).numerator
				if val != 0 and Fraction(val).denominator == num_outputs else
				0
				for val in v.values()
			)
			for k, v in output_flow.items()
		) / (num_inputs * num_outputs)
	s = 0
	for k in outputs:
		v = output_flow[k] if k in output_flow else {}
		for n in inputs:
			p = v[n] if n in v else 0
			p = Fraction(p)
			s += factorization_distance(p.denominator, num_outputs)
	error = s ** .5
	score = accuracy - 0.1 * error
	return accuracy, error, score


if __name__ == '__main__':
	for n1 in range(12):
		for n2 in range(n1, 12):
			print((n1, n2), factorization_distance(n1, n2))
			print((n2, n1), factorization_distance(n2, n1))
