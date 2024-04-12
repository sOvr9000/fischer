

from typing import List
from matplotlib import pyplot as plt


def plot_complex(z:List[complex], scatter=True, title=''):
	(plt.scatter if scatter else plt.plot)([z_.real for z_ in z], [z_.imag for z_ in z])
	plt.title(title)
	plt.legend(loc='best')
	plt.show()


# plot_complex([2+3j, 1-4j, 3+.5j], scatter=False)


