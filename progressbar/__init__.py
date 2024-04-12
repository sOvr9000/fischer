
from time import perf_counter_ns
from progress.bar import Bar

class StylizedProgressBar(Bar):
	bar_prefix = ' > '
	bar_suffix = ' < '
	fill = '\u25a0'
	empty_fill = '\u25a1'
	suffix = '%(index)5d / %(max)5d | average rate = %(avg_rate).1f/s | eta = %(real_eta).1fs'
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.start_time = perf_counter_ns()
	@property
	def real_elapsed(self): # in seconds
		return (perf_counter_ns() - self.start_time) * 1e-9
	@property
	def avg_rate(self):
		et = self.real_elapsed
		if et == 0:
			return 0.
		return self.index / et
	@property
	def real_eta(self):
		rate = self.avg_rate
		if rate == 0:
			return 0.
		return (self.max - self.index) / rate



