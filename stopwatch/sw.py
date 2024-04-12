
from time import perf_counter_ns as ns
from typing import Union
from enum import Enum, auto
from fischer.dt import dt



class TimeScale(Enum):
	Nanoseconds = auto()
	Microseconds = auto()
	Milliseconds = auto()
	Seconds = auto()
	Minutes = auto()
	Hours = auto()
	Days = auto()
	Weeks = auto()
	Months = auto()
	Years = auto()

TIME_SCALE_SYMBOLS = {
	TimeScale.Nanoseconds: 'ns',
	TimeScale.Microseconds: 'us',
	TimeScale.Milliseconds: 'ms',
	TimeScale.Seconds: 's',
	TimeScale.Minutes: 'm',
	TimeScale.Hours: 'h',
	TimeScale.Days: 'd',
	TimeScale.Weeks: 'w',
	TimeScale.Months: 'M',
	TimeScale.Years: 'y',
}


class Stopwatch:
	def __init__(self, time_scale: Union[TimeScale, str] = TimeScale.Milliseconds):
		'''
		time_scale must be a TimeScale enum.
		'''
		if isinstance(time_scale, str):
			time_scale = getattr(TimeScale, time_scale[0].upper() + time_scale[1:].lower())
		self.time_scale = time_scale
		self.start()
	@property
	def elapsed_time(self):
		return self.convert_time(ns() - self.start_time)
	def convert_time(self, t, scale: TimeScale = None):
		if scale is None:
			scale = self.time_scale
		if scale == TimeScale.Nanoseconds:
			return t
		elif scale == TimeScale.Microseconds:
			return t * 1e-3
		elif scale == TimeScale.Milliseconds:
			return t * 1e-6
		elif scale == TimeScale.Seconds:
			return t * 1e-9
		elif scale == TimeScale.Minutes:
			return t * 1e-9 * 0.01666666666666666666666666666667
		elif scale == TimeScale.Hours:
			return t * 1e-9 * 2.7777777777777777777777777777778e-4
		elif scale == TimeScale.Days:
			return t * 1e-9 * 1.1574074074074074074074074074074e-5
		elif scale == TimeScale.Weeks:
			return t * 1e-9 * 1.1574074074074074074074074074074e-5 * 0.14285714285714285714285714285714
		elif scale == TimeScale.Months:
			return t * 1e-9 * 1.1574074074074074074074074074074e-5 * 0.03333333333333333333333333333333
		elif scale == TimeScale.Years:
			return t * 1e-9 * 1.1574074074074074074074074074074e-5 * 0.00273972602739726027397260273973
		raise Exception(f'Unrecognized time scale: {scale}')
	def start(self):
		self.start_time = ns()
		self.laps = [self.start_time]
	def lap(self, log: bool = False):
		'''
		Set a lap time.  Return the time since the last lap.
		'''
		self.laps.append(ns())
		c = self.convert_time(self.laps[-1] - self.laps[-2])
		if log:
			print(f'[STOPWATCH] [{dt()}] {c} {TIME_SCALE_SYMBOLS[self.time_scale]}')
		return
	def try_lap(self, min_lap_time: float, log: bool = False) -> bool:
		'''
		Set a lap time only if the lap time would be greater than or equal to `min_lap_time`.  Return whether the lap was successfully recorded.  If returned True, then the actual lap time (which is greater than or equal to `min_lap_time`) is saved at `Stopwatch.laps[-1]`.
		
		This is useful for triggering periodic events that do not need to be triggered at identical intervals, such as writing data.
		'''
		if self.convert_time(ns() - (self.start_time if len(self.laps) == 0 else self.laps[-1])) >= min_lap_time:
			self.lap(log=log)
			return True
		return False
