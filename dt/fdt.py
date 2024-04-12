from datetime import datetime as _dt

def dt():
	return _dt.strftime(_dt.now(), '%Y%m%d%H%M%S')
