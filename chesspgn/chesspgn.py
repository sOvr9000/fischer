
from typing import TextIO, Generator, Callable, Tuple, Union
from chess.pgn import read_game, read_headers, Game, Headers, skip_game



__all__ = ['load_pgn', 'count_pgns', 'get_time_control', 'is_valid_pgn_header', 'load_valid_pgn', 'load_pgns_for_player']



def load_pgn(file_handle:TextIO, load_condition:Callable[[Headers], bool] = None) -> Generator[Game, None, None]:
	'''
	Load PGNs in the given file.
	
	If the PGN index in the file is between `start` (inclusive) and `end` (exclusive), then automatically parse the PGN as a `Game` object (which contains `Game.board` for the `Board` object).  Otherwise, if `yield_headers` is true, then yield the headers of that PGN.

	Once `end` is reached, the loop closes and the rest of the file is ignored regardless of the value of `yield_headers`.

	If `yield_headers` is true, then two items are yielded at once.  Otherwise, one item is yielded.
	'''
	if load_condition is None:
		load_condition = lambda h: True
	while True:
		line = file_handle.tell()
		headers = read_headers(file_handle)
		if headers is None:
			break
		cur_line = file_handle.tell()
		if load_condition is not None:
			if load_condition(headers):
				file_handle.seek(line)
				game = read_game(file_handle)
				if game is None: # This check exists for when games are abandoned.  So there's record of games being abandoned, and they may have no mainline moves.
					break
				yield game
				file_handle.seek(cur_line)

def count_pgns(pgn_file:str) -> int:
	'''
	Return the number of PGNs saved in the given file.
	'''
	K = 0
	with open(pgn_file, 'r') as f:
		while skip_game(f):
			K += 1
	return K

def get_time_control(tc_str:str) -> Union[Tuple[int, int], None]:
	'''
	Return the time control of a game as a two-tuple in the format (initial_seconds, increment_seconds).
	'''
	if tc_str == '-':
		return None
	return tuple(map(int, tc_str.split('+')))

def is_valid_pgn_header(headers:Headers) -> bool:
	tc = headers['TimeControl']
	if tc == '-':
		return False
	# tc_init, _ = get_time_control(headers['TimeControl'])
	# if tc_init == 0:
	# 	return False
	term = headers['Termination']
	if term in ('Normal', 'Time forfeit'):
		return True
	if term not in ('Abandoned',):
		print(term, file=open('weird-terminations.txt', 'a'))
	return False

def load_valid_pgn(file_handle, max_yields=None, load_condition=is_valid_pgn_header):
	if max_yields is None:
		yield from load_pgn(file_handle, load_condition=load_condition)
		file_handle.seek(0)
		yield from load_valid_pgn(file_handle, max_yields=None) # loop over file infinitely
	else:
		for i, game in enumerate(load_pgn(file_handle, load_condition=load_condition)):
			yield game
			if i + 1 >= max_yields:
				break
		else:
			file_handle.seek(0)
			yield from load_valid_pgn(file_handle, max_yields=max_yields-i-1)

def load_pgns_for_player(file_handle, player: str, max_yields=None, load_condition=None):
	if load_condition is None:
		load_condition = lambda h: is_valid_pgn_header(h) and (h['White'] == player or h['Black'] == player)
	yield from load_valid_pgn(file_handle, max_yields=max_yields, load_condition=load_condition)


