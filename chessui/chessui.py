
from pygenv import GridEnv
import chess
import chess.engine
import importlib.resources as pkg_resources

from . import assets

sprite_sheet_path = 'ss.png'


class ChessEnv(GridEnv):
	PIECE_INDEX_MAP = {
		chess.WHITE: {
			chess.PAWN: (3, 1),
			chess.KNIGHT: (3, 0),
			chess.BISHOP: (2, 2),
			chess.ROOK: (3, 3),
			chess.QUEEN: (3, 2),
			chess.KING: (2, 3),
		},
		chess.BLACK: {
			chess.PAWN: (0, 3),
			chess.KNIGHT: (0, 2),
			chess.BISHOP: (0, 0),
			chess.ROOK: (1, 1),
			chess.QUEEN: (1, 0),
			chess.KING: (0, 1),
		},
	}
	def __init__(self, engine_fpath = None):
		'''
		engine_fpath may be a string, but it can also be a list of two strings so that white has a separate engine from black.  If None, then no engine is loaded or able to be used.
		'''
		super().__init__((640, 640))
		self.set_dimensions(8, 8)
		self.set_pannable(False)
		self.set_scrollable(False)
		self.set_scale(80)
		self.center_camera()
		self.load_sprite_sheet(pkg_resources.open_binary(assets, sprite_sheet_path), 4, 4, 'chess')
		self.game = chess.Board()
		self.drag_piece = None
		self.drag_start_square = None
		self.popped_moves = []
		self.human_input = True
		self.is_game_over = self.game.is_game_over()
		if engine_fpath is not None:
			if isinstance(engine_fpath, list):
				self.white_engine = chess.engine.SimpleEngine.popen_uci(engine_fpath[0])
				self.black_engine = chess.engine.SimpleEngine.popen_uci(engine_fpath[1])
			else:
				self.white_engine = chess.engine.SimpleEngine.popen_uci(engine_fpath)
				self.black_engine = self.white_engine
		else:
			self.white_engine = None
			self.black_engine = None
		# self.add_ui_element('termination', font_size=72, text_color=(128, 0, 0), x=0, y=self.HALF_HEIGHT-36, w=200, h=40, text='')
		element = self.add_ui_element('termination',
			self.gui.elements.UILabel(
				self.pyg.Rect(0, self.HALF_HEIGHT-36, self.WIDTH, 72),
				'',
				self.ui_manager,
			),
			text_colour=(255, 0, 0),
			# font=self.get_font('arial', 72),
		)
	def set_human_input(self, flag):
		self.human_input = flag
	def engine_eval(self, time=1.0):
		engine = self.white_engine if self.game.turn == chess.WHITE else self.black_engine
		if engine is None:
			return None
		return engine.play(self.game, chess.engine.Limit(time=time))
	def engine_analysis(self, time=2.0):
		engine = self.white_engine if self.game.turn == chess.WHITE else self.black_engine
		if engine is None:
			return None
		return engine.analyse(self.game, chess.engine.Limit(time=time))
	def on_rendered_tile(self, x, y):
		self.draw_grid_sprite('chess', x, y, 2, (x+y)%2)
	def render(self):
		for square, piece in self.game.piece_map().items():
			if square != self.drag_start_square:
				self.draw_grid_sprite('chess', chess.square_file(square), chess.square_rank(square), *self.PIECE_INDEX_MAP[piece.color][piece.piece_type])
		if self.drag_piece is not None:
			# show available move squares
			for move in self.game.legal_moves:
				if move.from_square == self.drag_start_square:
					self.draw_grid_sprite('chess', chess.square_file(move.to_square), chess.square_rank(move.to_square), 1, 2)
			# hold piece in hand
			self.draw_grid_sprite('chess', self.mouse_grid_pos_x-self.left_mouse_drag_start_grid_x_mod, self.mouse_grid_pos_y-self.left_mouse_drag_start_grid_y_mod, *self.PIECE_INDEX_MAP[self.drag_piece.color][self.drag_piece.piece_type])
		if self.is_game_over:
			# self.set_label_absolute_position('termination', , self.HALF_HEIGHT-36)
			self.get_ui_element('termination').set_text(self.termination.upper())
		else:
			self.get_ui_element('termination').set_text('')
	def left_mouse_button_held(self):
		if not self.human_input:
			return
		if self.is_game_over:
			return
		self.drag_start_square = chess.square(self.left_mouse_drag_start_snapped_grid_x, self.left_mouse_drag_start_snapped_grid_y)
		self.drag_piece = self.game.piece_at(self.drag_start_square)
		if self.drag_piece is None:
			return
	def left_mouse_button_released(self):
		if not self.human_input:
			return
		if self.is_game_over:
			return
		promotion = chess.QUEEN if self.drag_piece is not None and self.drag_piece.piece_type == chess.PAWN and (self.snapped_mouse_grid_pos_y == 0 or self.snapped_mouse_grid_pos_y == 7) else None
		move = chess.Move(self.drag_start_square, chess.square(self.snapped_mouse_grid_pos_x, self.snapped_mouse_grid_pos_y), promotion=promotion)
		if self.game.is_legal(move):
			print(self.game.san(move))
			self.push(move)
		self.drag_piece = None
		self.drag_start_square = None
	def on_mouse_wheel(self, v):
		if not self.human_input:
			return
		if v < 0:
			if len(self.game.move_stack) > 0:
				self.pop()
		else:
			if len(self.popped_moves) > 0:
				self.push(self.popped_moves.pop(), clear_popped_moves=False)
	def key_held(self, key):
		if key == self.KEYCODES.K_v and not self.is_game_over and self.human_input:
			moves = list(self.game.legal_moves)
			if len(moves) > 0:
				self.push(self.random_choice(moves))
	def key_pressed(self, key):
		if key == self.KEYCODES.K_r and self.human_input:
			self.reset()
		if key == self.KEYCODES.K_g and self.human_input:
			self.cur_engine_eval = self.engine_eval(1.0)
			if self.cur_engine_eval is not None:
				self.push(self.cur_engine_eval.move)
	def push(self, move, clear_popped_moves = True):
		self.game.push(move)
		if clear_popped_moves:
			self.popped_moves.clear()
		self.is_game_over = self.game.is_game_over(claim_draw=True)
		if self.is_game_over:
			self.is_checkmate = self.game.is_checkmate()
			self.is_fifty_moves = self.game.is_fifty_moves()
			self.is_stalemate = self.game.is_stalemate()
			self.is_insufficient_material = self.game.is_insufficient_material()
			self.is_repetition = self.game.is_repetition()
			self.is_draw = self.is_fifty_moves or self.is_stalemate or self.is_insufficient_material or self.is_repetition
			self.termination = 'checkmate' if self.is_checkmate else 'fifty moves' if self.is_fifty_moves else 'stalemate' if self.is_stalemate else 'insufficient material' if self.is_insufficient_material else 'repetition' if self.is_repetition else 'unknown'
	def pop(self):
		self.popped_moves.append(self.game.pop())
		self.reset_flags()
	def reset_flags(self):
		self.is_game_over = False
		self.is_draw = False
		self.is_fifty_moves = False
		self.is_checkmate = False
		self.is_stalemate = False
		self.is_insufficient_material = False
		self.is_repetition = False
	def reset(self, board: chess.Board):
		self.game = board
		self.popped_moves.clear()
		self.reset_flags()





if __name__ == '__main__':
	env = ChessEnv(engine_fpath='C:/Chess Engines/SF15/stockfish_15_x64_avx2.exe')
	# env.set_human_input(False)
	env.run_loop()

