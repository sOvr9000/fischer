

from fischer.classicgames.sudoku.su import puzzle_str
from fischer.classicgames.sudoku.sudoku_io import read_puzzles



puzzles, solutions = read_puzzles('puzzles.sudk')
print()
print(len(puzzles))
print()
print(puzzle_str(puzzles[-1]))
print()
print(puzzle_str(solutions[-1]))
print()



