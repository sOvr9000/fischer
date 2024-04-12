import unittest
import numpy as np
from fischer.numbergames.link_m_diff_n import LinkMDiffN

class TestLinkMDiffN(unittest.TestCase):
    def setUp(self):
        self.game = LinkMDiffN()

    def test_init(self):
        self.assertEqual(self.game.size, (6, 6))
        self.assertEqual(self.game.chain_length, 5)
        self.assertEqual(self.game.max_diff, 2)
        self.assertEqual(self.game.grid.shape, (6, 6))
        self.assertEqual(self.game.links.shape, (6, 6, 2))

    def test_randomize_grid(self):
        self.game.randomize_grid(10)
        self.assertTrue((self.game.grid >= 1).all() and (self.game.grid <= 10).all())
        self.assertFalse(self.game.links.any())

    def test_get_adjacent_cell(self):
        self.assertEqual(self.game.get_adjacent_cell(2, 2, 0), (3, 2))
        self.assertEqual(self.game.get_adjacent_cell(2, 2, 1), (2, 3))
        self.assertEqual(self.game.get_adjacent_cell(2, 2, 2), (1, 2))
        self.assertEqual(self.game.get_adjacent_cell(2, 2, 3), (2, 1))

    def test_is_index_valid(self):
        self.assertTrue(self.game.is_index_valid(0, 0))
        self.assertTrue(self.game.is_index_valid(5, 5))
        self.assertFalse(self.game.is_index_valid(-1, 0))
        self.assertFalse(self.game.is_index_valid(0, 6))

    def test_get_adjacent_cells(self):
        adjacent_cells = list(self.game.get_adjacent_cells(2, 2))
        self.assertIn((3, 2, 0), adjacent_cells)
        self.assertIn((2, 3, 1), adjacent_cells)
        self.assertIn((1, 2, 2), adjacent_cells)
        self.assertIn((2, 1, 3), adjacent_cells)

    def test_get_direction(self):
        self.assertEqual(self.game.get_direction(2, 2, 3, 2), 0)
        self.assertEqual(self.game.get_direction(2, 2, 2, 3), 1)
        self.assertEqual(self.game.get_direction(2, 2, 1, 2), 2)
        self.assertEqual(self.game.get_direction(2, 2, 2, 1), 3)
        self.assertEqual(self.game.get_direction(2, 2, 3, 3), -1)

    def test_get_total_links_to_cell(self):
        self.game.links[2, 2, 0] = True
        self.game.links[2, 2, 1] = True
        self.game.links[1, 2, 0] = True
        self.game.links[2, 1, 1] = True
        self.assertEqual(self.game.get_total_links_to_cell(2, 2), 4)

    def test_cells_are_linked(self):
        self.game.links[2, 2, 0] = True
        self.assertTrue(self.game.cells_are_linked(2, 2, 3, 2))
        self.assertFalse(self.game.cells_are_linked(2, 2, 2, 3))

if __name__ == '__main__':
    unittest.main()