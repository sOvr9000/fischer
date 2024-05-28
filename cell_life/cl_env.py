


from fischer.pygenv import GridEnv
from .cl import CellLife



class CellLifeEnv(GridEnv):
    def __init__(self, world: CellLife):
        self.world = world
        super().__init__(screen_size=(self.world.size[0]*16, self.world.size[1]*16))
        self.set_bg_color((32, 32, 32))
        self.set_default_tile_color((0, 0, 0))
        self.set_dimensions(w=self.world.size[1], h=self.world.size[0])
        self.set_scale(16)
        self.center_camera()
        self.set_pannable(False)
        self.set_zoomable(False)
        self.set_fps(10)
    def render(self):
        for cell in self.world.cells:
            e = (cell.energy + 256) // 2
            self.draw_grid_rect((e, e, e), cell.x, cell.y, 1, 1)
    def update(self):
        self.world.update()


