

from functools import lru_cache
from math import acos, asin, atan2, cos, floor, ceil, sin, tan
import random
from typing import Any, Hashable, List, Tuple, Union, Callable
import pygame as pyg
# from pygame.locals import *
import pygame.locals as KEYCODES
import pygame_gui as gui
import pymunk as pym
import numpy as np
import cv2
# from noise.perlin import SimplexNoise
from colorsys import hsv_to_rgb, rgb_to_hsv
from time import time_ns
from enum import IntEnum, auto as enum_auto
from fischer.colorpalettes import scale_value, scale_saturation, get_palette
from fischer.stopwatch import Stopwatch
from fischer.weightedchoice import WeightedChoice
from fischer.dt import dt as current_date_time
from perlin_noise import PerlinNoise



class RectDrawMode(IntEnum):
    Corner = enum_auto()
    Center = enum_auto()

class SpriteDrawMode(IntEnum):
    Corner = enum_auto()
    Center = enum_auto()


# class KEYCODE:
# 	a = 97
# 	b = 98
# 	c = 99
# 	d = 100
# 	e = 101
# 	f = 102
# 	g = 103
# 	h = 104
# 	i = 105
# 	j = 106
# 	k = 107
# 	l = 108
# 	m = 109
# 	n = 110
# 	o = 111
# 	p = 112
# 	q = 113
# 	r = 114
# 	s = 115
# 	t = 116
# 	u = 117
# 	v = 118
# 	w = 119
# 	x = 120
# 	y = 121
# 	z = 122
# 	enter = 13
# 	space = 32
# 	right_shift = 303
# 	left_shift = 304
# 	right_control = 305
# 	left_control = 306
# 	backspace = 8
# 	delete = 127
# 	numpad_enter = 271
# 	F1 = 282
# 	F2 = 283
# 	F3 = 284
# 	F4 = 285
# 	F5 = 286
# 	F6 = 287
# 	F7 = 288
# 	F8 = 289
# 	F9 = 290
# 	F10 = 291
# 	F11 = 292
# 	F12 = 293
# 	key_0 = 48
# 	key_1 = 49
# 	key_2 = 50
# 	key_3 = 51
# 	key_4 = 52
# 	key_5 = 53
# 	key_6 = 54
# 	key_7 = 55
# 	key_8 = 56
# 	key_9 = 57
# 	numpad_0 = 256
# 	numpad_1 = 257
# 	numpad_2 = 258
# 	numpad_3 = 259
# 	numpad_4 = 260
# 	numpad_5 = 261
# 	numpad_6 = 262
# 	numpad_7 = 263
# 	numpad_8 = 264
# 	numpad_9 = 265
# 	slash = 47
# 	backslash = 92
# 	equals = 61
# 	grave = 96
# 	tab = 9
# 	caps_lock = 301 # MAYBE NOT A GOOD IDEA
# 	command = 311
# 	windows = 311
# 	right_alt = 307
# 	left_alt = 308
# 	right_bracket = 93
# 	left_bracket = 91
# 	hyphen = 45
# 	minus = 45
# 	dash = 45
# 	up_arrow = 273
# 	down_arrow = 274
# 	right_arrow = 275
# 	left_arrow = 276
# 	escape = 27
# 	home = 278
# 	end = 279
# 	page_up = 280
# 	page_down = 281
# 	dot = 46
# 	period = 46
# 	point = 46
# 	comma = 44
# 	apostrophe = 39
# 	single_quote = 39
# 	quote = 39
# 	semicolon = 59
# 	numpad_dot = 266
# 	numpad_point = 266
# 	numpad_divide = 267
# 	numpad_times = 268
# 	numpad_minus = 269
# 	numpad_plus = 270
# 	insert = 277
# 	pause_break = 19




def get_control_mode_keycodes(mode = 'arrows'):
    if mode is None:
        return -1, -1, -1, -1
    elif mode == 'wasd':
        return KEYCODES.K_d, KEYCODES.K_a, KEYCODES.K_w, KEYCODES.K_s
    elif mode == 'ijkl':
        return KEYCODES.K_l, KEYCODES.K_j, KEYCODES.K_i, KEYCODES.K_k
    elif mode == 'arrows':
        return KEYCODES.K_RIGHT, KEYCODES.K_LEFT, KEYCODES.K_UP, KEYCODES.K_DOWN
    elif mode == 'numpad_arrows':
        return KEYCODES.K_6, KEYCODES.K_4, KEYCODES.K_8, KEYCODES.K_2
    else:
        return get_control_mode_keycodes()




def load_font(name = 'arial', size = 16):
    print('Loading font "{}" with size {}'.format(name, size))
    f = pyg.font.SysFont(name, size)
    print('Loaded font.')
    return f


# class UIElement:
# 	def __init__(self, env, x = 0, y = 0, w = 200, h = 40, anchor_left = True, anchor_top = True, screen_reference_left = True, screen_reference_top = True, parent = None, children = []):
# 		self.env = env
# 		self.set_size(w, h)
# 		self.set_anchor(anchor_left, anchor_top)
# 		self.set_screen_reference(screen_reference_left, screen_reference_top)
# 		self.set_relative_position(x, y)
# 		self.set_parent(parent)
# 		self.children = []
# 		self.add_children(children)
# 		self.disable_bg_rect()

# 	def set_parent(self, parent):
# 		self.parent = parent
# 		self.update_children()
    
# 	def add_children(self, children):
# 		for c in children:
# 			self.children.append(c)
# 		self.update_children()
    
# 	def add_child(self, child):
# 		self.children.append(child)
# 		self.update_children()
    
# 	def remove_children(self):
# 		self.children.clear()
# 		self.update_children()
    
# 	def remove_child(self, child):
# 		if child in self.children:
# 			self.children.remove(child)
# 			self.update_children()

# 	def set_anchor(self, left, top):
# 		self.anchor_left = left
# 		self.anchor_top = top
    
# 	def set_screen_reference(self, left, top):
# 		self.screen_reference_left = left
# 		self.screen_reference_top = top

# 	def set_relative_position(self, x, y):
# 		if not self.screen_reference_left:
# 			x = self.env.WIDTH - x
# 		if not self.anchor_left:
# 			x -= self.w
# 		if not self.screen_reference_top:
# 			y = self.env.HEIGHT - y
# 		if not self.anchor_top:
# 			y -= self.h
# 		self.x = x
# 		self.y = y

# 	def set_absolute_position(self, x, y):
# 		self.x = x
# 		self.y = y
    
# 	def set_size(self, w, h):
# 		self.w = w
# 		self.h = h
    
# 	def contains_point(self, x, y):
# 		return x >= self.x and y >= self.y and x < self.x + self.w and y < self.y + self.h
    
# 	def update_children(self):
# 		pass

# 	def set_bg_rect(self, c = (0, 0, 0), x = 0, y = 0, w = 0, h = 0, width = 0, alpha = 255):
# 		self.bg_rect = (c, x, y, w, h, width, alpha)

# 	def disable_bg_rect(self):
# 		self.bg_rect = None



# class GridLayout(UIElement): # auto-sizes and auto-positions children to fit in rectangle
# 	def __init__(self, env, rows = 1, cols = 1, **kwargs):
# 		super().__init__(env, **kwargs)
# 		self.rows = rows
# 		self.cols = cols
    
# 	def update_children(self):
# 		pass

# class Stack(UIElement): # does not auto-size children, but auto-positions children based on their current size (along given dimension)
# 	pass

# class Label(UIElement):
# 	def __init__(self, env, font, initial_text = 'New Label', text_color = (0, 0, 0), **kwargs):
# 		super().__init__(env, **kwargs)
# 		self.font = font
# 		self.text_color = text_color
# 		self.text = None
# 		self.set_text(initial_text)

# 	def draw(self):
# 		if self.bg_rect is not None:
# 			c, x, y, w, h, width, alpha = self.bg_rect
# 			self.env.draw_screen_rect(c, self.x + x, self.y + y, self.w + w, self.h + h, width = width, alpha = alpha)
# 		self.env.screen.blit(self.rendered_text, (self.x, self.y))

# 	def set_text(self, text, text_color=None):
# 		ot = self.text
# 		self.text = text
# 		if text_color is not None:
# 			self.text_color = text_color
# 		if ot != self.text:
# 			self.render()

# 	def set_text_color(self, text_color):
# 		oc = self.text_color
# 		self.text_color = text_color
# 		if oc != self.text_color:
# 			self.render()

# 	def render(self):
# 		self.rendered_text = self.font.render(self.text, True, self.text_color)




class PygEnv:
    PI = 3.1415926535897932384626433832795
    TAU = 6.283185307179586476925286766559
    HALF_PI = 1.5707963267948966192313216916398
    QUARTER_PI = 0.78539816339744830961566084581988
    THIRD_PI = 1.0471975511965977461542144610932
    TWO_THIRDS_PI = 2.0943951023931954923084289221863
    THREE_QUARTERS_PI = 2.3561944901923449288469825374596
    FIFTH_PI = 0.6283185307179586476925286766559
    TWO_FIFTHS_PI = 1.2566370614359172953850573533118
    THREE_FIFTHS_PI = 1.8849555921538759430775860299677
    FOUR_FIFTHS_PI = 2.5132741228718345907701147066236

    PHI = 1.6180339887498948482045868343656
    EULER = 2.7182818284590452353602874713527
    SQRT2 = 1.4142135623730950488016887242097
    SQRT2_HALF = 0.70710678118654752440084436210485
    SQRT3 = 1.7320508075688772935274463415059
    SQRT3_HALF = 0.86602540378443864676372317075294
    SQRT3_QUARTER = 0.43301270189221932338186158537647
    RAD_2_DEG = 57.295779513082320876798154814105
    DEG_2_RAD = 0.01745329251994329576923690768489

    BODY_TYPE_DYNAMIC = pym.Body.DYNAMIC
    BODY_TYPE_KINEMATIC = pym.Body.KINEMATIC
    BODY_TYPE_STATIC = pym.Body.STATIC
    BODY_SHAPE_CIRCLE = 0
    BODY_SHAPE_RECT = 1
    BODY_SHAPE_POLY = 2
    BODY_SHAPE_SEGMENT = 3

    ADJACENCY_OFFSETS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    NEIGHBORHOOD_OFFSETS = ADJACENCY_OFFSETS + [(1, 1), (-1, 1), (-1, -1), (1, -1)]

    def __init__(self, screen_size=(640, 640)):
        pyg.init()
        if screen_size == 'fullscreen':
            self.screen = pyg.display.set_mode((0, 0), pyg.FULLSCREEN)
            self.WIDTH, self.HEIGHT = self.screen.get_size()
            self.HALF_WIDTH = self.WIDTH * 0.5
            self.HALF_HEIGHT = self.HEIGHT * 0.5
        else:
            self.WIDTH, self.HEIGHT = screen_size
            self.HALF_WIDTH = self.WIDTH * 0.5
            self.HALF_HEIGHT = self.HEIGHT * 0.5
            self.screen = pyg.display.set_mode(screen_size)
        self.KEYCODES = KEYCODES
        self.keys_held = []
        self.stopwatch = Stopwatch()
        self.bg_color = (0, 0, 0)
        self.set_frame_rate(60)
        self.clock = pyg.time.Clock()
        self.current_time_ns = time_ns()
        self._lmb_is_pressed = False
        self._mmb_is_pressed = False
        self._rmb_is_pressed = False
        self.world_zoom = 1
        self.zoom_sensitivity = 1.1
        self.set_zoom_bounds(1, 256)
        self.set_world_zoom(1)
        self.set_camera_pos(0, 0)
        self.run = True
        self.set_panning_limits()
        self.set_pan_controls('wasd')
        self.set_screen_pan_speed(4, 4)
        self.set_quit_key(KEYCODES.K_ESCAPE)
        self.set_frame_rate_controls('arrows')
        self.noises: dict[str, tuple[float, float, float, float, float, float]] = {}
        self.frame = -1
        self.mouse_pos_x = None
        self.mouse_pos_y = None
        self.mouse_world_pos_x = None
        self.mouse_world_pos_y = None
        self.left_mouse_drag_start_x = None
        self.left_mouse_drag_start_y = None
        self.middle_mouse_drag_start_x = None
        self.middle_mouse_drag_start_y = None
        self.right_mouse_drag_start_x = None
        self.right_mouse_drag_start_y = None
        self.target_camera_x = None
        self.target_camera_y = None
        self.target_follow_speed = 0.05
        self.unset_camera_target_on_pan = True
        self.pixel_data_history = None
        self.pixel_data_history_downscaling = 1.0
        self.downscaled_size = None
        self.bg_sprite = None
        self.do_draw_background = True
        self.pixel_data_is_grayscale = False
        self.pyg = pyg
        self.pym = pym
        self.pymunk_bodies = {}
        self.pymunk_shapes = {}
        self.collision_handlers: dict[tuple[int, int], pym.CollisionHandler] = {}
        self.loaded_fonts = {}
        # self.labels = {}
        self.set_rendering(True)
        self.sprite_sheets: dict[str, SpriteSheet] = {}
        self.palettes = {}
        self.current_gif_images = None
        self.cur_pixel_data = np.zeros((self.HEIGHT, self.WIDTH, 3))
        self.weighted_choices = {}
        self.using_physics = False
        self.gui = gui
        self.ui_manager = gui.ui_manager.UIManager((self.WIDTH, self.HEIGHT))
        self.ui_elements = {}
        self.rect_draw_mode = RectDrawMode.Center
        self.sprite_draw_mode = SpriteDrawMode.Center
        self.offsets_l1 = {}
        self.offsets_l2 = {}

    @staticmethod
    def sqrt(x:float) -> float:
        return x**0.5

    @staticmethod
    def sq(x:float) -> float:
        return x*x

    @staticmethod
    def lerp(a:float, b:float, t:float) -> float:
        return a+(b-a)*t

    @staticmethod
    def point_on_circle(angle:float, radius:float=1, cx:float=0, cy:float=0) -> Tuple[float, float]:
        '''
        Same as `PygEnv.polar_point()`, but `radius` may be unprovided and a default value would be assumed, and also a center of the circle can be given so as to offset the calculated point.
        '''
        x, y = PygEnv.polar_point(radius, angle)
        return cx + x, cy + y

    @staticmethod
    def snapped_to_grid(x:float, y:float, unit_x:float=1, unit_y:float=1, offset_x:float=0, offset_y:float=0) -> Tuple[float, float]:
        return floor(x / unit_x) * unit_x + offset_x, floor(y / unit_y) * unit_y + offset_y

    @staticmethod
    def random(a:float=0, b:float=1) -> float:
        return np.random.random()*(b-a)+a

    @staticmethod
    def randint(a:int, b:int) -> int:
        return np.random.randint(a,b)
    
    @staticmethod
    def random_normal(mu:float, sigma:float) -> float:
        return np.random.normal(mu, sigma)
    
    @staticmethod
    def random_choice(l:Union[list, tuple], p:list[float]=None) -> Any:
        if p is not None:
            return l[np.random.choice(np.arange(len(l)), p=p)]
        return l[np.random.randint(0, len(l))]
    
    @staticmethod
    def random_angle() -> float:
        return np.random.random() * PygEnv.TAU
    
    @staticmethod
    def random_rect_point(min_x:float=0, min_y:float=0, max_x:float=1, max_y:float=1) -> Tuple[float, float]:
        '''
        Return a random point chosen with uniform distribution inside the rectangle defined by corner points `{(min_x, min_y), (max_x, max_y)}`.
        '''
        return PygEnv.random(min_x, max_x), PygEnv.random(min_y, max_y)

    @staticmethod
    def random_circle_point(x:float, y:float, r:float) -> Tuple[float, float]:
        '''
        Return a random point chosen with uniform distribution inside the circle with center `(x, y)` and radius `r`.
        '''
        return PygEnv.point_on_circle(PygEnv.random_angle(), np.random.random() ** .5 * r, cx=x, cy=y)

    @staticmethod
    def hsv_to_rgb(h:float, s:float, v:float) -> Tuple[int, int, int]:
        '''
        h, s, and v must be floats 0-1

        Return r, g, and b as integers 0-255
        '''
        r, g, b = hsv_to_rgb(h, s, v)
        return int(r * 255), int(g * 255), int(b * 255)

    @staticmethod
    def rgb_to_hsv(r:int, g:int, b:int) -> Tuple[float, float, float]:
        '''
        r, g, and b can be integers 0-255, but can also be floats 0-255.

        Return h, s, and v as floats 0-1
        '''
        return rgb_to_hsv(r*0.0039215686274509803921568627451, g*0.0039215686274509803921568627451, b*0.0039215686274509803921568627451)
    
    @staticmethod
    def color_modify(c:Tuple[int, int, int], hue:float=0., saturation:float=0., value:float=0.) -> Tuple[int, int, int]:
        '''
        hue is added to the hue of c.

        If saturation < 0, then scale the saturation of c by the percentage (-saturation) toward zero.
        If saturation > 0, then scale the saturation of c by the percentage (saturation) toward one.
        If saturation = 0, then the saturation of c is left unchanged.

        If value < 0, then scale the value of c by the percentage (-value) toward zero.
        If value > 0, then scale the value of c by the percentage (value) toward one.
        If value = 0, then the value of c is left unchanged.
        '''
        h, s, v = PygEnv.rgb_to_hsv(*c)
        target_saturation = int(saturation >= 0)
        target_value = int(value >= 0)
        return PygEnv.hsv_to_rgb(
            h + hue,
            PygEnv.lerp(s, target_saturation, saturation * (2*target_saturation-1)),
            PygEnv.lerp(v, target_value, value * (2*target_value-1))
        )

    @staticmethod
    def color_distort(c:Tuple[int, int, int], hue:float=0., saturation:float=0., value:float=.1) -> Tuple[int, int, int]:
        '''
        Call color_modify() with randomized parameters based on hue, saturation, and value.
        '''
        return PygEnv.color_modify(
            c,
            PygEnv.random(-hue, hue),
            PygEnv.random(-saturation, saturation),
            PygEnv.random(-value, value)
        )

    @staticmethod
    def color_inverse(c: Tuple[int, int, int]) -> Tuple[int, int, int]:
        '''
        Return the additive inverse of the given color `c`, which is assumed to be in RGB format with integers between 0 and 255.
        '''
        r, g, b = c
        return 255-r, 255-g, 255-b

    @staticmethod
    def point_cluster(n:int, density:int=8, seed:Hashable=None) -> List[Tuple[float, float]]:
        '''
        Generate `n` 2D points where each point is at least 1 unit away from other points by Euclidean distance.

        `density` is the number of bisection iterations to determine how closely the points will be packed together while retaining the minimum distance constraint.
        '''
        if seed is not None:
            rng = random.Random()
            rng.seed(hash(seed))
            rand = rng.random
        else:
            rand = PygEnv.random
        r = rand()
        t = rand() * PygEnv.TAU
        p = [(r * cos(t), r * sin(t))]
        for _ in range(n):
            mx, my = np.mean(p, axis=0)
            r, t = n, atan2(-my, -mx) + rand()*.125-.0625
            cs, sn = cos(t), sin(t)
            a, b = 0, r
            for _ in range(density):
                npx, npy = r * cs, r * sn
                if any((npx - px) * (npx - px) + (npy - py) * (npy - py) < 1 for px, py in p):
                    a = r
                    r = (r + b) * .5
                else:
                    b = r
                    r = (r + a) * .5
            p.append((npx, npy))
        return p

    @staticmethod
    def polar_point(r:float, theta:float) -> Tuple[float, float]:
        '''
        Calculate the point (x, y) which lies on a circle (centered at the origin) of radius `r` at the angle `theta`.
        '''
        return r * cos(theta), r * sin(theta)
    
    @staticmethod
    def rotated_point(x:float, y:float, theta:float, pivot_x:float = 0, pivot_y:float = 0) -> Tuple[float, float]:
        '''
        Return the point `(x, y)` rotated about the point `(pivot_x, pivot_y)` by `theta` (which is measured in radians).
        '''
        sn = sin(theta)
        cs = cos(theta)
        # rx, ry = np.matmul([[cs, -sn], [sn, cs]], [x - pivot_x, y - pivot_y])
        # return pivot_x + rx, pivot_y + ry
        xp = x - pivot_x
        yp = y - pivot_y
        return pivot_x + cs * xp - sn * yp, pivot_y + sn * xp + cs * yp
    
    @staticmethod
    def normalized_vector(x:float, y:float, length:float=1, default:Tuple[float, float]=(1, 0)) -> Tuple[float, float]:
        '''
        Return the vector `(x, y)` normalized to have a length of `length`.

        If `(x, y) == (0, 0)`, then return `default`.
        '''
        if x == 0 and y == 0:
            return default
        inv_cur_length = length * (x * x + y * y) ** -.5
        return x * inv_cur_length, y * inv_cur_length
    
    @staticmethod
    def vector_add(x0:float, y0:float, x1:float, y1:float) -> Tuple[float, float]:
        '''
        Return a new vector which adds `(x0, y0)` to `(x1, y1)`.
        '''
        return x0 + x1, y0 + y1

    @staticmethod
    def vector_subtract(x0:float, y0:float, x1:float, y1:float) -> Tuple[float, float]:
        '''
        Return a new vector which subtracts `(x1, y1)` from `(x0, y0)`.
        '''
        return x0 - x1, y0 - y1

    @staticmethod
    def vector_multiply(x:float, y:float, v:float) -> Tuple[float, float]:
        '''
        Return a new vector which multiples `(x, y)` by `v`.
        '''
        return x * v, y * v

    @staticmethod
    def vector_magnitude(x:float, y:float) -> float:
        '''
        Return the length of the vector `(x, y)`.
        '''
        return (x*x+y*y)**.5
    
    @staticmethod
    def vector_sq_magnitude(x:float, y:float) -> float:
        '''
        Return the squared length of the vector `(x, y)`.
        '''
        return x*x+y*y

    @staticmethod
    def cos(x:float) -> float:
        return cos(x)

    @staticmethod
    def sin(x:float) -> float:
        return sin(x)

    @staticmethod
    def tan(x:float) -> float:
        return tan(x)

    @staticmethod
    def acos(x:float) -> float:
        return acos(x)

    @staticmethod
    def asin(x:float) -> float:
        return asin(x)

    @staticmethod
    def atan2(y:float, x:float) -> float:
        return atan2(y, x)

    @staticmethod
    def current_date_time() -> str:
        '''
        Return a numerical date-time string of the current time in the following format: `YYYYMMDDHHmmSS`

        where `MM` is month and `mm` is minute.

        The string is of constant length, as single-digit numbers are zero-padded.

        With this format, a lexicological sorting, from earliest in alphabet to latest, of the date-time strings also results in a chronological sorting, from oldest to newest.
        '''
        return current_date_time()

    @staticmethod
    def cmp(a:float, b:float) -> int:
        '''
        Return 1 if `a > b`, 0 if `a == b`, or -1 if `a < b`.
        '''
        return (a > b) - (a < b)

    @staticmethod
    def sign(x:float) -> int:
        '''
        Return 1 if `x` is positive, 0 if `x` is zero, or -1 if `x` is negative.
        '''
        return PygEnv.cmp(x, 0)
    
    @staticmethod
    def distance(x0:float, y0:float, x1:float, y1:float) -> float:
        '''
        Return the Euclidean distance between points `(x0, y0)` and `(x1, y1)`.
        '''
        return (PygEnv.sq(x1 - x0) + PygEnv.sq(y1 - y0)) ** .5

    @staticmethod
    def sq_distance(x0:float, y0:float, x1:float, y1:float) -> float:
        '''
        Return the squared Euclidean distance between points `(x0, y0)` and `(x1, y1)`.

        This is useful in particular when comparing distances so that square roots can be avoided in calculations.
        '''
        return PygEnv.sq(x1 - x0) + PygEnv.sq(y1 - y0)

    @staticmethod
    def project_vector(v0x:float, v0y:float, v1x:float, v1y:float) -> Tuple[float, float]:
        '''
        Project vector `(v0x, v0y)` onto vector `(v1x, v1y)`.
        '''
        return PygEnv.scale_vector(v1x, v1y, PygEnv.dot_product(v0x, v0y, v1x, v1y) / PygEnv.dot_product(v1x, v1y, v1x, v1y))
    
    @staticmethod
    def dot_product(v0x:float, v0y:float, v1x:float, v1y:float) -> float:
        '''
        Return the dot product of vectors `(v0x, v0y)` and `(v1x, v1y)`.
        '''
        return v0x * v1x + v0y * v1y
    
    @staticmethod
    def scale_vector(x:float, y:float, scalar:float) -> Tuple[float, float]:
        '''
        Scale the vector `(x, y)` by `scalar`.
        '''
        return x * scalar, y * scalar

    @staticmethod
    def point_orientation(x0:float, y0:float, x1:float, y1:float, x2:float, y2:float) -> int:
        return PygEnv.cmp((x2 - x1) * (y1 - y0), (x1 - x0) * (y2 - y1))

    @staticmethod
    def segments_intersect(
        x0:float, y0:float, x1:float, y1:float,
        x2:float, y2:float, x3:float, y3:float,
        # check_endpoints=True,
    ) -> bool:
        '''
        Given endpoints of two line segments, return whether the segments intersect each other.
        '''
        return PygEnv.point_orientation(x0, y0, x1, y1, x2, y2) != PygEnv.point_orientation(x0, y0, x1, y1, x3, y3) and PygEnv.point_orientation(x2, y2, x3, y3, x0, y0) != PygEnv.point_orientation(x2, y2, x3, y3, x1, y1)

    @staticmethod
    def line_intersection_point(
        x0:float, y0:float, x1:float, y1:float,
        x2:float, y2:float, x3:float, y3:float,
    ) -> Tuple[float, float]:
        '''
        Given two pairs of points `{(x0, y0), (x1, y1)}` and `{(x2, y2), (x3, y3)}` which define two lines, return the point of intersection between the two lines, or return None if there is no intersection.
        '''
        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
        D = (x0-x1)*(y2-y3)+(y1-y0)*(x2-x3)
        if D == 0:
            return None
        D = 1./D
        A = (x0*y1-y0*x1)*D
        B = (x2*y3-y2*x3)*D
        return A*(x2-x3)-B*(x0-x1), A*(y2-y3)-B*(y0-y1)

    @staticmethod
    def reflect_vector(vx:float, vy:float, rx:float, ry:float) -> Tuple[float, float]:
        '''
        Reflect vector `(vx, vy)` across vector `(rx, ry)`.
        '''
        s = (rx*rx + ry*ry) ** -.5
        nx = ry * s
        ny = -rx * s
        d = 2 * (vx * nx + vy * ny)
        return vx - d * nx, vy - d * ny

    @staticmethod
    def reflect_segment(
        x0:float, y0:float, x1:float, y1:float,
        x2:float, y2:float, x3:float, y3:float,
    ) -> Tuple[float, float, float, float]:
        '''
        The segment defined by `{(x0, y0), (x1, y1)}` is split into two smaller segments at the point of intersection `P` between `{(x0, y0), (x1, y1)}` and `{(x2, y2), (x3, y3)}`.
        Then, reflect the segment `{P, (x1, y1)}` across `{(x2, y2), (x3, y3)}` and denote it `{P, Q}`.
        Return the points `P` and `Q` as a flattened tuple (four floats in a single-dimensional tuple).
        '''
        Px, Py = PygEnv.line_intersection_point(x0, y0, x1, y1, x2, y2, x3, y3)
        rx, ry = PygEnv.reflect_vector(x1 - Px, y1 - Py, x3 - x2, y3 - y2)
        return Px, Py, Px + rx, Py + ry

    @staticmethod
    def rectangle_contains_point(rect_x0:float, rect_y0:float, rect_x1:float, rect_y1:float, point_x:float, point_y:float) -> bool:
        return point_x >= rect_x0 and point_x <= rect_x1 and point_y >= rect_y0 and point_y <= rect_y1

    @staticmethod
    def vector_angle(v0x:float, v0y:float, v1x:float, v1y:float) -> float:
        '''
        Return the angle between the two vectors.  The returned angle is always between 0 (inclusive) and π (exclusive).  Neither of the vectors may be zero.
        '''
        return acos((v0x * v1x + v0y * v1y) / ((v1x * v1x + v1y * v1y) * (v0x * v0x + v0y * v0y)) ** .5)
    
    @staticmethod
    def lerp_angle(a:float, b:float, t:float, max_delta:float=7) -> float:
        '''
        Return a new angle that is interpolated between angles `a` and `b` by the inerpolant `t` and is within the shorter arc between the points at angles `a` and `b`.

        `max_delta` is the maximum angle by which the returned angle is incremented/decremented from `a` toward `b`.
        '''
        a %= PygEnv.TAU
        b %= PygEnv.TAU
        delta = (b - a + int(abs(a-b) > PygEnv.PI) * (2 * int(a > b) - 1) * PygEnv.TAU) * t
        if abs(delta) > max_delta:
            if delta < 0:
                delta = -max_delta
            else:
                delta = max_delta
        return (a + delta) % PygEnv.TAU
    
    @staticmethod
    def angle_from_to(x0:float, y0:float, x1:float, y1:float) -> float:
        '''
        Return the angle from `(x0, y0)` to `(x1, y1)` in radians.
        '''
        return atan2(y1 - y0, x1 - x0)
    
    @staticmethod
    def angle_delta(a:float, b:float) -> float:
        '''
        Return the difference between angles `a` and `b`.  The returned value is always less than or equal to π.
        '''
        t = abs(a - b) % PygEnv.TAU
        if t >= PygEnv.PI:
            return PygEnv.TAU - t
        else:
            return t
    
    @staticmethod
    def lerp_angle_direction(a:float, b:float) -> int:
        return PygEnv.sign(b - a + int(abs(a-b) > PygEnv.PI) * (2 * int(a > b) - 1) * PygEnv.TAU)
    
    @staticmethod
    def lerp_vector(x0:float, y0:float, x1:float, y1:float, t:float, max_delta: float = None) -> Tuple[float, float]:
        '''
        Linearly interpolate from vector `(x0, y0)` to `(x1, y1)` by the interpolant `t`.
        '''
        if max_delta is None:
            return PygEnv.lerp(x0, x1, t), PygEnv.lerp(y0, y1, t)
        lx = PygEnv.lerp(x0, x1, t)
        ly = PygEnv.lerp(y0, y1, t)
        if PygEnv.sq_distance(x0, y0, lx, ly) > max_delta * max_delta:
            dx, dy = PygEnv.normalized_vector(lx - x0, ly - y0, length=max_delta)
            return x0 + dx, y0 + dy
        return lx, ly
    
    @staticmethod
    def get_world_shape_vertices(shape:Union[pym.Poly, pym.Segment]) -> List[pym.Vec2d]:
        if shape.body is None:
            raise Exception(f'Cannot convert vertices from local coordinates to world coordinates because the shape is not attached to a body.')
        p = shape.body.position
        if isinstance(shape, pym.Poly):
            world_vertices = []
            for v in shape.get_vertices():
                r = v.rotated(shape.body.angle)
                world_vertices.append(p+r)
        else:
            ra = shape.a.rotated(shape.body.angle)
            rb = shape.b.rotated(shape.body.angle)
            world_vertices = [p+ra, p+rb]
        return world_vertices
    
    @staticmethod
    def to_pymunk_vector(x:float, y:float) -> pym.Vec2d:
        return pym.Vec2d(x, y)

    @staticmethod
    def rect(x: float, y: float, w: float, h: float) -> pyg.Rect:
        '''
        Alias for pygame.Rect.
        '''
        return pyg.Rect(x, y, w, h)







    def generate_offsets_l1(self, range: float):
        '''
        Generate a list of all possible integer coordinates `(x, y)` such that `abs(x) + abs(y) <= range`.

        The list is saved as `PygEnv.offsets_l1[str(range)]` and can alternatively be obtained using `PygEnv.get_offsets_l1(range)`.

        Parameters
        ----------
        `range` : float
            The range of the square which contains all points in the generated list.

        Example usage
        -------------
        ```
        env = PygEnv()
        env.generate_offsets_l1(4.5)
        offsets = env.get_offsets_l1(4.5)
        ```
        '''
        r = int(range)
        self.offsets_l1[str(range)] = [
            (x, y)
            for y in range(-r, r+1)
            for x in range(-r, r+1)
            if abs(x)+abs(y)<=range
        ]

    def generate_offsets_l2(self, radius: float):
        '''
        Generate a list of all possible integer coordinates `(x, y)` such that `x*x + y*y <= radius*radius`.

        The list is saved as `PygEnv.offsets_l2[str(radius)]` and can alternatively be obtained using `PygEnv.get_offsets_l2(radius)`.

        Parameters
        -
        `radius` : float
            The radius of the circle which contains all points in the generated list.

        Example usage
        -------------
        ```
        env = PygEnv()
        env.generate_offsets_l2(4.5)
        offsets = env.get_offsets_l2(4.5)
        ```
        '''
        r = int(radius)
        r2 = radius*radius
        return [
            (x, y)
            for y in range(-r, r+1)
            for x in range(-r, r+1)
            if x*x+y*y<=r2
        ]

    def get_offsets_l1(self, range: float, generate: bool = False) -> list[tuple[int, int]]:
        '''
        Retrieve a list of coordinates centered around `(0, 0)`, generated with `PygEnv.generate_offsets_l1(range)`.

        Parameters
        ----------
        `range` : float
            The range of the square which contains all points in the returned list.
        `generate` : bool
            If True, and a list for the given `range` has not yet been generated, then generate a new list.
            Otherwise, raise an `Exception` when no list has been generated for the given `range`.
        
        Returns
        -------
        `list`
            A list of tuples, where each tuple contains two integers.
        
        Example usage
        -------------
        ```
        env = PygEnv()
        env.generate_offsets_l1(4.5)
        offsets = env.get_offsets_l1(4.5)
        '''
        if str(range) not in self.offsets_l1:
            if not generate:
                raise Exception(f'L1 offsets for range={range} have not been generated yet.')
            self.generate_offsets_l1(range)
        return self.offsets_l1[str(range)]

    def get_offsets_l2(self, radius: float, generate: bool = False) -> list[tuple[int, int]]:
        '''
        Retrieve a list of coordinates centered around `(0, 0)`, generated with `PygEnv.generate_offsets_l2(radius)`.

        Parameters
        ----------
        `range` : float
            The radius of the circle which contains all points in the returned list.
        `generate` : bool
            If True, and a list for the given `range` has not yet been generated, then generate a new list.
            Otherwise, raise an `Exception` when no list has been generated for the given `range`.
        
        Returns
        -------
        `list`
            A list of tuples, where each tuple contains two integers.
        
        Example usage
        -------------
        ```
        env = PygEnv()
        env.generate_offsets_l1(4.5)
        offsets = env.get_offsets_l1(4.5)
        '''
        if str(radius) not in self.offsets_l2:
            if not generate:
                raise Exception(f'L2 offsets for radius={radius} have not been generated yet.')
            self.generate_offsets_l2(radius)
        return self.offsets_l2[str(radius)]
    
    @staticmethod
    def get_adjacency_offset(index: int, wrap: bool = False) -> tuple[int, int]:
        '''
        Parameters
        ----------
        `index` : int
            The index of `PygEnv.ADJACENCY_OFFSETS` at which a value is returned.
        `wrap` : bool
            If True, then the index is wrapped to always be within the interval of 0-3, where 4 is wrapped to 0, 5 is wrapped to 1, etc.
            Otherwise, a `ValueError` is raised when the index is not in the interval of 0-3.

        Returns
        -------
        `tuple`
            A tuple of two integers representing an adjacency offset.
        
        Example usage
        -------------
        ```
        env = PygEnv()
        offset = env.get_adjacency_offset(0)
        
        x, y = 3, 5
        new_x, new_y = env.vector_add(x, y, *offset)
        ```
        '''
        if not wrap:
            if index < 0 or index >= 4:
                raise ValueError(f'index={index} is out of range for adjacency offsets.')
        else:
            index = index % 4
        return PygEnv.ADJACENCY_OFFSETS[index]

    def register_collision_callback(self, collision_type_a: int, collision_type_b: int, begin_callback: Callable[[pym.Arbiter], bool] = None, end_callback: Callable[[pym.Arbiter], bool] = None, post_solve_callback: Callable[[pym.Arbiter], bool] = None, pre_solve_callback: Callable[[pym.Arbiter], bool] = None):
        if not self.using_physics:
            self.init_physics()
        t = (collision_type_a, collision_type_b) if collision_type_a < collision_type_b else (collision_type_b, collision_type_a)
        if t in self.collision_handlers:
            handler = self.collision_handlers[t]
        else:
            handler = self.pymunk_space.add_collision_handler(collision_type_a=collision_type_a, collision_type_b=collision_type_b)
            self.collision_handlers[t] = handler
        if begin_callback is not None:
            def f(arbiter: pym.Arbiter, space: pym.Space, data: dict[any, any]) -> bool:
                return begin_callback(arbiter)
            handler.begin = f
        if end_callback is not None:
            def f(arbiter: pym.Arbiter, space: pym.Space, data: dict[any, any]) -> bool:
                return end_callback(arbiter)
            handler.separate = f
        if post_solve_callback is not None:
            def f(arbiter: pym.Arbiter, space: pym.Space, data: dict[any, any]) -> bool:
                return post_solve_callback(arbiter)
            handler.post_solve = f
        if pre_solve_callback is not None:
            def f(arbiter: pym.Arbiter, space: pym.Space, data: dict[any, any]) -> bool:
                return pre_solve_callback(arbiter)
            handler.pre_solve = f

    def init_physics(self) -> None:
        self.pymunk_space = pym.Space()
        self.using_physics = True
        self.pymunk_space.gravity = 0, -1

    def set_gravity(self, x:float, y:float) -> None:
        if not self.using_physics:
            self.init_physics()
        self.pymunk_space.gravity = x, y
    
    def set_physics_iterations_per_step(self, iterations: int):
        '''
        The default for pymunk is 10 iterations.  Increasing this improves physics accuracy but requires more CPU time.
        '''
        if not self.using_physics:
            self.init_physics()
        self.pymunk_space.iterations = iterations

    def add_static_line(self, color: tuple[int, int, int], p0: tuple[float, float], p1: tuple[float, float], elasticity: float = 0.5, friction: float = 0.8, radius: float = 1) -> pym.Shape:
        if not self.using_physics:
            self.init_physics()
        body = pym.Body(body_type=pym.Body.STATIC)
        shape = pym.Segment(body=body, a=p0, b=p1, radius=radius)
        shape.elasticity = elasticity
        shape.friction = friction
        shape.color = color
        self.pymunk_space.add(body)
        self.pymunk_space.add(shape)
        return shape
    
    def add_static_lines_around_camera_edge(self, elasticity: float = .5, friction: float = .8) -> tuple[pym.Shape, pym.Shape, pym.Shape, pym.Shape]:
        return (
            self.add_static_line((0, 0, 0), (self.camera_blc_world_x, self.camera_blc_world_y), (self.camera_trc_world_x, self.camera_blc_world_y), elasticity=elasticity, friction=friction, radius=0),
            self.add_static_line((0, 0, 0), (self.camera_trc_world_x, self.camera_blc_world_y), (self.camera_trc_world_x, self.camera_trc_world_y), elasticity=elasticity, friction=friction, radius=0),
            self.add_static_line((0, 0, 0), (self.camera_blc_world_x, self.camera_trc_world_y), (self.camera_trc_world_x, self.camera_trc_world_y), elasticity=elasticity, friction=friction, radius=0),
            self.add_static_line((0, 0, 0), (self.camera_blc_world_x, self.camera_blc_world_y), (self.camera_blc_world_x, self.camera_trc_world_y), elasticity=elasticity, friction=friction, radius=0),
        )

    def spawn_circle(self, color: tuple[int, int, int], pos: tuple[float, float] = (0, 0), radius: float = 1.0, density: float = 1, elasticity: float = 0, friction: float = 0.8, drag: float = 0.01, angular_drag: float = 0.01, draw_line: bool = True, collision_type: int = 0) -> pym.Shape:
        if not self.using_physics:
            self.init_physics()
        mass = pym.area_for_circle(0, radius) * density
        m = pym.moment_for_circle(mass, 0, radius)
        body = pym.Body(mass=mass, moment=m, body_type=pym.Body.DYNAMIC)
        body.position = pos
        body.drag = drag
        body.angular_drag = angular_drag
        shape = pym.Circle(body=body, radius=radius, offset=(0, 0))
        shape.elasticity = elasticity
        shape.friction = friction
        shape.color = color
        shape.draw_line = draw_line
        shape.collision_type = collision_type
        self.pymunk_space.add(body)
        self.pymunk_space.add(shape)
        return shape

    def spawn_poly(self, color: tuple[int, int, int], pos: tuple[float, float], verts: list[tuple[float, float]], vertices_rot: float = 0, density: float = 1, elasticity: float = 0, friction: float = 0.8, drag: float = 0.01, angular_drag: float = 0.01) -> pym.Shape:
        if not self.using_physics:
            self.init_physics()
        mass = pym.area_for_poly(vertices=verts, radius=1) * density
        m = pym.moment_for_poly(mass, vertices=verts, radius=1)
        body = pym.Body(mass=mass, moment=m, body_type=pym.Body.DYNAMIC)
        body.position = pos
        body.drag = drag
        body.angular_drag = angular_drag
        shape = pym.Poly(body=body, vertices=verts, radius=1, transform=pym.Transform.rotation(vertices_rot))
        shape.elasticity = elasticity
        shape.friction = friction
        shape.color = color
        self.pymunk_space.add(body)
        self.pymunk_space.add(shape)
        return shape

    def spawn_walls_on_camera_edge(self, elasticity: float = 0.5, friction: float = 0.8):
        self.add_static_line((0, 0, 0), (self.camera_blc_world_x, self.camera_blc_world_y), (self.camera_trc_world_x, self.camera_blc_world_y), elasticity=elasticity, friction=friction)
        self.add_static_line((0, 0, 0), (self.camera_trc_world_x, self.camera_blc_world_y), (self.camera_trc_world_x, self.camera_trc_world_y), elasticity=elasticity, friction=friction)
        self.add_static_line((0, 0, 0), (self.camera_blc_world_x, self.camera_trc_world_y), (self.camera_trc_world_x, self.camera_trc_world_y), elasticity=elasticity, friction=friction)
        self.add_static_line((0, 0, 0), (self.camera_blc_world_x, self.camera_blc_world_y), (self.camera_blc_world_x, self.camera_trc_world_y), elasticity=elasticity, friction=friction)

    # def add_body(self, shape:int, position:Tuple[float, float], mass:float=None, density:float=1, rotation:float=0, elasticity:float=0.3, color:Union[None, Tuple[int, int, int]]=None, body_type:int=None, as_name:Hashable=None, do_rendering:bool=True, **kwargs) -> pym.Shape:
    #     '''
    #     `shape` can be `PygEnv.BODY_SHAPE_CIRCLE`, `PygEnv.BODY_SHAPE_RECT`, `PygEnv.BODY_SHAPE_POLY`, or `PygEnv.BODY_SHAPE_SEGMENT`.

    #     `body_type` can be `PygEnv.BODY_TYPE_DYNAMIC`, `PygEnv.BODY_TYPE_KINEMATIC`, or `PygEnv.BODY_TYPE_STATIC`.  If None, then default is dynamic.

    #     If `color` is None, then it takes the value of the (additive) inverse of the current background color by default.

    #     `do_rendering` can be set to False to disable the automatic rendering with basic shapes.  This would allow you to do your own rendering instead.

    #     `rotation` is in radians.

    #     `kwargs` may include:

    #         (1) `radius` to define a circle's radius

    #         (2) `size` to define a rectangle's size, which can be a float (for a square) or a 2-sequence of floats (for a rectangle) specifying (width, height)

    #         (3) `vertices` to define a poly's vertices or a segment's endpoints

    #         (4) `width` to define the stroke width of a line segment body (`PygEnv.BODY_SHAPE_SEGMENT`)

    #     Example:
    #     ```
    #     env = PygEnv()
    #     env.set_world_zoom(20)
    #     env.set_gravity(0, -10)
    #     triangle = env.add_body(PygEnv.BODY_SHAPE_POLY, (0, 0), vertices=((-1, -1/3), (0, PygEnv.SQRT3-1/3), (1, -1/3)))
    #     platform = env.add_body(PygEnv.BODY_SHAPE_SEGMENT, (0, 0), body_type=PygEnv.BODY_TYPE_STATIC, vertices=((-1/2, -2), (-4, -1)))
    #     env.run_loop()
    #     ```
    #     '''
    #     def kwarg(key, default=None):
    #         if key in kwargs:
    #             return kwargs[key]
    #         # print(f'Expected keyword argument \'{key}\' for Pygenv.add_body().  Using {default} as the default value.')
    #         return default
    #     if not self.using_physics:
    #         self.init_physics()
    #     if body_type is None:
    #         body_type = PygEnv.BODY_TYPE_DYNAMIC
    #     if as_name is None:
    #         as_name = len(self.pymunk_space.bodies)
    #     body = pym.Body(body_type=body_type)
    #     body.position = position
    #     body.angle = rotation * PygEnv.RAD_2_DEG
    #     if shape == PygEnv.BODY_SHAPE_CIRCLE:
    #         pym_shape = pym.Circle(body, radius=kwarg('radius', 1))
    #     elif shape == PygEnv.BODY_SHAPE_RECT:
    #         size = kwarg('size', 1)
    #         if isinstance(size, tuple):
    #             w, h = size
    #         elif isinstance(size, (float, int)):
    #             w = size
    #             h = size
    #         pym_shape = pym.Poly.create_box(body, size=(w, h), radius=0.01)
    #     elif shape == PygEnv.BODY_SHAPE_POLY:
    #         pym_shape = pym.Poly(body, vertices=kwarg('vertices', []))
    #     elif shape == PygEnv.BODY_SHAPE_SEGMENT:
    #         v0, v1 = kwarg('vertices', (pym.Vec2d(0, 0), pym.Vec2d(1, 1)))
    #         if not isinstance(v0, pym.Vec2d):
    #             v0 = pym.Vec2d(*v0)
    #         if not isinstance(v1, pym.Vec2d):
    #             v1 = pym.Vec2d(*v1)
    #         pym_shape = pym.Segment(body, v0, v1, radius=kwarg('width', 1))
    #         delta = v1 - v0
    #         pym_shape.segment_angle = atan2(delta[1], delta[0])
    #     if color is None:
    #         color = PygEnv.color_inverse(self.bg_color)
    #     pym_shape.color = color
    #     pym_shape.do_rendering = do_rendering
    #     pym_shape.elasticity = elasticity
    #     if mass is None:
    #         pym_shape.density = density
    #     else:
    #         pym_shape.mass = mass
    #     self.pymunk_space.add(body)
    #     self.pymunk_space.add(pym_shape)
    #     self.pymunk_bodies[as_name] = body
    #     self.pymunk_shapes[as_name] = pym_shape
    #     return pym_shape
    
    def get_body_shape(self, body_name:Hashable) -> pym.Shape:
        '''
        `body_name` is the name of the body specified with the `as_name=` keyword argument in `PygEnv.add_body()`.  If no name was specified, serial integers are the default names.

        The `pymunk.Shape` object contains a reference to the body to which the shape is attached, on which you can apply forces, change position, etc.
        '''
        if body_name not in self.pymunk_shapes:
            raise KeyError(f'Could not find a shape under the name \'{body_name}\'.')
        return self.pymunk_shapes[body_name]

    # def is_shape_in_view(self, shape: pym.Shape):
    # 	if isinstance(shape, pym.Poly):
    # 		if shape.

    def clear_bodies(self):
        '''
        Delete all bodies in the world.
        '''
        for obj in self.pymunk_space.bodies + self.pymunk_space.shapes:
            self.pymunk_space.remove(obj)
        self.pymunk_bodies.clear()
        self.pymunk_shapes.clear()

    def remove_body(self, *args: list[pym.Body], remove_shapes: bool = True):
        for b in args:
            if remove_shapes:
                for s in b.shapes:
                    self.remove_shape(s, remove_body=False)
            if b in self.pymunk_space.bodies:
                self.pymunk_space.remove(b)

    def remove_shape(self, *args: list[pym.Shape], remove_body: bool = True):
        for b in args:
            if remove_body:
                self.remove_body(b.body, remove_shapes=False)
            if b in self.pymunk_space.shapes:
                self.pymunk_space.remove(b)

    def remove_constraint(self, *args: list[pym.Constraint]):
        for b in args:
            if b in self.pymunk_space.constraints:
                self.pymunk_space.remove(b)

    def raycast_first(self, p0: tuple[float, float], p1: tuple[float, float], radius: float = None) -> Union[pym.SegmentQueryInfo, None]:
        if radius is None:
            radius = 1. / self.world_zoom
        info = self.pymunk_space.segment_query_first(p0, p1, radius, pym.ShapeFilter())
        return info

    def raycast(self, p0: tuple[float, float], p1: tuple[float, float], radius: float = None) -> list[pym.SegmentQueryInfo]:
        if radius is None:
            radius = 1. / self.world_zoom
        info = self.pymunk_space.segment_query(p0, p1, radius, pym.ShapeFilter())
        return info
    
    def get_shape_under_mouse(self) -> Union[pym.Shape, None]:
        info = self.raycast_first((self.mouse_world_pos_x, self.mouse_world_pos_y), (self.mouse_world_pos_x+1/self.world_zoom, self.mouse_world_pos_y))
        if info is not None:
            return info.shape




    def stopwatch_lap(self) -> float:
        return self.stopwatch.lap()

    def create_weighted_choice(self, name:Hashable, initial_choices:List[Tuple[float, Any]] = None) -> None:
        if name in self.weighted_choices:
            raise IndexError(f'A weighted choice already exists with the name "{name}"')
        self.weighted_choices[name] = WeightedChoice([(1, None)])
        self.set_weighted_choice(name, initial_choices)

    def set_weighted_choice(self, name:Hashable, choices:List[Tuple[float, Any]]) -> None:
        if name not in self.weighted_choices:
            raise IndexError(f'A weighted choice does not exist with the name "{name}"')
        self.weighted_choices[name].set_choices(choices)

    def add_to_weighted_choice(self, name:Hashable, choice_item:Tuple[float, Any]) -> None:
        '''
        choice_item is a tuple of the form (weight, item).
        '''
        if name not in self.weighted_choices:
            raise IndexError(f'A weighted choice does not exist with the name "{name}"')
        wc = self.weighted_choices[name]
        for i,(weight,item) in enumerate(wc.choices):
            if item==choice_item[1]:
                wc.choices[i] = (weight+choice_item[0],item)
                wc.set_choices(wc.choices)
                return # update the item if found and stop
        # if not found, then add new item
        wc.set_choices(wc.choices + [choice_item])

    def weighted_choice(self, name:Hashable):
        '''
        Compute a weighted choice.  Call PygEnv.create_weighted_choice(name) first to initialize it.
        '''
        if name not in self.weighted_choices:
            raise IndexError(f'A weighted choice does not exist with the name "{name}"')
        return self.weighted_choices[name].choice()

    def load_palette(self, name:str) -> None:
        try:
            p = get_palette(name)
            self.palettes[name] = p
            print(f'Loaded palette "{name}".')
        except Exception as e:
            print(f'Could not load palette.  Exception: {e}')

    def get_palette_color(self, name:str, index:Union[int, float]) -> Tuple[int, int, int]:
        '''
        If `index` is an int, then use that exact index for the palette.  If `index` is a float, then interpolate over the length of the palette, using `index` as the interpolant.
        '''
        if name not in self.palettes:
            self.load_palette(name)
        if type(index) is int:
            p = self.palettes[name]
            return p[index % len(p)]
        elif type(index) is float:
            p = self.palettes[name]
            r1,g1,b1 = p[int(index)%len(p)]
            r2,g2,b2 = p[(int(index)+1)%len(p)]
            t = index%1
            return int(0.5+(1-t)*r1+t*r2), int(0.5+(1-t)*g1+t*g2), int(0.5+(1-t)*b1+t*b2)

    def get_font(self, name:str = 'arial', size:int = 16) -> pyg.font.Font:
        if (name, size) in self.loaded_fonts:
            return self.loaded_fonts[(name, size)]
        f = load_font(name = name, size = size)
        self.loaded_fonts[(name, size)] = f
        return f

    def add_ui_element(self, element_id, element: gui.core.ui_element.UIElement, **kwargs):
        # label = Label(
        # 	env = self,
        # 	font = self.get_font(name = font_name, size = font_size),
        # 	**kwargs
        # )
        self.ui_elements[element_id] = element
        changed = False
        for keyword, arg in kwargs.items():
            if hasattr(element, keyword):
                setattr(element, keyword, arg)
                changed = True
        if changed:
            element.rebuild()
        return self.ui_elements[element_id]
    
    def destroy_ui_element(self, element_id):
        if element_id in self.ui_elements:
            del self.ui_elements[element_id]
        else:
            raise ValueError(f'Element ID not recognized: {element_id}')
    
    def get_ui_element(self, element_id):
        if element_id in self.ui_elements:
            return self.ui_elements[element_id]
        else:
            raise ValueError(f'Element ID not recognized: {element_id}')

    # def set_label_text(self, label_id, new_text):
    # 	if label_id in self.labels:
    # 		self.labels[label_id].set_text(new_text)
    # 	else:
    # 		raise ValueError('label_id "{}" not recognized.'.format(label_id))

    # def set_label_relative_position(self, label_id, new_x, new_y):
    # 	if label_id in self.labels:
    # 		self.labels[label_id].set_relative_position(new_x, new_y)
    # 	else:
    # 		raise ValueError('label_id "{}" not recognized.'.format(label_id))

    # def set_label_absolute_position(self, label_id, new_x, new_y):
    # 	if label_id in self.labels:
    # 		self.labels[label_id].set_absolute_position(new_x, new_y)
    # 	else:
    # 		raise ValueError('label_id "{}" not recognized.'.format(label_id))

    # def set_label_bg_rect(self, label_id, **kwargs):
    # 	if label_id in self.labels:
    # 		self.labels[label_id].set_bg_rect(**kwargs)
    # 	else:
    # 		raise ValueError('label_id "{}" not recognized.'.format(label_id))

    # def destroy_label(self, label_id):
    # 	if label_id in self.labels:
    # 		del self.labels[label_id]
    # 	else:
    # 		raise ValueError('label_id "{}" not recognized.'.format(label_id))

    def tile_sprite(self, sprite_sheet, row, col, tiling_x, tiling_y, as_name, flip=False, scale=1.0):
        sheet = self.sprite_sheets[sprite_sheet]
        w, h = sheet.image_size
        surf = pyg.Surface((w * tiling_x * scale, h * tiling_y * scale), pyg.SRCALPHA)
        sheet.blit(surf, 0, 0, row, col, flip, scale, 0, tiling_x, tiling_y)
        self.tiled_sprites[as_name] = surf
        print(f'Tiled sprite \'{as_name}\' from sprite sheet \'{sprite_sheet}\' at row={row}, col={col} with tiling_x={tiling_x}, tiling_y={tiling_y}.  Total tiled size: ({surf.get_width()}, {surf.get_height()})')

    def set_bg_color(self, color):
        self.bg_color = color

    def clear_screen_each_frame(self, flag):
        '''
        True by default.  Set to False to prevent the screen from being cleared each frame.
        '''
        self.do_draw_background = flag
    
    def angle_to_mouse_world_pos(self, from_x:float=None, from_y:float=None) -> float:
        '''
        Return the angle (in radians) from world point `(from_x, from_y)` to the current mouse world position.

        If `from_x` is None, then it takes the value of the world position X that is at the center of the screen.
        If `from_y` is None, then it takes the value of the world position Y that is at the center of the screen.
        '''
        dx, dy = self.delta_to_mouse_world_pos(from_x, from_y)
        return PygEnv.atan2(dy, dx)
    
    def delta_to_mouse_world_pos(self, from_x:float=None, from_y:float=None) -> Tuple[float, float]:
        '''
        Return the delta/displacement/difference vector from world point `(from_x, from_y)` to the current mouse world position.

        If `from_x` is None, then it takes the value of the world position X that is at the center of the screen.
        If `from_y` is None, then it takes the value of the world position Y that is at the center of the screen.
        '''
        if from_x is None:
            from_x = self.camera_x
        if from_y is None:
            from_y = self.camera_y
        return self.mouse_world_pos_x - from_x, self.mouse_world_pos_y - from_y
    
    def distance_to_mouse_world_pos(self, from_x:float=None, from_y:float=None) -> float:
        '''
        Return the distance from world point `(from_x, from_y)` to the current mouse world position.

        If `from_x` is None, then it takes the value of the world position X that is at the center of the screen.
        If `from_y` is None, then it takes the value of the world position Y that is at the center of the screen.
        '''
        return self.sq_distance_to_mouse_world_pos(from_x, from_y) ** .5

    def sq_distance_to_mouse_world_pos(self, from_x:float=None, from_y:float=None) -> float:
        '''
        Return the squared distance from world point `(from_x, from_y)` to the current mouse world position.

        If `from_x` is None, then it takes the value of the world position X that is at the center of the screen.
        If `from_y` is None, then it takes the value of the world position Y that is at the center of the screen.
        '''
        if from_x is None:
            from_x = self.camera_x
        if from_y is None:
            from_y = self.camera_y
        return PygEnv.sq_distance(from_x, from_y, self.mouse_world_pos_x, self.mouse_world_pos_y)

    def distance_to_mouse_pos(self, from_x:float=None, from_y:float=None) -> float:
        '''
        Return the distance from world point `(from_x, from_y)` to the current mouse world position.

        If `from_x` is None, then it takes the value of the world position X that is at the center of the screen.
        If `from_y` is None, then it takes the value of the world position Y that is at the center of the screen.
        '''
        return self.sq_distance_to_mouse_pos(from_x, from_y) ** .5

    def sq_distance_to_mouse_pos(self, from_x:float=None, from_y:float=None) -> float:
        '''
        Return the squared distance from world point `(from_x, from_y)` to the current mouse world position.

        If `from_x` is None, then it takes the value of the world position X that is at the center of the screen.
        If `from_y` is None, then it takes the value of the world position Y that is at the center of the screen.
        '''
        if from_x is None:
            from_x = self.HALF_WIDTH
        if from_y is None:
            from_y = self.HALF_HEIGHT
        return PygEnv.sq_distance(from_x, from_y, self.mouse_pos_x, self.mouse_pos_y)

    def set_tiling_bg_sprite(self, sprite_file_name, parallax = 1):
        try:
            self.bg_sprite = pyg.image.load(sprite_file_name).convert_alpha()
            self.bg_sprite_size = self.bg_sprite.get_size()
            self.bg_sprite_parallax = parallax
        except Exception as e:
            print(f'Could not load file \'{sprite_file_name}\' as background sprite.  Exception: {e}')

    def set_frozen_bg_sprite(self, sprite_file_name):
        self.set_tiling_bg_sprite(sprite_file_name, parallax = 0)

    def set_frame_rate(self, fr):
        if fr < 1 or fr > 120:
            return
        self.frame_rate = fr
        self.inv_frame_rate = 1. / self.frame_rate

    def set_world_zoom(self, zoom):
        if zoom <= 0:
            raise Exception('Cannot set world zoom to a value less than or equal to zero.')
        old_zoom = self.world_zoom
        self.world_zoom = min(max(zoom, self.min_zoom), self.max_zoom)
        self.inv_world_zoom = 1. / self.world_zoom
        scale = old_zoom * self.inv_world_zoom
        if scale != 1:
            self.pan_speed_x *= scale
            self.pan_speed_y *= scale
            self.recalculate_camera_world_corners()
    
    def set_zoomable(self, flag):
        if flag:
            self.set_zoom_sensitivity(1.1)
        else:
            self.set_zoom_sensitivity(1)
    
    def set_zoom_sensitivity(self, s):
        self.zoom_sensitivity = s
    
    def set_zoom_bounds(self, min, max):
        self.min_zoom = min
        self.max_zoom = max
        self.set_world_zoom(self.world_zoom)

    def draw_world_rect(self, color, x, y, w, h, width = 0, alpha = 255, rotation = 0):
        '''
        `rotation` is in radians.
        '''
        self.draw_screen_rect(color, floor(self.HALF_WIDTH + self.world_zoom * (x - self.camera_x)), floor(self.HALF_HEIGHT - self.world_zoom * (y - self.camera_y)), w * self.world_zoom, h * self.world_zoom, width=int(.5+width*self.world_zoom), alpha=alpha, rotation=rotation)

    def draw_world_circle(self, color, x, y, r, width = 0):
        self.draw_screen_circle(color, floor(self.HALF_WIDTH + self.world_zoom * (x - self.camera_x)), floor(self.HALF_HEIGHT - self.world_zoom * (y - self.camera_y)), r * self.world_zoom, width=int(.5+width*self.world_zoom))
    
    def draw_world_poly(self, color, vertices, width = 0):
        self.draw_screen_poly(color, [
            (
                floor(self.HALF_WIDTH + self.world_zoom * (vx - self.camera_x)),
                floor(self.HALF_HEIGHT - self.world_zoom * (vy - self.camera_y))
            )
            for vx, vy in vertices
        ], width)

    def draw_world_regular_poly(self, color: tuple[int, int, int], num_vertices: int, x: float, y: float, r: float, rotation: float = 0, width = 0):
        a = self.TAU / num_vertices
        self.draw_world_poly(color, [
            (x + r * self.cos(t), y + r * self.sin(t))
            for t in (t * a + rotation for t in range(num_vertices))
        ], width)

    def draw_world_line(self, color, x0, y0, x1, y1, width = None):
        if width is None:
            width = self.inv_world_zoom
        self.draw_screen_line(
            color,
            floor(self.HALF_WIDTH + self.world_zoom * (x0 - self.camera_x)),
            floor(self.HALF_HEIGHT - self.world_zoom * (y0 - self.camera_y)),
            floor(self.HALF_WIDTH + self.world_zoom * (x1 - self.camera_x)),
            floor(self.HALF_HEIGHT - self.world_zoom * (y1 - self.camera_y)),
            width=int(.5+width*self.world_zoom)
        )

    def draw_world_point(self, color, x, y):
        self.draw_world_line(color, x, y, x, y)

    def draw_world_sprite(self, sprite_sheet_name:str, x:float, y:float, row:int, col:int, flip:bool=False, scale:float=1, rotation:float=0, pivot_point:Tuple[float, float]=None, tiling_x:int=1, tiling_y:int=1, tiling_scale:float=1):
        '''
        `rotation` is in radians.

        `pivot_point` is in world units (relative to the sprite's position).
        '''
        sprite_scale = scale * self.world_zoom / self.sprite_sheets[sprite_sheet_name].pixels_per_unit
        self.draw_screen_sprite(sprite_sheet_name, floor(self.HALF_WIDTH + self.world_zoom * (x - self.camera_x)), floor(self.HALF_HEIGHT - self.world_zoom * (y - self.camera_y)), row, col, flip, sprite_scale, rotation, pivot_point, tiling_x, tiling_y, tiling_scale)

    def draw_world_tiled_sprite(self, tiled_sprite_name, x, y):
        self.draw_screen_tiled_sprite(tiled_sprite_name, floor(self.HALF_WIDTH + self.world_zoom * (x - self.camera_x)), floor(self.HALF_HEIGHT - self.world_zoom * (y - self.camera_y)))
    
    def draw_world_background_tile_sprites(self, sprite_sheet_name, row_col_selector, tiling_scaling=1.05):
        '''
        `row_col_selector` may be a tuple of the form `(row, col)` to indicate a constant row/col of the sprite sheet, making the background consist of a single, repeated sprite.

        `row_col_selector` may be a callable taking the snapped world position `(x, y)` (two arguments) and returning either `(row, col)` or `(row, col, flip)` so that each tile may be independently selected and flipped based on position.  This is around 2.5x - 3x as slow in performance as using a single, repeated sprite.

        `tiling_scaling` is useful to ensure that no lines appear between the tiled sprites.
        '''
        tiling_x = int(3+self.WIDTH*self.inv_world_zoom)
        tiling_y = int(3+self.HEIGHT*self.inv_world_zoom)
        if isinstance(row_col_selector, tuple):
            wx, wy = self.snapped_to_grid(self.camera_blc_world_x, self.camera_trc_world_y, 1, 1)
            row, col = row_col_selector # constant
            self.draw_world_sprite(sprite_sheet_name, wx - 1, wy + 1, row, col, tiling_x=tiling_x, tiling_y=tiling_y, tiling_scale=tiling_scaling)
        else:
            wx, wy = self.snapped_to_grid(self.camera_blc_world_x, self.camera_blc_world_y, 1, 1)
            for y in range(tiling_y):
                for x in range(tiling_x):
                    _wx, _wy = floor(-0.5 + wx + x), floor(0.5 + wy + y)
                    self.draw_world_sprite(sprite_sheet_name, _wx, _wy, *row_col_selector(_wx, _wy), scale=tiling_scaling)

    def draw_screen_rect(self, color, x, y, w, h, width = 0, alpha = 255, rotation = 0):
        '''
        `rotation` is in radians.
        '''
        if alpha >= 255 and rotation == 0:
            if self.rect_draw_mode == RectDrawMode.Center:
                x -= w * .5
                y -= h * .5
            pyg.draw.rect(self.screen, color, [int(x), int(y), int(w), int(h)], width)
        elif alpha <= 0:
            pass # nothing to draw
        else:
            s = pyg.Surface((w,h))
            s = s.convert_alpha()
            if alpha < 255:
                s.set_alpha(alpha)
            if width > 0:
                s.fill((0, 0, 0, 0))
                pyg.draw.rect(s, color, (0, 0, w, h), width=width)
            else:
                s.fill(color)
            if rotation != 0:
                s = pyg.transform.rotate(s, rotation * PygEnv.RAD_2_DEG)
            self.screen.blit(s, (x - s.get_width() * .5, y - s.get_height() * .5))

    def draw_screen_circle(self, color, x, y, r, width = 0):
        pyg.draw.circle(self.screen, color, (int(x), int(y)), int(r), width)

    def draw_screen_poly(self, color, vertices, width = 0):
        pyg.draw.polygon(self.screen, color, vertices, width)

    def draw_screen_line(self, color, x0, y0, x1, y1, width = 1):
        pyg.draw.line(self.screen, color, (int(x0), int(y0)), (int(x1), int(y1)), width)
    
    def draw_screen_point(self, color, x, y):
        self.draw_screen_line(color, x, y, x, y, width=1)

    def load_sprite_sheet(self, file, rows, cols, as_name=None, margin=0, pixels_per_unit=None):
        if as_name is None:
            as_name = 'spritesheet' + str(len(self.sprite_sheets))
        if isinstance(file, str):
            file = open(file, 'r')
        self.sprite_sheets[as_name] = SpriteSheet(file, rows, cols, margin=margin, pixels_per_unit=pixels_per_unit)
        print(f'Loaded sprite sheet \'{as_name}\', with {rows*cols} sprites and {pixels_per_unit} pixels per world unit.')

    def draw_screen_sprite(self, sprite_sheet_name:str, x:float, y:float, row:int, col:int, flip:bool=False, scale:float=1, rotation:float=0, pivot_point:Tuple[float, float]=None, tiling_x:int=1, tiling_y:int=1, tiling_scale:float=1):
        '''
        `rotation` is in radians.

        If `pivot_point` is not None, then the sprite is repositioned accordingly.  Otherwise, if the current sprite draw mode is `SpriteDrawMode.Center`, then the `pivot_point` is automatically set to the center of the sprite image.
        '''
        ss: SpriteSheet = self.sprite_sheets[sprite_sheet_name]
        if self.sprite_draw_mode == SpriteDrawMode.Center:
            x -= ss.image_size[0] * .5
            y -= ss.image_size[1] * .5
        ss.blit(self.screen, x, y, row, col, flip, scale, rotation, pivot_point, int(tiling_x), int(tiling_y), tiling_scale)

    def draw_screen_tiled_sprite(self, tiled_sprite_name, x, y):
        self.screen.blit(self.tiled_sprites[tiled_sprite_name], (int(x), int(y)))

    def set_rect_draw_mode(self, mode):
        if isinstance(mode, str):
            if mode == 'center':
                self.set_rect_draw_mode(RectDrawMode.Center)
            elif mode == 'corner':
                self.set_rect_draw_mode(RectDrawMode.Corner)
            else:
                raise Exception(f'Unrecognized rect draw mode: {mode}.  Can be either \'center\' or \'corner\'.')
        else:
            self.rect_draw_mode = mode

    def set_camera_pos(self, x, y):
        self.camera_x = x
        self.camera_y = y
        self.recalculate_camera_world_corners()
    
    def recalculate_camera_world_corners(self):
        self.camera_blc_world_x = self.camera_x - self.HALF_WIDTH * self.inv_world_zoom
        self.camera_blc_world_y = self.camera_y - self.HALF_HEIGHT * self.inv_world_zoom
        self.camera_trc_world_x = self.camera_x + self.HALF_WIDTH * self.inv_world_zoom
        self.camera_trc_world_y = self.camera_y + self.HALF_HEIGHT * self.inv_world_zoom

    def translate_camera(self, x, y):
        self.set_camera_pos(
            min(max(self.camera_x + x, self.panning_minx), self.panning_maxx),
            min(max(self.camera_y + y, self.panning_miny), self.panning_maxy)
        )

    def set_target_follow_speed(self, speed):
        self.target_follow_speed = speed

    def set_camera_target_pos(self, x, y):
        self.target_camera_x = x
        self.target_camera_y = y

    def set_camera_rect(self, lower_x, lower_y, upper_x, upper_y):
        center_x = (lower_x + upper_x) * .5
        center_y = (lower_y + upper_y) * .5
        self.set_camera_pos(center_x, center_y)
        scale = self.HALF_HEIGHT / (upper_y - center_y)
        self.set_world_zoom(scale)

    def unset_camera_target_pos(self):
        self.set_camera_target_pos(None, None)

    def set_frame_rate_controls(self, mode):
        self.double_frame_rate_key, self.half_frame_rate_key, self.increase_frame_rate_key, self.decrease_frame_rate_key = get_control_mode_keycodes(mode)

    def set_pan_controls(self, mode):
        self.pan_x_pos_key, self.pan_x_neg_key, self.pan_y_pos_key, self.pan_y_neg_key = get_control_mode_keycodes(mode)

    def set_screen_pan_speed(self, x, y):
        self.pan_speed_x = x
        self.pan_speed_y = y

    def set_world_pan_speed(self, x, y):
        self.set_screen_pan_speed(self.world_zoom * self.scale * x, self.world_zoom * self.scale * y)

    def set_pannable(self, flag):
        if flag:
            self.set_screen_pan_speed(10, 10)
        else:
            self.set_screen_pan_speed(0, 0)

    def world_pos_is_in_view(self, x, y):
        return x < self.camera_trc_world_x and x >= self.camera_blc_world_x and y < self.camera_trc_world_y and y >= self.camera_blc_world_y

    def set_quit_key(self, key):
        self.quit_key = key

    def set_rendering(self, flag):
        self.doing_rendering = flag

    def set_panning_limits(self, minx=None, maxx=None, miny=None, maxy=None):
        if minx is None:
            minx = -np.inf
        if maxx is None:
            maxx = np.inf
        if miny is None:
            miny = -np.inf
        if maxy is None:
            maxy = np.inf
        self.panning_minx = minx
        self.panning_maxx = maxx
        self.panning_miny = miny
        self.panning_maxy = maxy

    def set_fps(self, fps):
        self.set_frame_rate(fps)

    def random_screen_point(self) -> Tuple[float, float]:
        '''
        Return a random point on the screen in screen coordinates.
        '''
        return PygEnv.random_rect_point(min_x=0, min_y=0, max_x=self.WIDTH, max_y=self.HEIGHT)

    def create_noise(self, freq_x = 0.05, freq_y = 0.05, min_value = 0, max_value = 1, origin_x = None, origin_y = None, seed = None, as_name = None):
        if origin_x is None:
            origin_x = np.random.random() * 20000 - 10000
        if origin_y is None:
            origin_y = np.random.random() * 20000 - 10000
        if freq_x is None:
            freq_x = 1
        if freq_y is None:
            freq_y = 1
        if seed is not None and seed < 0:
            seed = abs(seed)
        if as_name is None:
            as_name = len(self.noises)
        self.noises[as_name] = PerlinNoise(seed=seed), origin_x, origin_y, freq_x, freq_y, min_value, max_value

    def noise(self, name: str, x: float, y: float):
        if name not in self.noises:
            raise ValueError(f'name={name} is not a recognized name of a noise generator. Use create_noise(name={name}, ...) to create a noise generator.')
        noise_gen, origin_x, origin_y, freq_x, freq_y, min_value, max_value = self.noises[name]
        return min_value + (max_value - min_value) * (noise_gen([x * freq_x - origin_x, y * freq_y - origin_y]) + 1) * .5

    def compound_noise(self, noise_indices, x, y):
        return sum(
            self.noise(i, x, y)
            for i in noise_indices
        )

    def noise_int(self, n, x, y):
        return floor(self.noise(n, x, y))

    def compound_noise_int(self, noise_indices, x, y):
        return floor(sum(
            self.noise(i, x, y)
            for i in noise_indices
        ))

    # def random_secant_point(self, secant_distance, secant_angle, circle_radius=1):
    # 	'''
    # 	Return a point that is randomly selected with uniform distribution from the smaller area that's produced by a secant of a circle.  If secant_angle=0 and secant_distance=0, then points are sampled from the "right" semicircle (positive X).  If secant_distance>=circle_radius, an AssertionError is thrown.

    # 	This could be handled in a simpler manner, but the resulting algorithm would be unpredictably slow.  This method ensures uniform distribution with constant time complexity.  Most importantly, this implementation cannot fail, whereas the simpler, naive implementation could possibly fail.
    # 	'''
    # 	assert circle_radius > 0, f'circle_radius={circle_radius} is zero or negative'
    # 	assert secant_distance > 0, f'secant_distance={secant_distance} is zero or negative'
    # 	assert secant_distance < circle_radius, f'secant_distance={secant_distance} is greater than or equal to circle_radius={circle_radius}'
    # 	cr2 = circle_radius * circle_radius
    # 	h = sqrt(cr2 - secant_distance*secant_distance)
    # 	d = circle_radius - secant_distance
    # 	d2 = d*d
    # 	def dist(x):
    # 		v = d*x+secant_distance
    # 		return sqrt(cr2-v*v)/h
    # 	def cumulative_dist(x):
    # 		v = d*x+secant_distance
    # 		return sqrt(cr2-v*v) * \
    # 			(d*v - \
    # 				2*cr2*d * asinh(sqrt((x-1)*0.5*d/circle_radius)) / \
    # 				sqrt((x - 1) * circle_radius*d * (1 + v/circle_radius)) \
    # 			) / (2*a*d2)
    # 	M = cumulative_dist(circle_radius)
    # 	def inv_cumulative_dist(y):
    # 		# JUST APPROXIMATE lol
    # 		# cumulative_dist is always increasing, so the bisection method will work very nicely to get a quick and fairly accurate approximation
    # 		# I modified the bisection method to make it even better (I came up with the modification intuitively, and there's probably a name for this modified version)
    # 		# I think the modification might be an approximation itself of Newton's method, given that the function is always increasing, so this approximation of Newton's method works.
    # 		lb = 0
    # 		ub = circle_radius
    # 		lby = 0
    # 		uby = M
    # 		t = y/M
    # 		for t in range(5):
    # 			m = lb*(1-t)+ub*t
    # 			my = cumulative_dist(m)
    # 			t = (y-lby)/(uby-lby)
    # 			if my < y:
    # 				lb=m
    # 				lby=my
    # 			elif my > y:
    # 				ub=m
    # 				uby=my
    # 			else:
    # 				return m
    # 		return m
    # 	x = inv_cumulative_dist(self.random() * M)
    # 	y = (self.random() * 2 - 1) * dist(x)
    # 	c = cos(secant_angle)
    # 	s = sin(secant_angle)
    # 	return c*x - s*y, s*x + c*y

    def get_pixel_data(self):
        if self.pixel_data_history is None:
            return np.transpose(pyg.surfarray.array3d(self.screen), axes=(1,0,2))
        return self.cur_pixel_data

    def get_pixel_data_history(self, compress = True, indices = None):
        '''
        compress=True will turn the array of shape (length, height, width, 3) into an array of shape (height, width, 3*length).

        If PygEnv.set_pixel_data_history_length(grayscale=True), then (length, height, width, 3) -> (length, height, width, 1), and if compress=True, then etc.

        If indices=None, then the entire history is returned.  Otherwise, it must be a list or tuple of integers specifying the given timesteps in the history that will be returned, where timestep 0 corresponds to the most recently rendered frame, etc.
        '''
        if self.pixel_data_history is None:
            raise Exception('Pixel data history is None.  Before calling PygEnv.get_pixel_data_history(), make sure to initially call PygEnv.set_pixel_data_history_length().')
        l = self.pixel_data_history.shape[0]
        s = self.frame % l
        if compress:
            if indices is None:
                return np.dstack(tuple(self.pixel_data_history[(s-n)%l] for n in range(l)))
            return np.dstack(tuple(self.pixel_data_history[(s-n)%l] for n in indices))
        if indices is None:
            raise NotImplementedError('PygEnv.get_pixel_data_history(compress=False, indices=None) is untested.')
            #return np.roll(self.pixel_data_history, shift=s, axis=0) # THE SHIFT MIGHT BE INCORRECT, NEEDS TESTING
        return self.pixel_data_history[((s-n)%l for n in indices)]

    def set_pixel_data_history_length(self, length, downscaling = 0.25, grayscale = True):
        if length == 0:
            self.pixel_data_history = None
            return
        self.pixel_data_history_downscaling = downscaling
        self.downscaled_size = int(self.WIDTH*self.pixel_data_history_downscaling), int(self.HEIGHT*self.pixel_data_history_downscaling)
        self.pixel_data_history = np.zeros((length, self.downscaled_size[1], self.downscaled_size[0], 3), dtype=int)
        self.pixel_data_is_grayscale = grayscale
    
    def open_pixel_array(self):
        '''
        This locks the surface, so use PygEnv.close_pixel_array() when done.
        '''
        self.pxarray = pyg.PixelArray(self.screen)
    
    def close_pixel_array(self):
        self.pxarray.close()

    def run_loop(self):
        while self.run_step():
            pass

    def run_step(self):
        if self.run:
            self._update()
            return True
        return False

    def quit(self, exit_program=True):
        pyg.quit()
        self.run = False
        if exit_program:
            exit()

    def _update(self):
        self.clock.tick(self.frame_rate)
        if self.using_physics:
            self.pymunk_space.step(self.inv_frame_rate)
            for body in self.pymunk_space.bodies:
                if body.body_type == pym.Body.STATIC:
                    continue
                if body.drag > 0:
                    body.apply_force_at_world_point(body.velocity * -body.drag, body.position)
                    body.torque *= 1 - body.angular_drag
        self.frame += 1
        v = self.current_time_ns
        self.current_time_ns = time_ns()
        self.delta_time = (self.current_time_ns - v) * 1e-9
        if self.do_draw_background:
            self.screen.fill(self.bg_color)
        if self.bg_sprite is not None:
            x = (self.camera_x*self.bg_sprite_parallax) % self.bg_sprite_size[0]
            y = self.bg_sprite_size[1] + (self.camera_y*self.bg_sprite_parallax) % self.bg_sprite_size[1]
            for _y in range(-2,floor(self.HEIGHT/self.bg_sprite_size[1])):
                for _x in range(1+ceil(self.WIDTH/self.bg_sprite_size[0])):
                    self.screen.blit(self.bg_sprite, (_x*self.bg_sprite_size[0]-x, _y*self.bg_sprite_size[1]+y))
        if self.target_camera_x is not None and self.target_camera_y is not None:
            self.translate_camera(self.delta_time * self.target_follow_speed * (self.target_camera_x - self.camera_x), self.delta_time * self.target_follow_speed * (self.target_camera_y - self.camera_y))
        self._update_mouse_info()
        self.update_mouse_info()
        self._pre_update()
        self.pre_update()
        for event in pyg.event.get():
            if event.type == pyg.QUIT:
                self.quit()
            elif event.type == pyg.KEYDOWN:
                if event.key not in self.keys_held:
                    self.keys_held.append(event.key)
                    if event.key == self.double_frame_rate_key:
                        self.set_frame_rate(self.frame_rate*2)
                    elif event.key == self.half_frame_rate_key:
                        self.set_frame_rate(self.frame_rate//2)
                    elif event.key == self.increase_frame_rate_key:
                        self.set_frame_rate(self.frame_rate + 1)
                    elif event.key == self.decrease_frame_rate_key:
                        self.set_frame_rate(self.frame_rate - 1)
                    elif event.key == self.quit_key:
                        self.quit()
                    else:
                        self.key_pressed(event.key)
            elif event.type == pyg.KEYUP:
                if event.key in self.keys_held:
                    self.keys_held.remove(event.key)
                    self.key_released(event.key)
            elif event.type == pyg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._lmb_is_pressed = True
                    self.left_mouse_drag_start_x = self.mouse_pos_x
                    self.left_mouse_drag_start_y = self.mouse_pos_y
                    self._left_mouse_button_pressed()
                    self.left_mouse_button_pressed()
                elif event.button == 2:
                    self._mmb_is_pressed = True
                    self.middle_mouse_drag_start_x = self.mouse_pos_x
                    self.middle_mouse_drag_start_y = self.mouse_pos_y
                    self._middle_mouse_button_pressed()
                    self.middle_mouse_button_pressed()
                elif event.button == 3:
                    self._rmb_is_pressed = True
                    self.right_mouse_drag_start_x = self.mouse_pos_x
                    self.right_mouse_drag_start_y = self.mouse_pos_y
                    self._right_mouse_button_pressed()
                    self.right_mouse_button_pressed()
            elif event.type == pyg.MOUSEBUTTONUP:
                if event.button == 1:
                    self._lmb_is_pressed = False
                    self._left_mouse_button_released()
                    self.left_mouse_button_released()
                elif event.button == 2:
                    self._mmb_is_pressed = False
                    self._middle_mouse_button_released()
                    self.middle_mouse_button_released()
                elif event.button == 3:
                    self._rmb_is_pressed = False
                    self._right_mouse_button_released()
                    self.right_mouse_button_released()
                elif event.button == 4:
                    self.set_world_zoom(self.world_zoom * self.zoom_sensitivity)
                    self._on_mouse_wheel(1)
                    self.on_mouse_wheel(1)
                elif event.button == 5:
                    self.set_world_zoom(self.world_zoom / self.zoom_sensitivity)
                    self._on_mouse_wheel(-1)
                    self.on_mouse_wheel(-1)
            self.ui_manager.process_events(event)
        for key in self.keys_held:
            panned = False
            if key == self.pan_x_pos_key:
                self.translate_camera(self.delta_time * self.pan_speed_x, 0)
                panned = True
            elif key == self.pan_x_neg_key:
                self.translate_camera(self.delta_time * -self.pan_speed_x, 0)
                panned = True
            elif key == self.pan_y_pos_key:
                self.translate_camera(0, self.delta_time * self.pan_speed_y)
                panned = True
            elif key == self.pan_y_neg_key:
                self.translate_camera(0, self.delta_time * -self.pan_speed_y)
                panned = True
            if panned and self.unset_camera_target_on_pan:
                self.unset_camera_target_pos()
            self.key_held(key)
        if self._lmb_is_pressed:
            self._left_mouse_button_held()
            self.left_mouse_button_held()
        if self._mmb_is_pressed:
            self._middle_mouse_button_held()
            self.middle_mouse_button_held()
        if self._rmb_is_pressed:
            self._right_mouse_button_held()
            self.right_mouse_button_held()
        self.__update()
        self.update()
        self.ui_manager.update(self.delta_time)
        if self.doing_rendering:
            self._render()
            self.render()
            if self.using_physics:
                self.render_shapes()
            self.ui_manager.draw_ui(self.screen)
            # for _, label in self.labels.items():
            # 	label.draw()
            pyg.display.update()
            if self.pixel_data_history is not None:
                r = cv2.resize(self.get_pixel_data()/255., self.downscaled_size)
                if self.pixel_data_is_grayscale:
                    r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
                self.pixel_data_history[self.frame % self.pixel_data_history.shape[0]] = np.interp(r, (0, 1), (0, 255)).astype(int)
            self._post_render()
            self.post_render()
            if self.pixel_data_history is not None:
                self.cur_pixel_data = np.transpose(pyg.surfarray.array3d(self.screen), axes=(1,0,2))

    def render_shapes(self):
        for body in self.pymunk_space.bodies:
            for shape in body.shapes:
                if isinstance(shape, pym.Circle):
                    self.draw_world_circle(shape.color, body.position.x, body.position.y, shape.radius, width=0)
                    if shape.draw_line:
                        self.draw_world_line(self.color_modify(shape.color, value=-.3), body.position.x, body.position.y, *self.rotated_point(body.position.x+shape.radius, body.position.y, body.angle, body.position.x, body.position.y), width=2/self.world_zoom)
                elif isinstance(shape, pym.Segment):
                    self.draw_world_line((255, 255, 255), *self.rotated_point(*shape.a, body.angle, body.position.x, body.position.y), *self.rotated_point(*shape.b, body.angle, body.position.x, body.position.y), width=shape.radius)
                # elif isinstance(shape, pym.Poly):
                #     self.draw_world_poly()


        # old_mode = self.rect_draw_mode
        # self.set_rect_draw_mode(RectDrawMode.Center)
        # for shape in self.pymunk_shapes.values():
        #     if not shape.do_rendering:
        #         continue
        #     if isinstance(shape, pym.Poly):
        #         self.draw_world_poly(shape.color, PygEnv.get_world_shape_vertices(shape))
        #     elif isinstance(shape, pym.Circle):
        #         self.draw_world_circle(shape.color, *shape.body.position, shape.radius)
        #     elif isinstance(shape, pym.Segment):
        #         a, b = PygEnv.get_world_shape_vertices(shape)
        #         c = (a + b) * .5
        #         d = a - c
        #         w = d.length * 2
        #         h = shape.radius * 2
        #         self.draw_world_rect(shape.color, *c, w, h, rotation=shape.segment_angle)
        #         self.draw_world_circle(shape.color, *a, shape.radius)
        #         self.draw_world_circle(shape.color, *b, shape.radius)
        # self.set_rect_draw_mode(old_mode)

    def _update_mouse_info(self):
        pass

    def update_mouse_info(self):
        self.last_mouse_pos_x = self.mouse_pos_x
        self.last_mouse_pos_y = self.mouse_pos_y
        self.mouse_pos_x, self.mouse_pos_y = pyg.mouse.get_pos()
        if self.last_mouse_pos_x is None:
            self.mouse_vel_x = 0
            self.mouse_vel_y = 0
        else:
            self.mouse_vel_x = self.mouse_pos_x - self.last_mouse_pos_x
            self.mouse_vel_y = self.mouse_pos_y - self.last_mouse_pos_y
        self.mouse_speed = (self.mouse_vel_x * self.mouse_vel_x + self.mouse_vel_y * self.mouse_vel_y) ** .5
        self.last_mouse_world_pos_x = self.mouse_world_pos_x
        self.last_mouse_world_pos_y = self.mouse_world_pos_y
        self.mouse_world_pos_x = (self.mouse_pos_x - self.HALF_WIDTH) * self.inv_world_zoom + self.camera_x
        self.mouse_world_pos_y = (self.HALF_HEIGHT - self.mouse_pos_y) * self.inv_world_zoom + self.camera_y
        if self.last_mouse_world_pos_x is None:
            self.mouse_world_vel_x = 0
            self.mouse_world_vel_y = 0
        else:
            self.mouse_world_vel_y = self.mouse_world_pos_y - self.last_mouse_world_pos_y
            self.mouse_world_vel_x = self.mouse_world_pos_x - self.last_mouse_world_pos_x
        self.mouse_world_speed = (self.mouse_vel_x * self.mouse_vel_x + self.mouse_vel_y * self.mouse_vel_y) ** .5

    def key_is_held(self, key):
        return key in self.keys_held

    def add_gif_image(self):
        if self.current_gif_images is None:
            self.current_gif_images = []
        self.current_gif_images.append(self.get_pixel_data())

    def save_gif(self, file_name, frame_rate = 60):
        if self.current_gif_images is None:
            raise Exception('PygEnv.current_gif_images is None.  First call PygEnv.add_gif_image() during any frame to save the pixel data as one frame in a GIF.')
        if len(self.current_gif_images) == 0:
            raise Exception('PygEnv.current_gif_images is an empty list.  First call PygEnv.add_gif_image() during any frame to save the pixel data as one frame in a GIF.')
        if '.mp4' != file_name[-4:].lower():
            raise Exception('file_name does not have the .mp4 file extension.')
        vid = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (self.WIDTH, self.HEIGHT))
        for im in self.current_gif_images:
            vid.write(im[:,:,::-1])
        vid.release()

    def _pre_update(self):
        pass

    def pre_update(self):
        pass

    def __update(self):
        pass

    def update(self):
        pass

    def _render(self):
        pass

    def render(self):
        pass

    def _post_render(self):
        pass

    def post_render(self):
        pass

    def key_pressed(self, key):
        pass

    def key_released(self, key):
        pass

    def key_held(self, key):
        pass

    def left_mouse_button_pressed(self):
        pass

    def _left_mouse_button_pressed(self):
        pass

    def middle_mouse_button_pressed(self):
        pass

    def _middle_mouse_button_pressed(self):
        pass

    def right_mouse_button_pressed(self):
        pass

    def _right_mouse_button_pressed(self):
        pass

    def left_mouse_button_released(self):
        pass

    def _left_mouse_button_released(self):
        pass

    def middle_mouse_button_released(self):
        pass

    def _middle_mouse_button_released(self):
        pass

    def right_mouse_button_released(self):
        pass

    def _right_mouse_button_released(self):
        pass

    def left_mouse_button_held(self):
        pass

    def _left_mouse_button_held(self):
        pass

    def middle_mouse_button_held(self):
        pass

    def _middle_mouse_button_held(self):
        pass

    def right_mouse_button_held(self):
        pass

    def _right_mouse_button_held(self):
        pass
    
    def _on_mouse_wheel(self, v):
        pass

    def on_mouse_wheel(self, v):
        pass





class GridEnv(PygEnv):
    def __init__(self, screen_size):
        super().__init__(screen_size = screen_size)
        self.scale = 1
        self.set_scale_limits(16, 256)
        self.set_scale(64)
        self._grid = {}
        self.default_tile_color = (250, 245, 205)
        self.set_pan_controls('wasd')
        self.set_world_pan_speed(1, 1)
        self.set_grid_line_thickness(1)
        self.set_tile_outlines(False)
        self.set_scroll_speed(1.125)
        self.set_dimensions(np.inf, np.inf)

        self.snapped_mouse_grid_pos_x = None
        self.snapped_mouse_grid_pos_y = None
        self.last_snapped_mouse_grid_pos_x = self.snapped_mouse_grid_pos_x
        self.last_snapped_mouse_grid_pos_y = self.snapped_mouse_grid_pos_y
        self.mouse_grid_pos_x_mod = None
        self.mouse_grid_pos_y_mod = None
        self.left_mouse_drag_start_snapped_grid_x = None
        self.left_mouse_drag_start_snapped_grid_y = None
        self.left_mouse_drag_start_grid_x_mod = None
        self.left_mouse_drag_start_grid_y_mod = None
        self.middle_mouse_drag_start_snapped_grid_x = None
        self.middle_mouse_drag_start_snapped_grid_y = None
        self.middle_mouse_drag_start_grid_x_mod = None
        self.middle_mouse_drag_start_grid_y_mod = None
        self.right_mouse_drag_start_snapped_grid_x = None
        self.right_mouse_drag_start_snapped_grid_y = None
        self.right_mouse_drag_start_grid_x_mod = None
        self.right_mouse_drag_start_grid_y_mod = None
        self.tile_offsets_within_radius = {}

    def draw_grid_rect(self, color, x, y, w, h, width = None, alpha = 255):
        if width is None:
            width = self.default_stroke_width
        if self.grid_line_thickness <= 0:
            w,h = w * (self.scale + 1), h * (self.scale + 1)
        else:
            w,h = w * (self.scale - self.grid_line_thickness), h * (self.scale - self.grid_line_thickness)
        self.draw_world_rect(color, x * self.scale, y * self.scale + h - 1, w, h, width = width, alpha = alpha)

    def draw_grid_circle(self, color, x, y, r, width = 0):
        self.draw_world_circle(color, (x + 0.5) * self.scale, (y + 0.5) * self.scale, r * self.scale, width = width)

    def draw_grid_poly(self, color, vertices, width = 0):
        self.draw_world_poly(color, [
            (x * self.scale, y * self.scale)
            for x, y in vertices
        ], width = width)

    def draw_grid_line(self, color, x0, y0, x1, y1, width = 1):
        self.draw_world_line(color, x0 * self.scale, y0 * self.scale, x1 * self.scale, y1 * self.scale, width = width)

    def set_grid_line_color(self, color):
        self.set_bg_color(color)

    def set_scale(self, scale):
        if scale == 0:
            raise ValueError('scale is zero')
        if scale != self.scale:
            self.unset_camera_target_pos()
            s = self.scale
            self.float_scale = min(max(scale, self.min_scale), self.max_scale)
            self.scale = int(self.float_scale)
            s = self.scale / s
            self.set_camera_pos(self.camera_x * s, self.camera_y * s)
            self.set_panning_limits(
                self.panning_minx * s,
                self.panning_maxx * s,
                self.panning_miny * s,
                self.panning_maxy * s
            )

    def set_scale_limits(self, lower, upper):
        self.min_scale = lower
        self.max_scale = upper

    def set_scroll_speed(self, speed):
        self.scroll_speed = speed

    def set_scrollable(self, use_scrolling):
        if use_scrolling:
            if self.scroll_speed == 1:
                self.set_scroll_speed(1.1)
        else:
            self.set_scroll_speed(1)

    def set_tile_outlines(self, use_outlines):
        if use_outlines:
            self.default_stroke_width = 1
        else:
            self.default_stroke_width = 0

    def set_grid_lines(self, flag):
        self.set_grid_line_thickness(int(flag))

    def set_grid_line_thickness(self, t):
        self.grid_line_thickness = t

    def set_camera_grid_pos(self, x, y):
        self.set_camera_pos((x + 1) * self.scale, (y + 1) * self.scale)

    def set_camera_target_grid_pos(self, x, y):
        self.set_camera_target_pos((x + 1) * self.scale, (y + 1) * self.scale)

    def grid_translate_camera(self, x, y):
        self.camera_x = min(max(self.camera_x + x * self.scale, self.panning_minx), self.panning_maxx)
        self.camera_y = min(max(self.camera_y + y * self.scale, self.panning_miny), self.panning_maxy)

    def set_tile_panning_limits(self, minx=None, maxx=None, miny=None, maxy=None):
        if minx is None:
            minx = -np.inf if self.dimension_x_is_infinite else 0
        if maxx is None:
            maxx = np.inf if self.dimension_x_is_infinite else self.dimension_x - 1
        if miny is None:
            minx = -np.inf if self.dimension_y_is_infinite else 0
        if maxy is None:
            maxy = np.inf if self.dimension_y_is_infinite else self.dimension_y - 1
        self.panning_minx = (minx + 0.5) * self.scale
        self.panning_maxx = (maxx - 0.5) * self.scale
        self.panning_miny = (miny + 0.5) * self.scale
        self.panning_maxy = (maxy - 0.5) * self.scale

    def set_dimensions(self, w = 16, h = 16):
        self.dimension_x = w
        self.dimension_y = h
        self.dimension_x_is_infinite = np.isinf(self.dimension_x)
        self.dimension_y_is_infinite = np.isinf(self.dimension_y)

    def lock_camera_to_dimensions(self, flag = True):
        self.camera_is_locked_to_dimensions = flag if flag is not None and type(flag) is bool else False
        if self.camera_is_locked_to_dimensions:
            self.set_panning_limits()

    def center_camera(self):
        self.set_camera_grid_pos(
            0 if self.dimension_x_is_infinite else 0.5 * self.dimension_x - 1,
            0 if self.dimension_y_is_infinite else 0.5 * self.dimension_y - 1
        )

    def set_default_tile_color(self, color):
        self.default_tile_color = color

    def set_full_tile_data(self, x, y, d):
        self._grid[(x, y)] = d

    def get_full_tile_data(self, x, y):
        return self._grid[(x, y)] if (x, y) in self._grid else []

    def set_tile_data(self, x, y, i, v):
        if type(i) is not int:
            raise TypeError('i must be an integer')
        if self.tile_exists(x, y):
            if i >= len(self._grid[(x, y)]):
                self._grid[(x, y)] += [0] * (i - len(self._grid[(x, y)])) + [v]
            else:
                self._grid[(x, y)][i] = v
        else:
            self._grid[(x, y)] = [None] + [0] * (i - 1) + [v] if i >= 1 else [v]

    def get_tile_data(self, x, y, i):
        if type(i) is not int:
            raise TypeError('i must be an integer')
        if self.tile_exists(x, y):
            d = self._grid[(x, y)]
            if i >= len(d) or i < 0:
                return None
            return d[i]
        return None

    def tile_exists(self, x, y):
        return (x, y) in self._grid

    def set_tile_color(self, x, y, color):
        self.set_tile_data(x, y, 1, color)

    def get_tile_color(self, x, y):
        c = self.get_tile_data(x, y, 1)
        if c is None:
            return self.default_tile_color
        return c

    def get_tiles_where(self, minx, maxx, miny, maxy, selector):
        for wy in range(miny, maxy):
            for wx in range(minx, maxx):
                if selector(wx, wy):
                    yield (wx, wy)

    def get_tiles_within_radius(self, radius, center=(0,0), cache=True):
        cx,cy = center
        if cache:
            if radius in self.tile_offsets_within_radius:
                offsets = self.tile_offsets_within_radius[radius][:]
            else:
                offsets = []
                v = int(radius*0.707106781187)
                r2 = radius*radius
                for x in list(range(1, radius)):
                    for y in range(x+1 if x <= v else int((r2-x*x)**.5)+1):
                        offsets.append((x,y))
                        offsets.append((-y,x))
                        offsets.append((-x,-y))
                        offsets.append((y,-x))
                        if y > 0 and y != x:
                            offsets.append((y,x))
                            offsets.append((-x,y))
                            offsets.append((-y,-x))
                            offsets.append((x,-y))
                self.tile_offsets_within_radius[radius] = offsets[:]
            for x,y in offsets:
                yield cx+x,cy+y
        v = int(radius*0.707106781187)
        r2 = radius*radius
        for x in list(range(1, radius)):
            for y in range(x+1 if x <= v else int((r2-x*x)**.5)+1):
                yield (cx+x,cy+y)
                yield (cx-y,cy+x)
                yield (cx-x,cy-y)
                yield (cx+y,cy-x)
                if y > 0 and y != x:
                    yield (cx+y,cy+x)
                    yield (cx-x,cy+y)
                    yield (cx-y,cy-x)
                    yield (cx+x,cy-y)

    def render_range(self, minx, maxx, miny, maxy):
        for wy in range(miny, maxy + 1):
            if not self.dimension_y_is_infinite:
                if wy < 0 or wy >= self.dimension_y:
                    continue
            for wx in range(minx, maxx + 1):
                if not self.dimension_x_is_infinite:
                    if wx < 0 or wx >= self.dimension_x:
                        continue
                d = self.get_full_tile_data(wx, wy)
                l = len(d)
                if l >= 1:
                    if not d[0]:
                        self.on_rendering_new_tile(wx, wy)
                        self.set_tile_data(wx, wy, 0, True)
                else:
                    self.on_rendering_new_tile(wx, wy)
                    self.set_tile_data(wx, wy, 0, True)
                if l >= 2:
                    c = d[1]
                else:
                    c = self.default_tile_color
                self.on_rendering_tile(wx, wy)
                self.draw_grid_rect(c, wx, wy, 1, 1)
                self.on_rendered_tile(wx, wy)

    def _pre_update(self):
        self.camera_grid_lx = floor((self.camera_x - self.HALF_WIDTH) / self.scale) - 1
        self.camera_grid_ux = ceil((self.camera_x + self.HALF_WIDTH) / self.scale) + 1
        self.camera_grid_ly = floor((self.camera_y - self.HALF_HEIGHT) / self.scale) - 1
        self.camera_grid_uy = ceil((self.camera_y + self.HALF_HEIGHT) / self.scale) + 1
        self.mouse_grid_pos_x, self.mouse_grid_pos_y = self.screen_to_grid_space(self.mouse_pos_x, self.mouse_pos_y)
        self.snapped_mouse_grid_pos_x = round(self.mouse_grid_pos_x)
        self.snapped_mouse_grid_pos_y = round(self.mouse_grid_pos_y)
        if not self.dimension_x_is_infinite:
            self.snapped_mouse_grid_pos_x = max(0,min(self.dimension_x - 1, self.snapped_mouse_grid_pos_x))
        if not self.dimension_y_is_infinite:
            self.snapped_mouse_grid_pos_y = max(0,min(self.dimension_y - 1, self.snapped_mouse_grid_pos_y))
        lmgpxm = self.mouse_grid_pos_x_mod
        lmgpym = self.mouse_grid_pos_y_mod
        self.mouse_grid_pos_x_mod = self.mouse_grid_pos_x % 1.0
        self.mouse_grid_pos_y_mod = self.mouse_grid_pos_y % 1.0
        if self.snapped_mouse_grid_pos_x != self.last_snapped_mouse_grid_pos_x or self.snapped_mouse_grid_pos_y != self.last_snapped_mouse_grid_pos_y:
            self.tile_hover_entered_args = (self.snapped_mouse_grid_pos_x, self.snapped_mouse_grid_pos_y, self.mouse_grid_pos_x_mod, self.mouse_grid_pos_y_mod)
            self.tile_hover_stayed_args = (self.snapped_mouse_grid_pos_x, self.snapped_mouse_grid_pos_y, self.mouse_grid_pos_x_mod, self.mouse_grid_pos_y_mod)
            if self.last_snapped_mouse_grid_pos_x is not None and self.last_snapped_mouse_grid_pos_y is not None:
                self.tile_hover_left_args = (self.last_snapped_mouse_grid_pos_x, self.last_snapped_mouse_grid_pos_y, lmgpxm, lmgpym)
            else:
                self.tile_hover_left_args = None
        else:
            self.tile_hover_entered_args = None
            self.tile_hover_left_args = None
            self.tile_hover_stayed_args = (self.snapped_mouse_grid_pos_x, self.snapped_mouse_grid_pos_y, self.mouse_grid_pos_x_mod, self.mouse_grid_pos_y_mod)
        self.last_snapped_mouse_grid_pos_x = self.snapped_mouse_grid_pos_x
        self.last_snapped_mouse_grid_pos_y = self.snapped_mouse_grid_pos_y

    def __update(self):
        pass # it is okay to override this here; derived classes won't (shouldn't) implement this

    def _update_mouse_info(self):
        pass # it is okay to override this here; derived classes won't (shouldn't) implement this

    def _render(self):
        self.render_range(
            self.camera_grid_lx,
            self.camera_grid_ux,
            self.camera_grid_ly,
            self.camera_grid_uy
        )

    def _post_render(self):
        if self.tile_hover_entered_args is not None:
            self._tile_hover_entered(*self.tile_hover_entered_args)
        if self.tile_hover_stayed_args is not None:
            self._tile_hover_stayed(*self.tile_hover_stayed_args)
        if self.tile_hover_left_args is not None:
            self._tile_hover_left(*self.tile_hover_left_args)

    def on_rendering_new_tile(self, x, y):
        pass

    def on_rendering_tile(self, x, y):
        pass

    def on_rendered_tile(self, x, y):
        pass

    def preload_tiles(self, minx, maxx, miny, maxy):
        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                self.preload_tile(x, y)

    def preload_tile(self, x, y):
        pass

    def _on_mouse_wheel(self, v):
        if v > 0:
            self.set_scale(self.float_scale * self.scroll_speed)
        else:
            self.set_scale(self.float_scale / self.scroll_speed)

    def screen_to_grid_space(self, x, y):
        return (x - self.HALF_WIDTH + self.camera_x) / self.scale, (self.HALF_HEIGHT + self.camera_y - y) / self.scale - 1

    def grid_to_screen_space(self, x, y):
        return x * self.scale - self.camera_x + self.HALF_WIDTH, -y * self.scale + self.camera_y + self.HALF_HEIGHT

    def draw_grid_sprite(self, sprite_sheet_name, x, y, row, col, flip=False, scale=1, rotation=0):
        '''
        `rotation` is in radians.
        '''
        self.draw_world_sprite(sprite_sheet_name, x*self.scale, y*self.scale, row, col, flip, scale, rotation)

    def _left_mouse_button_pressed(self):
        self.left_mouse_drag_start_snapped_grid_x = self.snapped_mouse_grid_pos_x
        self.left_mouse_drag_start_snapped_grid_y = self.snapped_mouse_grid_pos_y
        self.left_mouse_drag_start_grid_x_mod = self.mouse_grid_pos_x_mod
        self.left_mouse_drag_start_grid_y_mod = self.mouse_grid_pos_y_mod
        self.left_mouse_button_pressed_on_tile(self.snapped_mouse_grid_pos_x, self.snapped_mouse_grid_pos_y, self.mouse_grid_pos_x_mod, self.mouse_grid_pos_y_mod)

    def _left_mouse_button_held(self):
        self.left_mouse_button_held_on_tile(self.snapped_mouse_grid_pos_x, self.snapped_mouse_grid_pos_y, self.mouse_grid_pos_x_mod, self.mouse_grid_pos_y_mod)

    def _left_mouse_button_released(self):
        self.left_mouse_button_released_on_tile(self.snapped_mouse_grid_pos_x, self.snapped_mouse_grid_pos_y, self.mouse_grid_pos_x_mod, self.mouse_grid_pos_y_mod)
    
    def _middle_mouse_button_pressed(self):
        self.middle_mouse_drag_start_snapped_grid_x = self.snapped_mouse_grid_pos_x
        self.middle_mouse_drag_start_snapped_grid_y = self.snapped_mouse_grid_pos_y
        self.middle_mouse_drag_start_grid_x_mod = self.mouse_grid_pos_x_mod
        self.middle_mouse_drag_start_grid_y_mod = self.mouse_grid_pos_y_mod
        self.middle_mouse_button_pressed_on_tile(self.snapped_mouse_grid_pos_x, self.snapped_mouse_grid_pos_y, self.mouse_grid_pos_x_mod, self.mouse_grid_pos_y_mod)

    def _middle_mouse_button_held(self):
        self.middle_mouse_button_held_on_tile(self.snapped_mouse_grid_pos_x, self.snapped_mouse_grid_pos_y, self.mouse_grid_pos_x_mod, self.mouse_grid_pos_y_mod)

    def _middle_mouse_button_released(self):
        self.middle_mouse_button_released_on_tile(self.snapped_mouse_grid_pos_x, self.snapped_mouse_grid_pos_y, self.mouse_grid_pos_x_mod, self.mouse_grid_pos_y_mod)
    
    def _right_mouse_button_pressed(self):
        self.right_mouse_drag_start_snapped_grid_x = self.snapped_mouse_grid_pos_x
        self.right_mouse_drag_start_snapped_grid_y = self.snapped_mouse_grid_pos_y
        self.right_mouse_drag_start_grid_x_mod = self.mouse_grid_pos_x_mod
        self.right_mouse_drag_start_grid_y_mod = self.mouse_grid_pos_y_mod
        self.right_mouse_button_pressed_on_tile(self.snapped_mouse_grid_pos_x, self.snapped_mouse_grid_pos_y, self.mouse_grid_pos_x_mod, self.mouse_grid_pos_y_mod)

    def _right_mouse_button_held(self):
        self.right_mouse_button_held_on_tile(self.snapped_mouse_grid_pos_x, self.snapped_mouse_grid_pos_y, self.mouse_grid_pos_x_mod, self.mouse_grid_pos_y_mod)

    def _right_mouse_button_released(self):
        self.right_mouse_button_released_on_tile(self.snapped_mouse_grid_pos_x, self.snapped_mouse_grid_pos_y, self.mouse_grid_pos_x_mod, self.mouse_grid_pos_y_mod)

    def _tile_hover_entered(self, x, y, u, v):
        self.tile_hover_entered(x, y, u, v)

    def _tile_hover_stayed(self, x, y, u, v):
        self.tile_hover_stayed(x, y, u, v)

    def _tile_hover_left(self, x, y, u, v):
        self.tile_hover_left(x, y, u, v)

    def left_mouse_button_pressed_on_tile(self, x, y, u, v):
        pass

    def left_mouse_button_held_on_tile(self, x, y, u, v):
        pass

    def left_mouse_button_released_on_tile(self, x, y, u, v):
        pass

    def middle_mouse_button_pressed_on_tile(self, x, y, u, v):
        pass

    def middle_mouse_button_held_on_tile(self, x, y, u, v):
        pass

    def middle_mouse_button_released_on_tile(self, x, y, u, v):
        pass

    def right_mouse_button_pressed_on_tile(self, x, y, u, v):
        pass

    def right_mouse_button_held_on_tile(self, x, y, u, v):
        pass

    def right_mouse_button_released_on_tile(self, x, y, u, v):
        pass

    def tile_hover_entered(self, x, y, u, v):
        pass

    def tile_hover_stayed(self, x, y, u, v):
        pass

    def tile_hover_left(self, x, y, u, v):
        pass





class SpriteSheet:
    def __init__(self, file, rows, cols, margin=0, pixels_per_unit=None):
        self.file = file
        self.rows = rows
        self.cols = cols
        self.margin = margin
        self.sprite_surfaces = []
        try:
            self.surface = pyg.image.load(file).convert_alpha()
            self.image_size = ((self.surface.get_width()-(self.cols-1)*self.margin)//self.cols, (self.surface.get_height()-(self.rows-1)*self.margin)//self.rows)
            if pixels_per_unit is None:
                pixels_per_unit = self.image_size[0]
            self.pixels_per_unit = pixels_per_unit
            for y in range(rows):
                for x in range(cols):
                    surf = pyg.Surface(self.image_size, pyg.SRCALPHA)
                    surf.blit(self.surface, (0, 0), pyg.Rect(x*(self.image_size[0]+self.margin), y*(self.image_size[1]+self.margin), *self.image_size))
                    surf.convert_alpha()
                    self.sprite_surfaces.append(surf)
            self.half_image_size = self.image_size[0] * .5, self.image_size[1] * .5
        except:
            raise Exception(f'Could not read file \'{file}\' to load sprite sheet.')
    @lru_cache(maxsize=1024)
    def get_sprite_surface(self, row, col, flip, total_scale):
        surf = self.sprite_surfaces[row*self.cols+col]
        if flip:
            surf = pyg.transform.flip(surf)
        if total_scale != 1:
            surf = pyg.transform.scale(surf, (self.image_size[0]*total_scale, self.image_size[1]*total_scale))
        return surf
    def blit(self, surface, x, y, from_row, from_column, flip, scale, rotation, pivot_point, tiling_x, tiling_y, tiling_scale):
        '''
        `rotation` is in radians.
        '''
        if self.surface is None:
            return
        surf = self.get_sprite_surface(from_row, from_column, flip, scale * tiling_scale)
        if rotation != 0:
            if pivot_point is not None:
                px, py = pivot_point
                rpx, rpy = PygEnv.rotated_point(px, py, -rotation, *self.half_image_size)
            surf = pyg.transform.rotate(surf, rotation * PygEnv.RAD_2_DEG)
            if tiling_x == 1 and tiling_y == 1:
                _x = x - (surf.get_width()-self.image_size[0]*scale)*.5
                _y = y - (surf.get_height()-self.image_size[1]*scale)*.5
                if pivot_point is not None:
                    _x += (self.half_image_size[0] - rpx) * scale
                    _y += (self.half_image_size[1] - rpy) * scale
                surface.blit(surf, (_x, _y))
            else:
                tile_width, tile_height = surf.get_width() / tiling_scale, surf.get_height() / tiling_scale
                ox = x-(tile_width-self.image_size[0]*scale)*.5
                oy = y-(tile_height-self.image_size[1]*scale)*.5
                mx = scale * tile_width
                my = scale * tile_height
                for ty in range(tiling_y):
                    for tx in range(tiling_x):
                        surface.blit(surf, (ox + tx * mx, oy * ty * my))
        else:
            if tiling_x == 1 and tiling_y == 1:
                if pivot_point is not None:
                    px, py = pivot_point
                else:
                    px, py = self.half_image_size
                surface.blit(surf, (x + self.half_image_size[0] - px, y + self.half_image_size[1] - py))
                # surface.blit(surf, (x, y))
            else:
                w, h = self.image_size
                mx = scale * w
                my = scale * h
                for ty in range(tiling_y):
                    for tx in range(tiling_x):
                        surface.blit(surf, (x + tx * mx, y + ty * my))


