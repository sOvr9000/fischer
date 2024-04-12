

import numpy as np
import cv2
from .lsys import LSystem
from colorsys import hsv_to_rgb

from fischer.imgpostproc import diffuse_step


__all__ = ['render_cv2']



def render_cv2(s: str, scale: float = 0.95, rot: float = 0.0, turns_per_360: float = 4.0, img_size: tuple[int, int] = (960, 960), save_fname: str = None) -> None:
    '''
    "-" | Forward

    "A" | Right

    "B" | Left

    "[" | Push Position & Direction

    "]" | Pop Position & Direction

    "<" | Read back from beginning of string only once, skip this character next time
    '''
    position_stack = []
    visited = []
    turn_delta = 2. * np.pi / turns_per_360
    x = 0.
    y = 0.
    d = rot
    points = np.zeros((len(s) + 1, 2))
    h, w = img_size
    hh = h * .5
    hw = w * .5
    restart = True
    while restart:
        restart = False
        for i, c in enumerate(s):
            if c == '-':
                x += np.cos(d)
                y += np.sin(d)
            elif c == 'A':
                d += turn_delta
            elif c == 'B':
                d -= turn_delta
            elif c == '[':
                position_stack.append((x, y, d))
            elif c == ']':
                x, y, d = position_stack.pop()
            elif c == '<':
                if i not in visited:
                    visited.append(i)
                    restart = True
            points[i+1] = x, y
    M = np.max(points, axis=0)
    m = np.min(points, axis=0)
    cx, cy = (M + m) * .5
    # r = h * scale / max(*(M - m))
    dm = M - m
    r = scale * min(w / dm[0], h / dm[1])
    img = np.zeros((*img_size, 3))
    for i, ((px, py), (x, y)) in enumerate(zip(points[:-1], points[1:])):
        if px != x or py != y:
            c = hsv_to_rgb(i / points.shape[0], 1, 1)
            img = cv2.line(img, (int((px - cx) * r + hw), int((py - cy) * r + hh)), (int((x - cx) * r + hw), int((y - cy) * r + hh)), c, thickness=1)
    
    # # diffusion post-processing
    # for _ in range(24):
    #     img = diffuse_step(img)

    if save_fname is None:
        cv2.imshow('L-System Render', img)
        cv2.waitKey()
    else:
        # img = np.repeat(img[:, :, np.newaxis], 3, 2) * 255.
        img *= 255.
        cv2.imwrite(save_fname, img)


if __name__ == '__main__':
    # sys = LSystem(seed='AAAA', patterns={
    #     'A': 'CA',
    #     'B': 'BC',
    #     'C': 'ACB-',
    # })
    # turns_per_360 = 4
    sys = LSystem(seed='D', patterns={
        'A': ']DB',
        'B': 'CA<',
        'C': 'BC[D',
        'D': 'C-A'
    })
    turns_per_360 = 6
    for _ in range(128):
        if len(sys) >= 65536: break
        sys.step()
    render_cv2(sys.current_str, img_size=(1024, 1024), save_fname='./test_render.png', turns_per_360=turns_per_360)



