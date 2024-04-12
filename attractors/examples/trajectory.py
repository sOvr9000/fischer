
import numpy as np
from fischer.attractors import *
from fischer.trajplot import trajectory_image, trajectory_mp

import colorcet as cc



def main():
    trajectory_image(
        traj=trajectory_mp(
            f=portrait,
            iterations_per_worker=1618034,
            workers=14,
            batches=16,
            clip_std=3,
        ),
        width=2560,
        height=1440,
        cmap=cc.kbc,
        rotation=-np.pi/2,
    )



if __name__ == '__main__':
    main()
