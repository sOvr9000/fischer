
from math import cos, sin



class Pendulum:
    '''
    Double pendulum simulation.  Both `Pendulum.theta1` and `Pendulum.theta2` are zero when the pendulum is at rest (pointing downward, etc.).
    '''
    def __init__(self, theta1 = 1, theta2 = 0, length1 = 1, length2 = 1, mass1 = 1, mass2 = 1, gravity = 1, friction = 0):
        self.gravity = gravity
        self.length1 = length1
        self.length2 = length2
        self.l1s = self.length1 * self.length1
        self.l2s = self.length2 * self.length2
        self.l12l2 = self.length2 * self.l1s
        self.mass1 = mass1
        self.mass2 = mass2
        self.l1l2 = self.length1 * self.length2
        self.l2m2 = self.length2 * self.mass2
        self.l1l2m2 = self.l1l2 * self.mass2
        self.l1l22m2 = self.l1l2m2 * self.length2
        self.l2sm2 = self.l2s * self.mass2
        self.sm = self.mass1 + self.mass2
        self.l1sm = self.length1 * self.sm
        self.l1ssm = self.l1sm * self.length1
        self.l1smg = self.l1sm * self.gravity
        self.l2m2g = self.l2m2 * self.gravity
        self.friction = friction # XXX Currently does nothing
        self.theta1 = theta1
        self.theta2 = theta2
        self.pt1 = 0
        self.pt2 = 0
        self.dtheta1 = 0
        self.dtheta2 = 0
        self.dpt1 = 0
        self.dpt2 = 0
    def step(self, dt=0.01):
        # https://scienceworld.wolfram.com/physics/DoublePendulum.html
        c = cos(self.theta1 - self.theta2)
        s = sin(self.theta1 - self.theta2)
        s2 = s * s
        d = self.mass1 + self.mass2 * s2
        self.dtheta1 = (self.length2 * self.pt1 - self.length1 * self.pt2 * c) / (self.l12l2 * d)
        self.dtheta2 = (self.l1sm * self.pt2 - self.l2m2 * c * self.pt1) / (self.l1l22m2 * d)
        p = self.pt1 * self.pt2
        v = 1 / (self.l1l2 * d)
        v = p * s * v - (self.l2sm2 * self.pt1 * self.pt1 + self.l1ssm * self.pt2 * self.pt2 - self.l1l2m2 * self.pt1 * self.pt2 * c) * s * c / (v * v)
        self.dpt1 = -v - sin(self.theta1) * self.l1smg
        self.dpt2 = v - sin(self.theta2) * self.l2m2g
        self.theta1 += self.dtheta1 * dt
        self.theta2 += self.dtheta2 * dt
        self.pt1 += self.dpt1 * dt
        self.pt2 += self.dpt2 * dt




def main():
    pend = Pendulum()
    for _ in range(50):
        pend.step(dt=0.1)
        print(pend.theta1, pend.theta2)


if __name__ == '__main__':
    main()
