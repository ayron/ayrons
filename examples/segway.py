"""Examples of using ayron to simulate and visualize a double pendulum."""


from ayrons.dynamics import System, Rectangle, Circle, Pin, Rolling
from ayrons.visualization import Visualizer

import numpy as np

'''
class Motor(Moment):

    def moment(self, t, state):

        F_x = np.cos(t)
        F_y = np.sin(t)

        return F_x, F_y

class Accelerometer(Sensor):

    pass

class MyController(Controller):

    pass
'''

if __name__ == '__main__':

    # Rigid bodies
    wheel = Circle(radius=1.0,
                   mass=1.0)
    arm   = Rectangle(width=5.0,
                      height=0.5,
                      mass=1.0,
                      ics=[0.0, 0.0, 0.4*3.14, 0, 0, 0])

    # Constraints
    roll = Rolling(wheel)
    pin  = Pin(wheel, (0.0, 0.0),
               arm,   (-2.5, 0.0))

    # Dynamic System
    double_pendulum = System()
    double_pendulum.rigid_bodies = [wheel, arm]
    double_pendulum.constraints  = [roll, pin]

    double_pendulum.initialize()

    # Visualization
    vis = Visualizer(double_pendulum)
    vis.run()
