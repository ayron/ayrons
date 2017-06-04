"""Examples of using ayron to simulate and visualize a double pendulum."""


from ayrons.dynamics import System, Rectangle, Rigid, Pin
from ayrons.visualization import Visualizer


if __name__ == '__main__':

    # Rigid bodies
    base1 = Rectangle(width=2.0,
                      height=1.0,
                      mass=1.0)

    arm1 = Rectangle(width=5.0,
                     height=0.5,
                     mass=1.0,
                     ics=[0.0, -2.5, -3.14/2, 0, 0, 0])

    arm2 = Rectangle(width=5.0,
                     height=0.5,
                     mass=1.0,
                     ics=[3, -2.5, 3.14/2, 0, 0, 0])

    arm3 = Rectangle(width=5.0,
                     height=0.5,
                     mass=1.0,
                     ics=[0.0, -3, 0.0, 0, 0, 0])

    arm4 = Rectangle(width=5.0,
                     height=0.5,
                     mass=1.0,
                     ics=[4.0, -3, 0.0, 0, 0, 0])

    arm5 = Rectangle(width=5.0,
                     height=0.5,
                     mass=1.0,
                     ics=[4.0, -3, 0.0, 0, 0, 0])




    # Constraints
    ground1 = Rigid(base1)

    pin1   = Pin(base1, (0.0, 0.0),
                 arm1, (-2.5, 0.0))
    pin2   = Pin(arm1, (2.5, 0.0),
                 arm2, (-2.5, 0.0))
    pin3   = Pin(arm2, (2.5, 0.0),
                 arm3, (-2.5, 0.0))
    pin4   = Pin(arm2, (1.0, 0.0),
                 arm4, (-2.5, 0.0))
    pin5   = Pin(arm1, (1.0, 0.0),
                 arm5, (-2.5, 0.0))
    pin6   = Pin(arm5, (2.5, 0.0),
                 arm3, (2.5, 0.0))


    # Dynamic System
    double_pendulum = System()
    double_pendulum.rigid_bodies = [base1, arm1, arm2, arm3, arm4, arm5]
    double_pendulum.constraints  = [ground1, pin1, pin2, pin3, pin4, pin5, pin6]

    double_pendulum.initialize()

    # Visualization
    vis = Visualizer(double_pendulum)
    vis.run()
