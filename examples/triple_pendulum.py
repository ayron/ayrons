"""Examples of using ayron to simulate and visualize a double pendulum."""


from ayrons.dynamics import System, Rectangle, Rigid, Pin
from ayrons.visualization import Visualizer


if __name__ == '__main__':

    # Rigid bodies
    base = Rectangle(width=2.0,
                     height=1.0,
                     mass=1.0)
    arm1 = Rectangle(width=5.0,
                     height=0.5,
                     mass=1.0,
                     ics=[1.0, 2.0, 3.14/4, 0, 0, 0])
    arm2 = Rectangle(width=5.0,
                     height=0.5,
                     mass=1.0,
                     ics=[3.0, 4.0, -3.14/2, 0, 0, 0])
    arm3 = Rectangle(width=5.0,
                     height=0.5,
                     mass=1.0,
                     ics=[3.0, 4.0, -3.14, 0, 0, 0])

    # Constraints
    ground = Rigid(base)
    pin1   = Pin(base, (0.0, 0.0),
                 arm1, (-2.5, 0.0))
    pin2   = Pin(arm1, (2.5, 0.0),
                 arm2, (-2.5, 0.0))
    pin3   = Pin(arm2, (2.5, 0.0),
                 arm3, (-2.5, 0.0))

    # Dynamic System
    double_pendulum = System()
    double_pendulum.rigid_bodies = [base, arm1, arm2, arm3]
    double_pendulum.constraints  = [ground, pin1, pin2, pin3]
    print(double_pendulum)

    double_pendulum.initialize()

    # Visualization
    vis = Visualizer(double_pendulum)
    vis.run()
