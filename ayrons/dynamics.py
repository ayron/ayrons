"""OO Multi Body Dynamics
"""

import scipy.linalg as la
import numpy as np

from numpy import dot, array, sin, cos
from scipy.integrate import ode
from scipy.optimize import minimize
from scipy.linalg import norm

import pyglet
import pyglet.gl as gl

np.set_printoptions(precision=2)

# Physical constants
g = 4.81    # m/s/s


def R(a):
    return array([[ cos(a), -sin(a)],
                  [ sin(a),  cos(a)]])


def dR(a):
    return array([[-sin(a), -cos(a)],
                  [ cos(a), -sin(a)]])


def get_indices(iterator):
    ns = [0]
    for item in iterator:
        ns.append(item.size + ns[-1])
    return ns


class System(object):
    """Class for construction of a system of linear equations."""

    def __init__(self):

        self.rigid_bodies = []
        self.constraints  = []

    def initialize(self):

        # Calculate indices
        self.indices = [0]
        for item in (self.rigid_bodies + self.constraints):
            self.indices.append(item.size + self.indices[-1])

        # Initial matrix
        self.A = np.zeros((self.indices[-1], self.indices[-1]))
        self.b = np.zeros(self.indices[-1])

        # Initialize the rigid bodies and contraints
        for i, item in zip(self.indices, self.rigid_bodies + self.constraints):
            item.initialize(self, i)

        # Initial Conditions
        self.t0    = 0.0
        self.state = self.generate_initial_conditions()

        self.propogator = ode(self.dynamic_system).set_integrator('vode', method='adams')
        self.propogator.set_initial_value(self.state, self.t0)

    def generate_initial_conditions(self):
        """Collect the initial conditions provided by the user,
        and constrain the given the constraints."""

        # Collect ICs
        ics = array([rb.ics for rb in self.rigid_bodies], np.float64)
        ics = np.hstack((ics[:,0:3].flatten(), ics[:,3:6].flatten()))

        # Constrain
        def error(state):

            error = sum(c.error(state) for c in self.constraints)
            return error

        res = minimize(error, ics, method='Nelder-Mead')
        print res
        return res.x


    def dynamic_system(self, t, state):
        """Solve the dynamic system."""

        ps = state[:len(state)/2]
        vs = state[len(state)/2:]

        # y  ~ [p1, p2, v1, v2]
        # dy ~ [v1, v2, a1, a2]

        for i, c in enumerate(self.constraints):

            c.update_system(self, state)

        # Make A symetrical
        D = self.A + self.A.T - np.diag(self.A.diagonal())

        x = la.solve(D, self.b)
        dy = np.hstack((vs, x[:3*len(self.rigid_bodies)]))

        return dy

    def propogate(self, dt):
        """Propogate the system forward in time."""

        self.state = self.propogator.integrate(self.propogator.t + dt)


class RigidBody(object):

    def __init__(self, mass=1, moi=1,
                 ics=6*[0]):

        self.size = 3

        self.m = mass  # kg
        self.I = moi   # kgm^2
        self.ics = ics

        # Mass matrix
        self.M = np.array([[self.m, 0, 0],
                           [0, self.m, 0],
                           [0, 0, self.I]], np.float64)

    def initialize(self, system, i):

        self.rb_i = i
        self.system = system

        s = self.size
        system.A[i:i+s, i:i+s] = self.M

        # Add gravity
        system.b[i+1] = -g*self.m


class Rectangle(RigidBody):

    def __init__(self, width=1, height=1, **kwargs):

        mass = kwargs['mass']
        moi = (mass/12.0)*(height**2 + width**2)
        super(Rectangle, self).__init__(moi=moi, **kwargs)

        ps = ( 0.5*width,  0.5*height,
              -0.5*width,  0.5*height,
              -0.5*width, -0.5*height,
               0.5*width, -0.5*height)
        cs = (255, 255, 255)
        self.vl = pyglet.graphics.vertex_list(4, ('v2f', ps), ('c3B', 4*cs))

    def draw(self):

        gl.glLoadIdentity()
        x, y, h = self.system.state[self.rb_i:self.rb_i+3]
        gl.glTranslatef(x, y, 0)
        gl.glRotatef(np.rad2deg(h), 0, 0, 1)
        self.vl.draw(pyglet.gl.GL_LINE_LOOP)


class Constraint(object):

    def update_system(self, system, state):

        pass

    def draw(self):

        pass

    def error(self, state):

        return 0


class Rigid(Constraint):

    def __init__(self, rigid_body):

        self.rigid_body = rigid_body
        self.window = None

        self.size = 3

    def initialize(self, system, i):

        # Find index of rigib body
        rb_i = system.indices[system.rigid_bodies.index(self.rigid_body)]

        # Create a slice
        self.window = system.A[rb_i:rb_i+self.size, i:i+self.size]
        self.window[:] = np.eye(3)


class Pin(Constraint):

    def __init__(self, rb1, p1, rb2, p2):

        self.size = 2
        self.p1 = p1
        self.p2 = p2
        self.rb1 = rb1
        self.rb2 = rb2

        # Drawing Stuff
        cs = (0, 255, 0)
        N = 12
        r = 0.1
        angles = np.arange(N)*2*np.pi/N
        xs = r*cos(angles)
        ys = r*sin(angles)
        ps = np.c_[xs, ys].flatten()

        self.vl = pyglet.graphics.vertex_list(N, ('v2f', ps), ('c3B', N*cs))


    def draw(self):

        gl.glLoadIdentity()
        pc = self.system.state[self.rb1_i:self.rb1_i+2]
        h  = self.system.state[self.rb1_i+2]

        p = pc + dot(R(h), self.p1)
        gl.glTranslatef(p[0], p[1], 0)
        self.vl.draw(pyglet.gl.GL_LINE_LOOP)

        gl.glLoadIdentity()
        pc = self.system.state[self.rb2_i:self.rb2_i+2]
        h  = self.system.state[self.rb2_i+2]

        p = pc + dot(R(h), self.p2)
        gl.glTranslatef(p[0], p[1], 0)
        self.vl.draw(pyglet.gl.GL_LINE_LOOP)


    def initialize(self, system, i):

        self.system = system

        # Find rb indices
        self.rb1_i = system.indices[system.rigid_bodies.index(self.rb1)]
        self.rb2_i = system.indices[system.rigid_bodies.index(self.rb2)]

        self.w1 = system.A[self.rb1_i:self.rb1_i+3, i:i+self.size]
        self.w2 = system.A[self.rb2_i:self.rb2_i+3, i:i+self.size]
        self.w3 = system.b[i:i+self.size]

    def update_system(self, system, state):

        theta1 = state[self.rb1_i+2]
        theta2 = state[self.rb2_i+2]

        dtheta1 = state[self.rb1_i+2 + 3*len(system.rigid_bodies)]
        dtheta2 = state[self.rb2_i+2 + 3*len(system.rigid_bodies)]

        self.w1[:] =  np.vstack(( np.eye(2),
                               dot(dR(theta1), self.p1) ))
        self.w2[:] = -np.vstack(( np.eye(2),
                               dot(dR(theta2), self.p2) ))

        self.w3[:] = dtheta1**2*dot(R(theta1), self.p1) - dtheta2**2*dot(R(theta2), self.p2)

    def error(self, state):

        pc1 = state[self.rb1_i:self.rb1_i+2]
        h1  = state[self.rb1_i+2]

        pc2 = state[self.rb2_i:self.rb2_i+2]
        h2  = state[self.rb2_i+2]

        error = norm(pc1 + dot(R(h1), self.p1) - pc2 - dot(R(h2), self.p2))
        return error
