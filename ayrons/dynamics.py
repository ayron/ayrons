"""OO Multi Body Dynamics
"""

import scipy.linalg as la
import numpy as np
import sympy as sp

from numpy import dot, array, sin, cos
from scipy.integrate import ode
from scipy.optimize import minimize
from scipy.linalg import norm
from sympy.physics.vector import dynamicsymbols

import pyglet
import pyglet.gl as gl

from time import time

np.set_printoptions(precision=2)

# Physical constants
g_value = 4.81    # m/s/s


def R(a):
    return sp.Matrix([[ sp.cos(a), -sp.sin(a)],
                      [ sp.sin(a),  sp.cos(a)]])


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

    def __str__(self):
        return 'System of {} rigid bodies and {} constraints' \
            ''.format(len(self.rigid_bodies), len(self.constraints))

    def initialize(self):
        """Derive the system of dynamic equations."""

        N = len(self.rigid_bodies)
        C = len(self.constraints)

        print('Provided with {} rigid bodies'.format(N))
        print('Provided with {} constraints'.format(N))

        # Create algebiac symbols
        t, g = sp.symbols('t g')

        ms = sp.symbols('m_0:{}'.format(N))
        Is = sp.symbols('I_0:{}'.format(N))

        vx = dynamicsymbols('vy_0:{}'.format(N))
        vy = dynamicsymbols('vx_0:{}'.format(N))
        vh = dynamicsymbols('w_0:{}'.format(N))

        xs = dynamicsymbols('x_0:{}'.format(N))
        ys = dynamicsymbols('y_0:{}'.format(N))
        hs = dynamicsymbols('h_0:{}'.format(N))

        # Take first and second derivative
        dxs = [sp.diff(x, t) for x in xs]
        dys = [sp.diff(y, t) for y in ys]
        dhs = [sp.diff(h, t) for h in hs]

        ddxs = [sp.diff(x, t, t) for x in xs]
        ddys = [sp.diff(y, t, t) for y in ys]
        ddhs = [sp.diff(h, t, t) for h in hs]

        # The Lagrangian
        L = sum([sp.S(1)/2*m*(dx**2 + dy**2) + sp.S(1)/2*I*dh**2 - m*g*y
                 for m, I, dx, dy, dh, y
                 in zip(ms, Is, dxs, dys, dhs, ys)])

        #print('The Lagrangian is:')
        #sp.pprint(L)

        # Assign symbols to rigid_bodies
        for rb, x, y, h in zip(self.rigid_bodies, xs, ys, hs):
            rb.symbols = [x, y, h]


        # Get all constraint equations
        Ceqs = [e
                for c
                in self.constraints
                for e
                in c.equations2()]

        #print('Constraint Equations:')
        #for e in Ceqs:
        #    sp.pprint(e)

        # Lagrangian Multipliers
        As = dynamicsymbols('A_0:{}'.format(len(Ceqs)))

        # Use to ensure constaints are met during the simulation
        self.constraint_equations = sp.lambdify(xs+ys+hs, Ceqs, 'numpy')

        ddCeqs = [sp.diff(e, t, t) for e in Ceqs]

        # Sum of contraint equations with lagrange multipliers
        K = sum([A*e for A, e in zip(As, Ceqs)])

        # External forces
        # TODO
        Q = 0

        # Lagrange's Equations
        leq = [sp.diff(L + K, s) - sp.diff(L, ds, t) + Q
               for s, ds in zip(xs + ys + hs, dxs + dys + dhs)]

        system = sp.Matrix(leq + ddCeqs)
        uks = sp.Matrix(ddxs + ddys + ddhs + As)

        # Create a matrix form for the equations
        A = system.jacobian(uks)
        b = A*uks-system


        # Lambdify does not handle general expressions as inputs,
        # such as dx/dt, thus we substitute speed for these values
        A = A.subs(dict(zip(dxs+dys+dhs, vx+vy+vh)))
        b = b.subs(dict(zip(dxs+dys+dhs, vx+vy+vh)))

        #sp.pprint(A)
        #sp.pprint(b)

        # Subsitute real values
        masses = [rb.m for rb in self.rigid_bodies]
        mois   = [rb.I for rb in self.rigid_bodies]
        A = A.subs(dict(zip(ms, masses)))
        A = A.subs(dict(zip(Is, mois)))
        A = A.subs(g, 9.81)

        b = b.subs(dict(zip(ms, masses)))
        b = b.subs(dict(zip(Is, mois)))
        b = b.subs(g, 9.81)

        # This system of questions ready for numpy to solve
        self.A_f = sp.lambdify(xs+ys+hs+vx+vy+vh, A, 'numpy')
        self.b_f = sp.lambdify(xs+ys+hs+vx+vy+vh, b, 'numpy')

        from inspect import signature

        # Initial Conditions
        self.t0    = 0.0
        self.state = self.generate_initial_conditions()

        self.propogator = ode(self.dynamic_system).set_integrator('vode', method='adams')
        self.propogator.set_initial_value(self.state, self.t0)


    def generate_initial_conditions(self):
        """Collect the initial conditions provided by the user,
        and constrain the given the constraints."""

        # Collect ICs
        ics = array([rb.state for rb in self.rigid_bodies], np.float64)
        ics = ics.T.flatten()

        # Constrain
        def error(state):

            errors = self.constraint_equations(*state)
            return np.dot(errors, errors)

        res = minimize(error, ics[:len(ics)/2], method='Nelder-Mead')
        ics[:len(ics)/2] = res.x

        return ics


    def dynamic_system(self, t, state):
        """Solve the dynamic system."""

        ps = state[:len(state)/2]
        vs = state[len(state)/2:]

        # y  ~ [p1, p2, v1, v2]
        # dy ~ [v1, v2, a1, a2]

        solution = np.linalg.solve(self.A_f(*state), self.b_f(*state))
        accel    = solution[:len(ps),0]

        dstate = np.hstack((vs, accel))

        return dstate

    def propogate(self, dt):
        """Propogate the system forward in time."""

        self.state = self.propogator.integrate(self.propogator.t + dt)

        # Copy state to rigid bodies
        N = len(self.rigid_bodies)
        for i, rb in enumerate(self.rigid_bodies):
            rb.state = self.state[i::N]


class RigidBody(object):

    def __init__(self, mass=1, moi=1, ics=6*[0]):

        self.size = 3

        self.m = mass  # kg
        self.I = moi   # kgm^2
        self.state = ics

        # Mass matrix
        self.M = np.array([[self.m, 0, 0],
                           [0, self.m, 0],
                           [0, 0, self.I]], np.float64)

        self.symbols = dynamicsymbols('x y h')

    @property
    def x(self):
        return self.symbols[0]

    @property
    def y(self):
        return self.symbols[1]

    @property
    def h(self):
        return self.symbols[2]

    @property
    def p(self):
        return sp.Matrix([[self.symbols[0]],
                          [self.symbols[1]]])



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

        x, y, h, _, _, _ = self.state

        gl.glLoadIdentity()
        gl.glTranslatef(x, y, 0)
        gl.glRotatef(np.rad2deg(h), 0, 0, 1)
        self.vl.draw(pyglet.gl.GL_LINE_LOOP)


class Actuator(object):
    pass

class Force(Actuator):

    def init(self, rb):

        self.rb = rb

    def equations(self, t):

        return 0

class Moment(Actuator):

    def init(self, rb):

        self.rb = rb

    def equations(self, t):

        return [sp.sin(t), 0]


class Constraint(object):
    """Base class for a constraint."""

    def __init__(self, rigid_bodies, size):

        self.rigid_bodies = rigid_bodies
        self.N = len(rigid_bodies)
        self.size = size

    def draw(self):

        pass

    def error(self, state):

        return 0


class Rigid(Constraint):

    def __init__(self, rigid_body):

        super().__init__([rigid_body], 3)

    def equations(self, xs, ys, hs):

        return [xs[0], ys[0], hs[0]]

    def equations2(self):

        rb = self.rigid_bodies[0]
        return [rb.x, rb.y, rb.h]

class Pin(Constraint):

    def __init__(self, rb1, p1, rb2, p2):

        super().__init__([rb1, rb2], 2)

        self.p1 = sp.Matrix(p1)
        self.p2 = sp.Matrix(p2)

        # Drawing Stuff
        cs = (0, 255, 0)
        N = 12
        r = 0.1
        angles = np.arange(N)*2*np.pi/N
        xs = r*cos(angles)
        ys = r*sin(angles)
        ps = np.c_[xs, ys].flatten()

        self.vl = pyglet.graphics.vertex_list(N, ('v2f', ps), ('c3B', N*cs))

    '''
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
    '''

    def equations(self, xs, ys, hs):

        pc0 = sp.Matrix([[xs[0]], [ys[0]]])
        pc1 = sp.Matrix([[xs[1]], [ys[1]]])

        eq = pc0 + R(hs[0])*self.p1 - pc1 - R(hs[1])*self.p2

        return eq

    def equations2(self):

        p0 = self.rigid_bodies[0].p
        h0 = self.rigid_bodies[0].h

        p1 = self.rigid_bodies[1].p
        h1 = self.rigid_bodies[1].h

        eq = p0 + R(h0)*self.p1 - p1 - R(h1)*self.p2

        return eq


class Rolling(Constraint):

    def __init__(self, rigid_body):

        super(Rolling, self).__init__([rigid_body], 2)
        self.R = rigid_body.radius

    def equations(self, xs, ys, hs):

        pass
