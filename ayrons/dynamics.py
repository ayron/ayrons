"""OO Multi Body Dynamics
"""

# import scipy.linalg as la
import numpy as np
import sympy as sp

from numpy import array, sin, cos
from scipy.integrate import ode
from scipy.optimize import minimize
# from scipy.linalg import norm
from sympy.physics.vector import dynamicsymbols

import pyglet
import pyglet.gl as gl

# from time import time

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

        self.paused       = True

    def __str__(self):
        return 'System of {} rigid bodies and {} constraints' \
            ''.format(len(self.rigid_bodies), len(self.constraints))

    def initialize(self):
        """Derive the system of dynamic equations."""

        N = len(self.rigid_bodies)

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
        self.Ceqs = Ceqs

        #print('Constraint Equations:')
        #for e in Ceqs:
        #    sp.pprint(e)

        dCeqs = [sp.diff(e, t).subs(dict(zip(dxs+dys+dhs, vx+vy+vh)))
                 for e in self.Ceqs]

        # Use to ensure constaints are met during the simulation
        self.constraint_equations = sp.lambdify(xs+ys+hs+vx+vy+vh, Ceqs+dCeqs, 'numpy')

        ddCeqs = [sp.diff(e, t, t) for e in Ceqs]

        # Lagrangian Multipliers
        As = dynamicsymbols('A_0:{}'.format(len(Ceqs)))


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

        # Initial Conditions
        self.t0    = 0.0
        self.state = self.generate_initial_conditions()

        self.propogator = ode(self.dynamic_system).set_integrator('vode', method='adams')
        self.propogator.set_initial_value(self.state, self.t0)

    def constraint_error(self, state):

        errors = self.constraint_equations(*state)
        error = np.dot(errors, errors)
        return error

    def generate_initial_conditions(self):
        """Collect the initial conditions provided by the user,
        and constrain the given the constraints."""

        # Collect ICs
        ics = array([rb.state for rb in self.rigid_bodies], np.float64)
        ics = ics.T.flatten()

        # Constrain

        # TODO: It should be easy to determine the jacobian of these constraints
        res = minimize(self.constraint_error, ics, method='Powell')

        ics = res.x
        if not res.success:
            print('Constraints could not be met while calculating ICs')

        # Copy state to rigid bodies
        N = len(self.rigid_bodies)
        for i, rb in enumerate(self.rigid_bodies):
            rb.state = ics[i::N]

        return ics


    def dynamic_system(self, t, state):
        """Solve the dynamic system."""

        ps = state[:len(state)//2]
        vs = state[len(state)//2:]

        # y  ~ [p1, p2, v1, v2]
        # dy ~ [v1, v2, a1, a2]

        #print(self.constraint_error(state))

        A = self.A_f(*state)
        b = self.b_f(*state).flatten()

        # Add forces
        body_forces       = self.forces(t, state)
        constraint_forces = np.zeros(len(b) - len(body_forces))
        b = b - np.hstack((body_forces, constraint_forces))
        solution = np.linalg.solve(A, b)
        accel    = solution[:len(ps)]

        dstate = np.hstack((vs, accel))

        return dstate

    def forces(self, t, state):
        """Evaluate forces being applied at time t."""

        forces = [rb.net_force(t, state) for rb in self.rigid_bodies]
        return np.hstack(zip(*forces))

    def propogate(self, dt):
        """Propogate the system forward in time."""

        if not self.paused:
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

        self.forces = []

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

    def add_force(self, Class, location):

        self.forces.append(Class(self, location))

    def net_force(self, t, state):
        """Returns (Fx, Fy, M)"""

        forces = np.array([f.evaluate(t, state) for f in self.forces], dtype=np.float64)

        if forces.size:
            return np.sum(forces, axis=0)
        else:
            return np.zeros(3, dtype=np.float64)

    def draw_forces(self):

        for f in self.forces:
            f.draw()

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

        self.draw_forces()

class Force(object):

    def __init__(self, rb, force_location):

        self.rb = rb
        self.r  = sp.Matrix(force_location)    # In body frame

        self.F_x = 0.0
        self.F_y = 0.0
        self.M   = 0.0


        # Drawing lists
        ps = ( 0.0,  0.1,
               0.0, -0.1,
               0.2,  0.0)
        cs = (255, 255, 0)
        self.arrow_head = pyglet.graphics.vertex_list(len(ps)//2, ('v2f', ps), ('c3B', (len(ps)//2)*cs))
        self.arrow_body = pyglet.graphics.vertex_list(2, ('v2f', (0.0, 0.0, 1.0, 0.0)), ('c3B', 2*cs))

    def force(self, t, state):
        """Force applied at location."""

        return 0, 0

    def moment(self, t, state):
        """Moment about CoM."""

        return 0

    def evaluate(self, t, state):

        F_x, F_y = self.force(t, state)

        # Moment due to force F
        r = R(self.rb.h)*self.r
        M_F = r[0]*F_y - r[1]*F_x

        M = self.moment(t, state) + M_F

        self.F_x = F_x
        self.F_y = F_y
        self.M   = M

        return F_x, F_y, M

    def draw(self):

        x, y, h, _, _, _ = self.rb.state

        r     = R(self.rb.h)*self.r
        angle = np.arctan2(self.F_y, self.F_x)
        mag   = np.sqrt(self.F_x**2 + self.F_y**2)

        if mag > 0.0:
            gl.glLoadIdentity()
            gl.glTranslatef(x+r[0], y+r[1], 0)
            gl.glScalef(mag, mag, mag)
            gl.glRotatef(np.rad2deg(angle), 0, 0, 1)
            self.arrow_body.draw(pyglet.gl.GL_LINE_LOOP)

            gl.glLoadIdentity()
            gl.glTranslatef(x+r[0]+self.F_x, y+r[1]+self.F_y, 0)
            gl.glRotatef(np.rad2deg(angle), 0, 0, 1)
            self.arrow_head.draw(pyglet.gl.GL_LINE_LOOP)


class Moment(object):

    def __init__(self):

        self.F_x = 0.0
        self.F_y = 0.0
        self.M   = 0.0

    def moment(self, t, state):
        """Moment about CoM."""

        return 0

    def evaluate(self, t, state):

        M = self.moment(t, state)

        self.F_x = 0.0
        self.F_y = 0.0
        self.M   = self.moment(t, state)

        return self.F_x, self.F_y, self.M

    def draw(self):

        pass


class Constraint(object):
    """Base class for a constraint."""

    def __init__(self, rigid_bodies, size):

        self.rigid_bodies = rigid_bodies
        self.N = len(rigid_bodies)
        self.size = size

    def draw(self):

        pass


class Rigid(Constraint):

    def __init__(self, rigid_body):

        super().__init__([rigid_body], 3)

    def equations2(self):

        rb = self.rigid_bodies[0]
        return [rb.x-rb.state[0],
                rb.y-rb.state[1],
                rb.h-rb.state[2]]

class Friction(Moment):

    def __init__(self, rb1, rb2):
        super().__init__()

        self.rb1 = rb1
        self.rb2 = rb2

    def moment(self, t, state):

        w1 = self.rb1.state[5]
        w2 = self.rb2.state[5]

        b = 10.0
        M = b*(w2-w1)

        return M

class Pin(Constraint):

    def __init__(self, rb1, p1, rb2, p2):

        super().__init__([rb1, rb2], 2)

        self.p1 = sp.Matrix(p1)
        self.p2 = sp.Matrix(p2)

        # Create friction forces for both rbs
        rb1.forces.append(Friction(rb1, rb2))
        rb2.forces.append(Friction(rb2, rb1))

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

        rs = [self.p1, self.p2]

        for i in range(2):
            gl.glLoadIdentity()
            x, y, h, _, _, _ = self.rigid_bodies[i].state
            pc = sp.Matrix([x, y])

            p = pc + R(h)*rs[i]
            gl.glTranslatef(p[0], p[1], 0)
            self.vl.draw(pyglet.gl.GL_LINE_LOOP)

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
