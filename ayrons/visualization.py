"""A simple multibody dynamics simulator
"""

import numpy as np
import pyglet
import pyglet.gl as gl


class Visualizer(pyglet.window.Window):

    def __init__(self):
        super(Visualizer, self).__init__(resizable=False)

        self.center     = np.zeros(2)
        self.zoom       = 10.0
        self.zoom_scale = 0.10    # percentage scale during zoom for each scroll

    def set_system(self, system):
        self.system = system
        self.system.initialize()
        pyglet.clock.schedule_interval(self.system.propogate, 1/60.)

    def run(self):

        pyglet.app.run()

    def on_draw(self):

        self.clear()

        for rb in self.system.rigid_bodies:
            rb.draw()

        for c in self.system.constraints:
            c.draw()

    def on_resize(self, width, height):

        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        w  = self.zoom
        h  = height*w/width

        left   = self.center[0] - w/2
        right  = self.center[0] + w/2
        top    = self.center[1] + h/2
        bottom = self.center[1] - h/2

        gl.glOrtho(left, right, bottom, top, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """Zoom"""

        p = np.array([x, y], dtype=np.float32)
        s = np.array([self.width, self.height])
        self.zoom2  = self.zoom * (1 - scroll_y*self.zoom_scale)
        self.center = self.center + (p/s - 0.5)*(self.zoom - self.zoom2)
        self.zoom = self.zoom2

        self.on_resize(self.width, self.height)
