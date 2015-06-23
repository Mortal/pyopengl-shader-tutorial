# From http://pyopengl.sourceforge.net/context/tutorials/shader_4.html
import textwrap
import collections

import numpy as np

from OpenGLContext import testingcontext
from OpenGL import GL as G
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
from OpenGLContext.events.timer import Timer

BaseContext = testingcontext.getInteractive()


ShaderVar = collections.namedtuple('ShaderVar', 'name kind type')
ShaderType = collections.namedtuple('ShaderType', 'name n')

KINDS = ('uniform', 'attribute', 'varying')
TYPES = {
    t.name: t
    for t in (
        ShaderType(name='float', n=1),
        ShaderType(name='vec3', n=3),
        ShaderType(name='vec4', n=4),
    )
}


class Shader(object):
    @staticmethod
    def compile_shader(source, kind):
        try:
            shader = shaders.compileShader(source, kind)
        except (G.GLError, RuntimeError) as err:
            print("Shader compilation failed\n%s\n%s" %
                  (err, textwrap.dedent(source.strip())))
            raise SystemExit()
        return shader

    @classmethod
    def compile(cls, vertex_source, fragment_source):
        self = cls()
        self._shader = shaders.compileProgram(
            Shader.compile_shader(vertex_source, G.GL_VERTEX_SHADER),
            Shader.compile_shader(fragment_source, G.GL_FRAGMENT_SHADER))
        self._vars = {}
        self._attrs = []
        self._locs = {}
        for line in vertex_source.splitlines():
            line = line.strip()
            try:
                kind, type_, name = line.strip().rstrip(';').split()
            except ValueError:
                continue
            if kind in KINDS and type_ in TYPES:
                v = ShaderVar(name, kind, TYPES[type_])
                self._vars[name] = v
                if v.kind == 'attribute':
                    self._attrs.append(v)

        return self

    def init_vbo(self, n):
        self._vbo_data = np.zeros((n, sum(a.type.n for a in self._attrs)),
                                  dtype=np.float32)
        self._vbo_stride = self._vbo_data.strides[0]
        self._vbo = vbo.VBO(self._vbo_data)
        for a in self._vars.values():
            if a.kind == 'uniform':
                self._locs[a.name] = G.glGetUniformLocation(
                    self._shader, a.name)
            elif a.kind == 'attribute':
                self._locs[a.name] = G.glGetAttribLocation(
                    self._shader, a.name)
            elif a.kind == 'varying':
                pass

    def __enter__(self):
        shaders.glUseProgram(self._shader)
        self._vbo.bind()
        self._attr_enabled = []
        for a in self._attrs:
            loc = self._locs[a.name]
            G.glEnableVertexAttribArray(loc)
            self._attr_enabled.append(loc)
        o = 0
        for a in self._attrs:
            G.glVertexAttribPointer(
                self._locs[a.name], a.type.n, G.GL_FLOAT, False,
                self._vbo_stride, self._vbo + o)
            o += 4 * a.type.n

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            while self._attr_enabled:
                loc = self._attr_enabled.pop()
                G.glDisableVertexAttribArray(loc)
        finally:
            try:
                self._vbo.unbind()
            finally:
                shaders.glUseProgram(0)

    def setattr(self, name, values):
        values = np.asarray(values, dtype=np.float32)
        n, m = values.shape
        if n != self._vbo_data.shape[0]:
            raise ValueError("Wrong row count for %s: Got %s, expected %s" %
                             (name, n, self._vbo_data.shape[0]))

        i = next(i for i, a in enumerate(self._attrs) if a.name == name)
        o = sum(a.type.n for a in self._attrs[:i])
        if m != self._attrs[i].type.n:
            raise ValueError("Wrong column count for %s: Got %s, expected %s" %
                             (name, m, self._attrs[i].type.n))
        self._vbo_data[:, o:o+m] = values

    def setuniform(self, name, value):
        G.glUniform1f(self._locs[name], value)


class TestContext(BaseContext):
    """
    Demonstrates use of attribute types in GLSL
    """
    def OnInit(self):
        self.shader = Shader.compile("""
            uniform float tween;
            attribute vec3 position;
            attribute vec3 tweened;
            attribute vec3 color;
            varying vec4 baseColor;
            void main() {
                gl_Position = gl_ModelViewProjectionMatrix * mix(
                    vec4(position, 1.0),
                    vec4(tweened, 1.0),
                    tween
                );
                baseColor = vec4(color, 1.0);
            }
        """, """
            varying vec4 baseColor;
            void main() {
                gl_FragColor = baseColor;
            }
        """)
        self.shader.init_vbo(9)
        self.shader.setattr('position', [
            [0,  1,  0],
            [-1, -1, 0],
            [1,  -1, 0],
            [2,  -1, 0],
            [4,  -1, 0],
            [4,  1,  0],
            [2,  -1, 0],
            [4,  1,  0],
            [2,  1,  0],
        ])
        self.shader.setattr('tweened', [
            [1,  3,  0],
            [-1, -1, 0],
            [1,  -1, 0],
            [2,  -1, 0],
            [4,  -1, 0],
            [4,  9,  0],
            [2,  -1, 0],
            [1,  3,  0],
            [1,  -1, 0],
        ])
        self.shader.setattr('color', [
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
        ])

        self.tween_fraction = 0.0

        self.time = Timer(duration=2.0, repeating=1)
        self.time.addEventHandler("fraction", self.OnTimerFraction)
        self.time.register(self)
        self.time.start()

    def Render(self, mode=0):
        """Render the geometry for the scene."""
        super().Render(mode)
        with self.shader:
            self.shader.setuniform('tween', self.tween_fraction)
            G.glDrawArrays(G.GL_TRIANGLES, 0, 9)

    def OnTimerFraction(self, event):
        frac = event.fraction()
        if frac > .5:
            frac = 1.0-frac
        frac *= 2
        self.tween_fraction = frac
        self.triggerRedraw()


if __name__ == "__main__":
    TestContext.ContextMainLoop()
