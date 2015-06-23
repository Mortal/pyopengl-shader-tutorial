# From http://pyopengl.sourceforge.net/context/tutorials/shader_5.html
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
        value = np.asarray(value, dtype=np.float32)
        a = self._vars[name]
        # Either glUniform1f, glUniform3f or glUniform4f
        fname = 'glUniform%df' % a.type.n
        getattr(G, fname)(self._locs[name], *value)


class TestContext(BaseContext):
    """
    Demonstrates use of attribute types in GLSL
    """
    def OnInit(self):
        self.shader = Shader.compile("""
        float phong_weightCalc(
            in vec3 light_pos, // light position
            in vec3 frag_normal // geometry normal
        ) {
            // returns vec2( ambientMult, diffuseMult )
            float n_dot_pos = max( 0.0, dot(
                frag_normal, light_pos
            ));
            return n_dot_pos;
        }

        uniform vec4 Global_ambient;
        uniform vec4 Light_ambient;
        uniform vec4 Light_diffuse;
        uniform vec3 Light_location;
        uniform vec4 Material_ambient;
        uniform vec4 Material_diffuse;
        attribute vec3 Vertex_position;
        attribute vec3 Vertex_normal;
        varying vec4 baseColor;
        void main() {
            gl_Position = gl_ModelViewProjectionMatrix * vec4(
                Vertex_position, 1.0
            );
            vec3 EC_Light_location = gl_NormalMatrix * Light_location;
            float diffuse_weight = phong_weightCalc(
                normalize(EC_Light_location),
                normalize(gl_NormalMatrix * Vertex_normal)
            );
            baseColor = clamp(
            (
                // global component
                (Global_ambient * Material_ambient)
                // material's interaction with light's contribution
                // to the ambient lighting...
                + (Light_ambient * Material_ambient)
                // material's interaction with the direct light from
                // the light.
                + (Light_diffuse * Material_diffuse * diffuse_weight)
            ), 0.0, 1.0);
        }""", """
        varying vec4 baseColor;
        void main() {
            gl_FragColor = baseColor;
        }
        """)
        self.n = 18
        self.shader.init_vbo(self.n)
        self.shader.setattr('Vertex_position', [
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [-1, 0, 0],
            [0, 1, 1],
            [-1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
            [2, 0, 0],
            [2, 1, 0],
            [1, 0, 1],
            [2, 1, 0],
            [1, 1, 1],
        ])
        self.shader.setattr('Vertex_normal', [
            [-1, 0, 1],
            [-1, 0, 2],
            [-1, 0, 2],
            [-1, 0, 1],
            [-1, 0, 2],
            [-1, 0, 1],
            [-1, 0, 2],
            [1, 0, 2],
            [1, 0, 2],
            [-1, 0, 2],
            [1, 0, 2],
            [-1, 0, 2],
            [1, 0, 2],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 2],
            [1, 0, 1],
            [1, 0, 2],
        ])

    def Render(self, mode=0):
        """Render the geometry for the scene."""
        super().Render(mode)
        with self.shader:
            self.shader.setuniform('Global_ambient', [.3, .05, .05, .1])
            self.shader.setuniform('Light_ambient', [.2, .2, .2, 1.0])
            self.shader.setuniform('Light_diffuse', [1, 1, 1, 1])
            self.shader.setuniform('Light_location', [2, 2, 10])
            self.shader.setuniform('Material_ambient', [.2, .2, .2, 1.0])
            self.shader.setuniform('Material_diffuse', [1, 1, 1, 1])
            G.glDrawArrays(G.GL_TRIANGLES, 0, self.n)


if __name__ == "__main__":
    TestContext.ContextMainLoop()
