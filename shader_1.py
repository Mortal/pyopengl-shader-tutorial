# From http://pyopengl.sourceforge.net/context/tutorials/shader_5.html
import re
import textwrap
import collections

import numpy as np

from OpenGLContext import testingcontext
from OpenGL import GL as G
from OpenGL.arrays import vbo
from OpenGL.GL import shaders

BaseContext = testingcontext.getInteractive()


ShaderVar = collections.namedtuple('ShaderVar', 'name qual type')
ShaderType = collections.namedtuple('ShaderType', 'name n suffix dtype')

TYPES = {
    t.name: t
    for t in (
        ShaderType(name='float', n=1, suffix='f', dtype=np.float32),
        ShaderType(name='vec3', n=3, suffix='f', dtype=np.float32),
        ShaderType(name='vec4', n=4, suffix='f', dtype=np.float32),
    )
}

DECL = re.compile(r'''
    (?P<qual>uniform|attribute|varying)\s+
    (?P<type>%(types)s)\s+
    (?P<name>[a-zA-Z0-9_]+)\s*
    (?P<array>\[)?
    ''' % {'types': '|'.join(TYPES.keys())},
    re.X)

# KINDS = ('uniform', 'attribute', 'varying')


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
            o = DECL.match(line.strip())
            if o is not None:
                v = ShaderVar(o.group('name'), o.group('qual'),
                              TYPES[o.group('type')])
                self._vars[v.name] = v
                if v.qual == 'attribute':
                    self._attrs.append(v)
        self._vertices = []

        return self

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

    def add(self, *args, **kwargs):
        for i, arg in enumerate(args):
            a = self._attrs[i]  # Raise IndexError if too many args
            if a.name in kwargs:
                raise ValueError("Multiply specified: %s" % a.name)
            kwargs[a.name] = arg

        attrs = set(a.name for a in self._attrs)
        values = {}
        for k, v in kwargs.items():
            attrs.remove(k)
            a = self._vars[k]
            v = np.asarray(v, dtype=a.type.dtype)
            if len(v) != a.type.n:
                raise ValueError(
                    "Attribute %s has wrong length: Got %s, expected %s" %
                    (k, len(v), a.type.n))
            values[k] = v
        if attrs:
            raise ValueError("Missing attribute(s): %r" % (attrs,))
        vertex = np.asarray([values[a.name] for a in self._attrs]).ravel()
        self._vertices.append(vertex)

    def init_vbo(self):
        self._vbo_data = np.asarray(self._vertices)
        self._vbo_stride = self._vbo_data.strides[0]
        self._vbo = vbo.VBO(self._vbo_data)
        for a in self._vars.values():
            if a.qual == 'uniform':
                self._locs[a.name] = G.glGetUniformLocation(
                    self._shader, a.name)
            elif a.qual == 'attribute':
                self._locs[a.name] = G.glGetAttribLocation(
                    self._shader, a.name)
            elif a.qual == 'varying':
                pass

    def set_vertices(self, vertices):
        self._vertices = []
        for v in vertices:
            self.add(*v)
        self.init_vbo()

    def setuniform(self, name, value):
        a = self._vars[name]
        value = np.asarray(value, dtype=a.type.dtype)
        # Either glUniform1f, glUniform3f or glUniform4f
        fname = 'glUniform%d%s' % (a.type.n, a.type.suffix)
        getattr(G, fname)(self._locs[name], *value)

    def setuniforms(self, name, value):
        a = self._vars[name]
        value = np.asarray(value, dtype=a.type.dtype)
        # Either glUniform1fv, glUniform3fv or glUniform4fv
        fname = 'glUniform%d%sv' % (a.type.n, a.type.suffix)
        getattr(G, fname)(self._locs[name], len(value), value)

    def draw(self):
        G.glDrawArrays(G.GL_TRIANGLES, 0, len(self._vertices))


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
        self.shader.set_vertices([
            [[-1, 0, 0], [-1, 0, 1]],
            [[0, 0, 1],  [-1, 0, 2]],
            [[0, 1, 1],  [-1, 0, 2]],
            [[-1, 0, 0], [-1, 0, 1]],
            [[0, 1, 1],  [-1, 0, 2]],
            [[-1, 1, 0], [-1, 0, 1]],
            [[0, 0, 1],  [-1, 0, 2]],
            [[1, 0, 1],  [1, 0, 2]],
            [[1, 1, 1],  [1, 0, 2]],
            [[0, 0, 1],  [-1, 0, 2]],
            [[1, 1, 1],  [1, 0, 2]],
            [[0, 1, 1],  [-1, 0, 2]],
            [[1, 0, 1],  [1, 0, 2]],
            [[2, 0, 0],  [1, 0, 1]],
            [[2, 1, 0],  [1, 0, 1]],
            [[1, 0, 1],  [1, 0, 2]],
            [[2, 1, 0],  [1, 0, 1]],
            [[1, 1, 1],  [1, 0, 2]],
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
            self.shader.draw()


if __name__ == "__main__":
    TestContext.ContextMainLoop()
