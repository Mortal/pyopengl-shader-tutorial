# From http://pyopengl.sourceforge.net/context/tutorials/shader_6.html
import re
import textwrap
import collections

import numpy as np

from OpenGLContext import testingcontext
from OpenGL import GL as G
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
from OpenGLContext.scenegraph.basenodes import Sphere

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
        all_source = '%s\n%s' % (vertex_source, fragment_source)
        for line in all_source.splitlines():
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
        if self._indices_vbo is not None:
            self._indices_vbo.bind()
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
            try:
                a = self._attrs[i]
            except IndexError:
                raise ValueError(
                    "Supplied %d args, but attrs are: %s" %
                    (len(args), ', '.join(a.name for a in self._attrs)))
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
        vbo_data = np.asarray(self._vertices)
        self._vbo_stride = vbo_data.strides[0]
        self._vbo = vbo.VBO(vbo_data)
        if self._indices is not None:
            self._indices_vbo = vbo.VBO(
                np.asarray(self._indices, dtype=np.uint16),
                target='GL_ELEMENT_ARRAY_BUFFER')
        else:
            self._indices_vbo = None
        for a in self._vars.values():
            if a.qual == 'uniform':
                self._locs[a.name] = G.glGetUniformLocation(
                    self._shader, a.name)
            elif a.qual == 'attribute':
                self._locs[a.name] = G.glGetAttribLocation(
                    self._shader, a.name)
            elif a.qual == 'varying':
                pass

    def set_vertices(self, vertices, indices=None):
        self._vertices = []
        for v in vertices:
            self.add(*v)
        self._indices = indices
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
        if self._indices_vbo is not None:
            G.glDrawElements(G.GL_TRIANGLES, len(self._indices),
                             G.GL_UNSIGNED_SHORT, self._indices_vbo)
        else:
            G.glDrawArrays(G.GL_TRIANGLES, 0, len(self._vertices))


class TestContext(BaseContext):
    """
    Demonstrates use of attribute types in GLSL
    """
    def OnInit(self):
        self.shader = Shader.compile("""
        attribute vec3 Vertex_position;
        attribute vec3 Vertex_normal;
        varying vec3 baseNormal;
        void main() {
            gl_Position = gl_ModelViewProjectionMatrix * vec4(
                Vertex_position, 1.0
            );
            baseNormal = gl_NormalMatrix * normalize(Vertex_normal);
        }
        """, """
        vec2 phong_weightCalc(
            in vec3 light_pos,  // light position
            in vec3 half_light,  // half-way vector between light and view
            in vec3 frag_normal,  // geometry normal
            in float shininess
        ) {
            float ambientMult = max(0.0, dot(
                frag_normal, light_pos
            ));
            float diffuseMult = 0.0;
            if (ambientMult > -.05) {
                diffuseMult = pow(max(0.0, dot(
                    half_light, frag_normal
                )), shininess);
            }
            return vec2(ambientMult, diffuseMult);
        }
        uniform vec4 Global_ambient;
        uniform vec4 Light_ambient;
        uniform vec4 Light_diffuse;
        uniform vec4 Light_specular;
        uniform vec3 Light_location;
        uniform float Material_shininess;
        uniform vec4 Material_specular;
        uniform vec4 Material_ambient;
        uniform vec4 Material_diffuse;
        varying vec3 baseNormal;
        void main() {
            // normalized eye-coordinate Light location
            vec3 EC_Light_location = normalize(
                gl_NormalMatrix * Light_location
            );
            // half-vector calculation
            vec3 Light_half = normalize(
                EC_Light_location - vec3(0, 0, -1)
            );
            vec2 weights = phong_weightCalc(
                EC_Light_location,
                Light_half,
                baseNormal,
                Material_shininess
            );
            gl_FragColor = clamp(
            (
                (Global_ambient * Material_ambient)
                + (Light_ambient * Material_ambient)
                + (Light_diffuse * Material_diffuse * weights.x)
                // material's shininess is the only change here
                + (Light_specular * Material_specular * weights.y)
            ), 0.0, 1.0);
        }
        """)
        coords, indices = Sphere(
            radius=1
        ).compileArrays()
        pos = coords[:, 0:3]
        norm = coords[:, 5:8]
        vertices = list(zip(pos, norm))
        self.shader.set_vertices(vertices, indices)

    def Render(self, mode=0):
        """Render the geometry for the scene."""
        super().Render(mode)
        with self.shader:
            self.shader.setuniform('Global_ambient', [.05, .05, .05, .1])
            self.shader.setuniform('Light_ambient', [.1, .1, .1, 1.0])
            self.shader.setuniform('Light_diffuse', [.25, .25, .25, 1])
            self.shader.setuniform('Light_specular', [1.0, 1.0, 0, 1])
            self.shader.setuniform('Light_location', [6, 2, 4])
            self.shader.setuniform('Material_ambient', [.1, .1, .1, 1.0])
            self.shader.setuniform('Material_diffuse', [.15, .15, .15, 1])
            self.shader.setuniform('Material_specular', [1.0, 1.0, 1.0, 1.0])
            self.shader.setuniform('Material_shininess', [.95])
            self.shader.draw()


if __name__ == "__main__":
    TestContext.ContextMainLoop()
