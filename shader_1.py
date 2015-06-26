# From http://pyopengl.sourceforge.net/context/tutorials/shader_10.html
import re
import time
import textwrap
import collections

import numpy as np
import PIL.Image

from OpenGLContext import testingcontext
import OpenGL.GL as G
import OpenGL.GLU as GLU
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
from OpenGLContext.scenegraph import basenodes as N

BaseContext = testingcontext.getInteractive()


Light = collections.namedtuple(
    'Light', 'ambient diffuse specular position attenuation spot spotdir')
ShaderVar = collections.namedtuple('ShaderVar', 'name qual type')
ShaderType = collections.namedtuple('ShaderType', 'name n suffix dtype')

TYPES = {
    t.name: t
    for t in (
        ShaderType(name='float', n=1, suffix='f', dtype=np.float32),
        ShaderType(name='vec2', n=2, suffix='f', dtype=np.float32),
        ShaderType(name='vec3', n=3, suffix='f', dtype=np.float32),
        ShaderType(name='vec4', n=4, suffix='f', dtype=np.float32),
        ShaderType(name='sampler2D', n=1, suffix='i', dtype=np.int32),
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
    def compile_shader(source, shaderType):
        source = source.encode()
        shader = G.glCreateShader(shaderType)
        G.glShaderSource(shader, source)
        G.glCompileShader(shader)
        result = G.glGetShaderiv(shader, G.GL_COMPILE_STATUS)
        if not result:
            print("Shader compilation failed\n%s\n%s" %
                  (G.glGetShaderInfoLog(shader).decode(),
                   textwrap.dedent(source.decode().strip())))
            raise SystemExit()
        return shader

    @classmethod
    def compile(cls, vertex_source, fragment_source):
        self = cls()
        self._vars = {}
        self._attrs = []
        self._locs = {}
        self._vertices = []

        self._shader = shaders.ShaderProgram(G.glCreateProgram())
        vertex_shader = Shader.compile_shader(
            vertex_source, G.GL_VERTEX_SHADER)
        fragment_shader = Shader.compile_shader(
            fragment_source, G.GL_FRAGMENT_SHADER)
        G.glAttachShader(self._shader, vertex_shader)
        G.glAttachShader(self._shader, fragment_shader)

        all_source = '%s\n%s' % (vertex_source, fragment_source)
        for line in all_source.splitlines():
            o = DECL.match(line.strip())
            if o is not None:
                v = ShaderVar(o.group('name'), o.group('qual'),
                              TYPES[o.group('type')])
                self._vars[v.name] = v
                if v.qual == 'attribute':
                    self._attrs.append(v)

        self.init_attribute_locs()
        G.glLinkProgram(self._shader)
        self._shader.check_validate()
        self._shader.check_linked()
        G.glDeleteShader(vertex_shader)
        G.glDeleteShader(fragment_shader)
        self.init_uniform_locs()

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
                if self._indices_vbo is not None:
                    self._indices_vbo.unbind()
                self._vbo.unbind()
            finally:
                shaders.glUseProgram(0)

    def add(self, *args):
        if tuple(len(a) for a in args) != tuple(a.type.n for a in self._attrs):
            raise ValueError("Incorrect args")
        vertex = np.concatenate(args)
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

    def init_attribute_locs(self):
        l = 0
        for a in sorted(self._vars.values(), key=lambda a: a.name):
            if a.qual != 'attribute':
                continue
            G.glBindAttribLocation(self._shader, l, a.name)
            self._locs[a.name] = l
            l += 1

    def init_uniform_locs(self):
        l = 0
        for a in sorted(self._vars.values(), key=lambda a: a.name):
            if a.qual != 'uniform':
                continue
            l = G.glGetUniformLocation(self._shader, a.name)
            self._locs[a.name] = l

    def set_vertices(self, vertices, indices=None):
        t1 = time.time()
        self._vertices = []
        vertices = [np.asarray(list(xs)) for xs in zip(*vertices)]
        vertices = list(zip(*vertices))
        for v in vertices:
            self.add(*v)
        self._indices = indices
        t2 = time.time()
        print("%s times add() took %.4f s" % (len(vertices), t2 - t1))
        self.init_vbo()
        t3 = time.time()
        print("init_vbo took %.4f s" % (t3 - t2,))

    def setuniform(self, name, value):
        a = self._vars[name]
        value = np.asarray(value, dtype=a.type.dtype)
        # Either glUniform1f, glUniform3f or glUniform4f
        fname = 'glUniform%d%s' % (a.type.n, a.type.suffix)
        getattr(G, fname)(self._locs[name], *list(value.reshape((-1,))))

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

    def set_material(self, appearance, mode):
        """Convert VRML97 appearance node to series of uniform calls"""
        material = appearance.material
        alpha = 1.0 - material.transparency

        def as4(v):
            return np.asarray(list(v) + [alpha])

        color = as4(material.diffuseColor)
        ambient = material.ambientIntensity * color
        self.setuniform('material_shininess', material.shininess)
        self.setuniform('material_ambient', ambient)
        self.setuniform('material_diffuse', color)
        self.setuniform('material_specular', as4(material.specularColor))


def read_shader(filename, D=None):
    with open(filename) as fp:
        source = fp.read()
    if D is not None:
        defines = ''.join('#define %s %s\n' % (k, v)
                          for k, v in D.items())
    else:
        defines = ''
    return defines + source


class TestContext(BaseContext):
    """
    Demonstrates use of attribute types in GLSL
    """
    def OnInit(self):
        light_nodes = [
            N.DirectionalLight(
                color=(0, 1, .1),
                intensity=1.0,
                ambientIntensity=0.1,
                direction=(-.4, -1, -.4),
            ),
            N.SpotLight(
                location=(-2.5, 2.5, 2.5),
                color=(1, 0, .3),
                ambientIntensity=.1,
                attenuation=(0, 0, 1),
                beamWidth=np.pi/2,
                cutOffAngle=np.pi*.9,
                direction=(2.5, -5.5, -2.5),
                intensity=.5,
            ),
            N.PointLight(
                location=(0, -3.06, 3.06),
                color=(.05, .05, 1),
                intensity=.5,
                ambientIntensity=.1,
            ),
        ]
        self.lights = [self.light_node_as_struct(l) for l in light_nodes]
        shader_common = read_shader(
            'shader_common.h', D={'NLIGHTS': len(self.lights)})

        phong_weightCalc = read_shader('phong_weightCalc.h')
        phong_preCalc = read_shader('phong_preCalc.h')

        light_preCalc = read_shader('light_preCalc.h')

        self.shader = Shader.compile(
            shader_common + phong_preCalc + light_preCalc +
            read_shader('vertex.h'),
            shader_common + phong_weightCalc +
            read_shader('fragment.h'))

        self.set_terrain_vertices()

    def set_view(self):
        G.glMatrixMode(G.GL_MODELVIEW)
        eyeX, eyeY, eyeZ = 2, 2, 2
        centerX, centerY, centerZ = 0.5, 0.5, 0.5
        upX, upY, upZ = 0, 0, 1
        GLU.gluLookAt(eyeX, eyeY, eyeZ,
                      centerX, centerY, centerZ,
                      upX, upY, upZ)
        G.glMatrixMode(G.GL_PROJECTION)
        G.glLoadIdentity()
        GLU.gluPerspective(12, 1, 1, 100)

    def set_terrain_vertices(self):
        t1 = time.time()
        heights = np.asarray(PIL.Image.open('/home/rav/rasters/ds11.tif').convert('F'))
        t2 = time.time()
        print("Reading heights took %.4f s" % (t2 - t1,))
        heights = heights[:40, :40]
        # quads[i] is [norm, a, b, c, d],
        # abcd counter-clockwise around norm
        quads = []
        for y, row in enumerate(heights):
            for x, z in enumerate(row):
                norm = [0, 0, 1]
                quads.append(
                    ([0, 0, 1],
                     [x, y, z],
                     [x + 1, y, z],
                     [x + 1, y + 1, z],
                     [x, y + 1, z]))
                z2 = heights[y, x + 1] if x + 1 < len(row) else 0
                if z2 < z:
                    quads.append(
                        ([1, 0, 0],
                         [x + 1, y, z],
                         [x + 1, y, z2],
                         [x + 1, y + 1, z2],
                         [x + 1, y + 1, z]))
                z2 = heights[y, x - 1] if x > 0 else 0
                if z2 < z:
                    quads.append(
                        ([-1, 0, 0],
                         [x, y + 1, z2],
                         [x, y, z2],
                         [x, y, z],
                         [x, y + 1, z]))
                z2 = heights[y + 1, x] if y + 1 < len(heights) else 0
                if z2 < z:
                    quads.append(
                        ([0, 1, 0],
                         [x + 1, y + 1, 0],
                         [x, y + 1, 0],
                         [x, y + 1, z],
                         [x + 1, y + 1, z]))
                z2 = heights[y - 1, x] if y > 0 else 0
                if z2 < z:
                    quads.append(
                        ([0, -1, 0],
                         [x, y, z],
                         [x, y, 0],
                         [x + 1, y, 0],
                         [x + 1, y, z]))
        t3 = time.time()
        print("Creating %s quads from %s cells took %.4f s" % (len(quads), len(heights.ravel()), t3 - t2))
        vertices = []
        normals = []
        indices = []
        for norm, a, b, c, d in quads:
            ai, bi, ci, di = range(len(vertices), len(vertices) + 4)
            vertices += [a, b, c, d]
            normals += 4*[norm]
            indices += [
                ai, bi, di,
                bi, ci, di,
            ]
        vertices = np.asarray(vertices)
        vmin = vertices.min(axis=0, keepdims=True)
        vmax = vertices.max(axis=0, keepdims=True)
        vertices = (vertices - vmin) / (vmax - vmin)
        v = list(zip(vertices, normals))
        t4 = time.time()
        print("Post-processing quads took %.4f s" % (t4 - t3,))
        self.shader.set_vertices(v, indices)
        t5 = time.time()
        print("set_vertices took %.4f s" % (t5 - t4,))

    def Render(self, mode=0):
        """Render the geometry for the scene."""
        super().Render(mode)
        with self.shader:
            for name, val in [
                ('Global_ambient', (.05, .05, .05, 1.0)),
                ('material_ambient', (.2, .2, .2, 1.0)),
                ('material_diffuse', (.5, .5, .5, 1.0)),
                ('material_specular', (.8, .8, .8, 1.0)),
                ('material_shininess', (.8,)),
            ]:
                self.shader.setuniform(name, val)
            for k in Light._fields:
                self.shader.setuniforms(
                    'lights_' + k, [getattr(l, k) for l in self.lights])
            self.set_view()
            self.shader.draw()

    def light_node_as_struct(self, light):
        """Given a single VRML97 light-node, produce light value array"""
        if not light.on:
            z = np.zeros(len(Light._fields), 4)
            return Light(*z)
        color = light.color

        def as4(v, w=1.0):
            return np.asarray(list(v) + [w])

        if isinstance(light, N.DirectionalLight):
            position = -as4(light.direction, 0)
            attenuation = spot = spotdir = np.zeros(4)
        else:
            position = as4(light.location)
            attenuation = as4(light.attenuation)
            if isinstance(light, N.SpotLight):
                spot = [np.cos(light.beamWidth / 4),
                        light.cutOffAngle / light.beamWidth,
                        0, 1.0]
                spotdir = as4(light.direction)
            else:
                spot = spotdir = np.zeros(4)

        return Light(
            ambient=as4(color * light.ambientIntensity),
            diffuse=as4(color * light.intensity),
            specular=as4(color * light.intensity),
            position=position,
            attenuation=attenuation,
            spot=spot,
            spotdir=spotdir,
        )


if __name__ == "__main__":
    TestContext.ContextMainLoop()
