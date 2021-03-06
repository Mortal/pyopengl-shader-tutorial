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
from OpenGLContext.events.timer import Timer

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
        vertex = np.asarray(args, dtype=np.float32).ravel()
        self._vertices.append(vertex)

    def init_vbo(self):
        vbo_data = np.asarray(self._vertices)
        self._vbo_stride = vbo_data.strides[0]
        self._vbo = vbo.VBO(vbo_data)
        if self._indices is not None:
            self._indices_vbo = vbo.VBO(
                np.asarray(self._indices, dtype=np.uint32),
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
        vertices = [np.asarray(list(xs), dtype=a.type.dtype)
                    for xs, a in zip(zip(*vertices), self._attrs)]
        self._vertices = np.hstack(vertices)
        self._indices = indices
        self.init_vbo()

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
                             G.GL_UNSIGNED_INT, self._indices_vbo)
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

    def get_lights(self):
        light1 = self.angle + np.pi * 1.2
        light2 = self.angle + np.pi * 1.8
        light_nodes = [
            N.DirectionalLight(
                color=(.1, .8, 0),
                intensity=0.8,
                ambientIntensity=0.1,
                direction=(np.cos(light1), np.sin(light1), -1),
            ),
            N.DirectionalLight(
                color=(.1, .8, 0),
                intensity=0.4,
                ambientIntensity=0.1,
                direction=(np.cos(light2), np.sin(light2), -1),
            ),
            # N.SpotLight(
            #     location=(0.5, 0.5, 1),
            #     color=(1, 1, 1),
            #     ambientIntensity=.1,
            #     attenuation=(0, 0, 1),
            #     beamWidth=np.pi*0.2,
            #     cutOffAngle=np.pi*.9,
            #     direction=(0, 0, -1),
            #     intensity=.5,
            # ),
            # N.PointLight(
            #     location=(0.5, 0.5, 0.2),
            #     color=(0.5, 0.5, 0.5),
            #     intensity=.1,
            #     ambientIntensity=0,
            #     attenuation=(0, .5, 0),
            # ),
        ]
        return [self.light_node_as_struct(l) for l in light_nodes]

    def OnInit(self):
        self.angle = 0

        shader_common = read_shader(
            'shader_common.h', D={'NLIGHTS': len(self.get_lights())})

        phong_weightCalc = read_shader('phong_weightCalc.h')
        phong_preCalc = read_shader('phong_preCalc.h')

        light_preCalc = read_shader('light_preCalc.h')

        self.shader = Shader.compile(
            shader_common + phong_preCalc + light_preCalc +
            read_shader('vertex.h'),
            shader_common + phong_weightCalc +
            read_shader('fragment.h'))

        self.time = Timer(duration=20.0, repeating=1)
        self.time.addEventHandler("fraction", self.OnTimerFraction)
        self.time.register(self)
        self.time.start()

        self.set_terrain_ttd()

        self.vw = self.vh = 300

    def OnResize(self, *args):
        self.vw, self.vh = args
        super().OnResize(*args)

    def set_view(self):
        G.glMatrixMode(G.GL_MODELVIEW)
        center = np.array([0.5, 0.5, 0])
        radius = 4.5
        v = np.pi * 0.2
        eye = (center +
               radius *
               np.array([np.cos(self.angle), np.sin(self.angle), 1]) *
               np.array([np.cos(v), np.cos(v), np.sin(v)]))
        eyeX, eyeY, eyeZ = eye
        centerX, centerY, centerZ = center
        upX, upY, upZ = 0, 0, 1
        G.glLoadIdentity()
        GLU.gluLookAt(eyeX, eyeY, eyeZ,
                      centerX, centerY, centerZ,
                      upX, upY, upZ)

        G.glMatrixMode(G.GL_PROJECTION)
        G.glLoadIdentity()
        vfov = self.vh / 150
        if vfov > 10:
            vfov = 10 * (1 + np.log(vfov / 10))
        GLU.gluPerspective(vfov, self.vw / self.vh, 0.1, 100)

    def load_terrain(self):
        t1 = time.time()
        heights = np.asarray(
            PIL.Image.open('/home/rav/rasters/ds11.tif').convert('F'))
        t2 = time.time()
        print("Reading heights took %.4f s" % (t2 - t1,))
        return heights[:200, :200]

    def set_terrain_mc(self):
        heights = self.load_terrain()
        t2 = time.time()
        # quads[i] is [norm, a, b, c, d],
        # abcd right-handed counter-clockwise around norm
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
        self.normalize_vertices(vertices)
        v = list(zip(vertices, normals))
        t4 = time.time()
        print("Post-processing quads took %.4f s" % (t4 - t3,))
        self.shader.set_vertices(v, indices)
        t5 = time.time()
        print("set_vertices took %.4f s" % (t5 - t4,))

    def set_terrain_ttd(self):
        heights = self.load_terrain()

        ys, xs = np.indices(heights.shape)
        xyz = np.dstack((xs, ys, heights))
        aa = xyz[:-1, :-1]
        bb = xyz[:-1, 1:]
        cc = xyz[1:, 1:]
        dd = xyz[1:, :-1]
        # ac[y, x] and bd[y, x] are the two ways of triangulating the
        # [x, x+1]*[y, y+1] quad
        bd = np.concatenate((aa, bb, dd, bb, cc, dd), axis=2)
        ac = np.concatenate((aa, cc, dd, aa, bb, cc), axis=2)

        # bd_lower[y, x] is true if the bd diagonal is lower than ac
        bd_lower = aa[:, :, 2] + cc[:, :, 2] > bb[:, :, 2] + dd[:, :, 2]
        # ts[y, x, :] is the triangulation of the [x, x+1]*[y, y+1] quad
        # with the lower diagonal.
        ts = np.choose(bd_lower[:, :, np.newaxis], (ac, bd))
        # Number of triangles = 2 * number of quads
        nts = 2 * ts.shape[0] * ts.shape[1]
        # Reshape to list of triangles
        ts = ts.reshape((nts, 9))
        # Compute the surface normals
        p1 = ts[:, 0:3]
        p2 = ts[:, 3:6]
        p3 = ts[:, 6:9]
        n = np.cross(p2 - p1, p3 - p1)  # surface normals
        # Turn into list of (point xyz, normal xyz)
        v = np.c_[p1, n, p2, n, p3, n].reshape(3 * nts, 2, 3)
        # Normalize all point xyz
        self.normalize_vertices(v[:, 0, :])
        self.shader.set_vertices(v)

    def normalize_vertices(self, vertices):
        vmin = vertices.min(axis=0, keepdims=True)
        vmax = vertices.max(axis=0, keepdims=True)
        vertices[:] = (vertices - vmin) / (vmax - vmin)
        vertices[:, 2] -= 0.5
        vertices[:, 2] /= (vmax[0, 2] - vmin[0, 2]) / 40

    def Render(self, mode=0):
        """Render the geometry for the scene."""
        super().Render(mode)
        with self.shader:
            for name, val in [
                ('Global_ambient', (.2, .2, .2, 1.0)),
                ('material_ambient', (.5, .8, .5, 1.0)),
                ('material_diffuse', (.2, .8, .2, 1.0)),
                ('material_specular', (.05, .05, .05, 1.0)),
                ('material_shininess', (1,)),
            ]:
                self.shader.setuniform(name, val)
            lights = self.get_lights()
            for k in Light._fields:
                self.shader.setuniforms(
                    'lights_' + k, [getattr(l, k) for l in lights])
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

    def OnTimerFraction(self, event):
        frac = event.fraction()
        self.angle = 2 * np.pi * frac
        self.triggerRedraw()


if __name__ == "__main__":
    TestContext.ContextMainLoop()
