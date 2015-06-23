# From http://pyopengl.sourceforge.net/context/tutorials/shader_10.html
import re
import textwrap
import collections

import numpy as np

from OpenGLContext import testingcontext
from OpenGL import GL as G
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
            print("Shader compilation failed\n%s\n%s\n%s" %
                  (result, G.glGetShaderInfoLog(shader).decode(),
                   textwrap.dedent(source.strip())))
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
        shader_common = """
        const int nlights = %(nlights)s;
        uniform vec4 lights_position[nlights];
        uniform vec4 lights_ambient[nlights];
        uniform vec4 lights_diffuse[nlights];
        uniform vec4 lights_specular[nlights];
        uniform vec4 lights_attenuation[nlights];
        // [ cos_spot_cutoff, spot_exponent, ignored, is_spot ]
        uniform vec4 lights_spot[nlights];
        uniform vec4 lights_spotdir[nlights];
        varying vec3 EC_Light_half[nlights];
        varying vec3 EC_Light_location[nlights];
        varying float Light_distance[nlights];
        varying vec3 baseNormal;
        """ % dict(nlights=len(self.lights))

        phong_weightCalc = """
        vec3 phong_weightCalc(
            in vec3 light_pos, // light position/direction
            in vec3 half_light, // half-way vector between light and view
            in vec3 frag_normal, // geometry normal
            in float shininess, // shininess exponent
            in float distance, // distance for attenuation calculation...
            in vec4 attenuations, // attenuation parameters...
            in vec4 spot_params, // spot control parameters...
            in vec4 spot_direction // model-space direction
        ) {
            // returns vec3( ambientMult, diffuseMult, specularMult )
            float n_dot_pos = max( 0.0, dot(
                frag_normal, light_pos
            ));
            float n_dot_half = 0.0;
            float attenuation = 1.0;
            if (n_dot_pos > -.05) {
                float spot_effect = 1.0;
                if (spot_params.w != 0.0) {
                    // is a spot...
                    float spot_cos = dot(
                        gl_NormalMatrix * normalize(spot_direction.xyz),
                        normalize(-light_pos)
                    );
                    if (spot_cos <= spot_params.x) {
                        // is a spot, and is outside the cone-of-light...
                        return vec3( 0.0, 0.0, 0.0 );
                    } else {
                        if (spot_cos == 1.0) {
                            spot_effect = 1.0;
                        } else {
                            spot_effect = pow(
                                    (1.0-spot_params.x)/(1.0-spot_cos),
                                    spot_params.y
                                );
                        }
                    }
                }
                n_dot_half = pow(
                    max(0.0,dot(
                        half_light, frag_normal
                    )),
                    shininess
                );
                if (distance != 0.0) {
                    float attenuation = 1.0/(
                        attenuations.x +
                        (attenuations.y * distance) +
                        (attenuations.z * distance * distance)
                    );
                    n_dot_half *= spot_effect;
                    n_dot_pos *= attenuation;
                    n_dot_half *= attenuation;
                }
            }
            return vec3( attenuation, n_dot_pos, n_dot_half);
        }
        """

        phong_preCalc = """
        // Vertex-shader pre-calculation for lighting...
        void phong_preCalc(
            in vec3 vertex_position,
            in vec4 light_position,
            out float light_distance,
            out vec3 ec_light_location,
            out vec3 ec_light_half
        ) {
            // This is the core setup for a phong lighting pass
            // as a reusable fragment of code.
            // vertex_position -- un-transformed vertex position (world-space)
            // light_position -- un-transformed light location (direction)
            // light_distance -- output giving world-space distance-to-light
            // ec_light_location -- output giving loc. of light in eye coords
            // ec_light_half -- output giving the half-vector optimization
            if (light_position.w == 0.0) {
                // directional rather than positional light...
                ec_light_location = normalize(
                    gl_NormalMatrix *
                    light_position.xyz
                );
                light_distance = 0.0;
            } else {
                // positional light, we calculate distance in
                // model-view space here, so we take a partial
                // solution...
                vec3 ms_vec = (
                    light_position.xyz -
                    vertex_position
                );
                vec3 light_direction = gl_NormalMatrix * ms_vec;
                ec_light_location = normalize( light_direction );
                light_distance = abs(length( ms_vec ));
            }
            // half-vector calculation
            ec_light_half = normalize(
                ec_light_location + vec3( 0,0,1 )
            );
        }
        """

        light_preCalc = """
        void light_preCalc( in vec3 vertex_position ) {
            // This function is dependent on the uniforms and
            // varying values we've been using, it basically
            // just iterates over the phong_lightCalc passing in
            // the appropriate pointers...
            vec3 light_direction;
            for (int i = 0; i< nlights; i++ ) {
                phong_preCalc(
                    vertex_position,
                    lights_position[i],
                    // following are the values to fill in...
                    Light_distance[i],
                    EC_Light_location[i],
                    EC_Light_half[i]
                );
            }
        }
        """

        self.shader = Shader.compile(
            shader_common + phong_preCalc + light_preCalc + """
        attribute vec3 Vertex_position;
        attribute vec3 Vertex_normal;
        void main() {
            gl_Position = gl_ModelViewProjectionMatrix * vec4(
                Vertex_position, 1.0
            );
            baseNormal = gl_NormalMatrix * normalize(Vertex_normal);
            light_preCalc(Vertex_position);
        }
        """,


            shader_common + phong_weightCalc + """
        uniform vec4 material_ambient;
        uniform vec4 material_diffuse;
        uniform vec4 material_specular;
        uniform float material_shininess;
        uniform vec4 Global_ambient;
        void main() {
            vec4 fragColor = Global_ambient * material_ambient;
            int i;
            for (i=0;i<nlights;i+=1) {
                vec3 weights = phong_weightCalc(
                    normalize(EC_Light_location[i]),
                    normalize(EC_Light_half[i]),
                    normalize(baseNormal),
                    material_shininess,
                    // some implementations will produce negative values
                    // interpolating positive float-arrays!
                    // so we have to do an extra abs call for distance
                    abs(Light_distance[i]),
                    lights_attenuation[i],
                    lights_spot[i],
                    lights_spotdir[i]
                );
                fragColor = (
                    fragColor
                    + (lights_ambient[i] * material_ambient * weights.x)
                    + (lights_diffuse[i] * material_diffuse * weights.y)
                    + (lights_specular[i] * material_specular * weights.z)
                );
            }
            gl_FragColor = fragColor;
        }
        """)

        self.set_terrain_vertices()

    def set_terrain_vertices(self):
        heights = np.asarray([
            [5, 3, 2, 1],
            [3, 2, 1, 0],
            [2, 1, 0.5, 0],
        ])
        vertices = []
        indices = []
        for y, row in enumerate(heights):
            for x, z in enumerate(row):
                norm = [0, 0, 1]
                nw, ne, sw, se = range(len(vertices), len(vertices) + 4)
                vertices += [
                    [[x, y, z], norm],
                    [[x + 1, y, z], norm],
                    [[x, y + 1, z], norm],
                    [[x + 1, y + 1, z], norm],
                ]
                indices += [
                    nw, ne, sw,
                    sw, ne, se,
                ]
        self.shader.set_vertices(vertices, indices)

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
