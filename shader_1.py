# From http://pyopengl.sourceforge.net/context/tutorials/shader_3.html
import textwrap

from OpenGLContext import testingcontext
from OpenGL import GL as G
from OpenGL.arrays import vbo
from OpenGLContext import arrays as A
from OpenGL.GL import shaders

BaseContext = testingcontext.getInteractive()


def compile_shader(source, kind):
    try:
        shader = shaders.compileShader(source, kind)
    except (G.GLError, RuntimeError) as err:
        print("Shader compilation failed\n%s\n%s" %
              (err, textwrap.dedent(source.strip())))
        raise SystemExit()
    return shader


def compile_vertex_shader(source):
    return compile_shader(source, G.GL_VERTEX_SHADER)


def compile_fragment_shader(source):
    return compile_shader(source, G.GL_FRAGMENT_SHADER)


class TestContext(BaseContext):
    """
    This shader adds a simple linear fog to the shader Shows use of uniforms,
    and a few simple calculations within the vertex shader.
    """
    def OnInit(self):
        vertex_shader = compile_vertex_shader("""
            uniform float end_fog;
            uniform vec4 fog_color;
            void main() {
                float fog; // amount of fog to apply
                float fog_coord; // distance for fog calculation
                // ftransform is generally faster and is guaranteed
                // to produce the same result on each run.
                // gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                gl_Position = ftransform();
                fog_coord = abs(gl_Position.z);
                fog_coord = clamp(fog_coord, 0.0, end_fog);
                fog = (end_fog - fog_coord)/end_fog;
                fog = clamp(fog, 0.0, 1.0);
                gl_FrontColor = mix(fog_color, gl_Color, fog);
            }
        """)
        fragment_shader = compile_fragment_shader("""
            void main() {
                gl_FragColor = gl_Color;
            }
        """)
        self.shader = shaders.compileProgram(vertex_shader, fragment_shader)
        self.vbo_data = \
            A.array([
                [0,   1, 0, 0, 1, 0],
                [-1, -1, 0, 1, 1, 0],
                [1,  -1, 0, 0, 1, 1],
                [2,  -1, 0, 1, 0, 0],
                [4,  -1, 0, 0, 1, 0],
                [4,   1, 0, 0, 0, 1],
                [2,  -1, 0, 1, 0, 0],
                [4,   1, 0, 0, 0, 1],
                [2,   1, 0, 0, 1, 1],
            ], 'f')
        self.vbo_stride = self.vbo_data.strides[0]
        self.vbo = vbo.VBO(self.vbo_data)

        self.uniform_locations = {
            'end_fog': G.glGetUniformLocation(self.shader, 'end_fog'),
            'fog_color': G.glGetUniformLocation(self.shader, 'fog_color'),
        }

    def Render(self, mode=0):
        """Render the geometry for the scene."""
        super().Render(mode)
        shaders.glUseProgram(self.shader)
        G.glUniform1f(self.uniform_locations['end_fog'], 15)
        G.glUniform4f(self.uniform_locations['fog_color'], 1, 1, 1, 1)
        G.glRotate(45, 0, 1, 0)
        G.glScale(3, 3, 3)
        try:
            self.vbo.bind()
            try:
                G.glEnableClientState(G.GL_VERTEX_ARRAY)
                G.glEnableClientState(G.GL_COLOR_ARRAY)
                G.glVertexPointer(3, G.GL_FLOAT, self.vbo_stride, self.vbo)
                G.glColorPointer(3, G.GL_FLOAT, self.vbo_stride, self.vbo + 12)
                G.glDrawArrays(G.GL_TRIANGLES, 0, 9)
            finally:
                self.vbo.unbind()
                G.glDisableClientState(G.GL_COLOR_ARRAY)
                G.glDisableClientState(G.GL_VERTEX_ARRAY)
        finally:
            shaders.glUseProgram(0)


if __name__ == "__main__":
    TestContext.ContextMainLoop()
