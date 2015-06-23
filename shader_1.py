# From http://pyopengl.sourceforge.net/context/tutorials/shader_2.html
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
    This shader just passes gl_Color from an input array to the fragment
    shader, which interpolates the values across the face (via a "varying"
    data type).
    """
    def OnInit(self):
        """Initialize the context once we have a valid OpenGL environ"""
        vertex_shader = compile_vertex_shader("""
            varying vec4 vertex_color;
            void main() {
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                vertex_color = gl_Color;
            }
        """)
        fragment_shader = compile_fragment_shader("""
            varying vec4 vertex_color;
            void main() {
                gl_FragColor = vertex_color;
            }
        """)
        self.shader = shaders.compileProgram(vertex_shader, fragment_shader)
        data = \
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
        self.vbo_stride = data.strides[0]
        self.vbo = vbo.VBO(data)

    def Render(self, mode):
        """Render the geometry for the scene."""
        super().Render(mode)
        shaders.glUseProgram(self.shader)
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
