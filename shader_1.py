# From http://pyopengl.sourceforge.net/context/tutorials/shader_1.html
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
            #version 120
            void main() {
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            }
        """)
        fragment_shader = compile_fragment_shader("""
            #version 120
            void main() {
                gl_FragColor = vec4( 0, 1, 0, 1 );
            }
        """)
        self.shader = shaders.compileProgram(vertex_shader, fragment_shader)
        self.vbo = vbo.VBO(
            A.array([
                [0,   1, 0],
                [-1, -1, 0],
                [1,  -1, 0],
                [2,  -1, 0],
                [4,  -1, 0],
                [4,   1, 0],
                [2,  -1, 0],
                [4,   1, 0],
                [2,   1, 0],
            ], 'f')
        )

    def Render(self, mode):
        """Render the geometry for the scene."""
        shaders.glUseProgram(self.shader)
        try:
            self.vbo.bind()
            try:
                G.glEnableClientState(G.GL_VERTEX_ARRAY)
                G.glVertexPointerf(self.vbo)
                G.glDrawArrays(G.GL_TRIANGLES, 0, 9)
            finally:
                self.vbo.unbind()
                G.glDisableClientState(G.GL_VERTEX_ARRAY)
        finally:
            shaders.glUseProgram(0)


if __name__ == "__main__":
    TestContext.ContextMainLoop()
