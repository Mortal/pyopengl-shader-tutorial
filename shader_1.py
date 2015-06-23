# From http://pyopengl.sourceforge.net/context/tutorials/shader_1.html
from OpenGLContext import testingcontext
from OpenGL import GL as G
from OpenGL.arrays import vbo
from OpenGLContext import arrays as A
from OpenGL.GL import shaders

BaseContext = testingcontext.getInteractive()


class TestContext(BaseContext):
    """Creates a simple vertex shader..."""

    def OnInit(self):
        VERTEX_SHADER = shaders.compileShader("""#version 120
        void main() {
            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
        }""", G.GL_VERTEX_SHADER)
        FRAGMENT_SHADER = shaders.compileShader("""#version 120
        void main() {
            gl_FragColor = vec4( 0, 1, 0, 1 );
        }""", G.GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)
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
