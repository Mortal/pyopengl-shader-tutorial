# From http://pyopengl.sourceforge.net/context/tutorials/shader_4.html
import textwrap

from OpenGLContext import testingcontext
from OpenGL import GL as G
from OpenGL.arrays import vbo
from OpenGLContext import arrays as A
from OpenGL.GL import shaders
from OpenGLContext.events.timer import Timer

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


def compile_shaders(vertex_source, fragment_source):
    return shaders.compileProgram(
        compile_vertex_shader(vertex_source),
        compile_fragment_shader(fragment_source))


class TestContext(BaseContext):
    """
    Demonstrates use of attribute types in GLSL
    """
    def OnInit(self):
        self.shader = compile_shaders("""
            uniform float tween;
            attribute vec3 position;
            attribute vec3 tweened;
            attribute vec3 color;
            varying vec4 baseColor;
            void main() {
                gl_Position = gl_ModelViewProjectionMatrix * mix(
                    vec4(position, 1.0),
                    vec4(tweened, 1.0),
                    tween
                );
                baseColor = vec4(color, 1.0);
            }
        """, """
            varying vec4 baseColor;
            void main() {
                gl_FragColor = baseColor;
            }
        """)
        self.vbo_data = \
            A.array([
                [0,  1,  0, 1,  3,  0, 0, 1, 0],
                [-1, -1, 0, -1, -1, 0, 1, 1, 0],
                [1,  -1, 0, 1,  -1, 0, 0, 1, 1],
                [2,  -1, 0, 2,  -1, 0, 1, 0, 0],
                [4,  -1, 0, 4,  -1, 0, 0, 1, 0],
                [4,  1,  0, 4,  9,  0, 0, 0, 1],
                [2,  -1, 0, 2,  -1, 0, 1, 0, 0],
                [4,  1,  0, 1,  3,  0, 0, 0, 1],
                [2,  1,  0, 1,  -1, 0, 0, 1, 1],
            ], 'f')
        self.vbo_stride = self.vbo_data.strides[0]
        self.vbo = vbo.VBO(self.vbo_data)

        self.position_location = G.glGetAttribLocation(self.shader, 'position')
        self.tweened_location = G.glGetAttribLocation(self.shader, 'tweened')
        self.color_location = G.glGetAttribLocation(self.shader, 'color')
        self.tween_location = G.glGetUniformLocation(self.shader, 'tween')

        self.tween_fraction = 0.0

        self.time = Timer(duration=2.0, repeating=1)
        self.time.addEventHandler("fraction", self.OnTimerFraction)
        self.time.register(self)
        self.time.start()

    def Render(self, mode=0):
        """Render the geometry for the scene."""
        super().Render(mode)
        shaders.glUseProgram(self.shader)
        G.glUniform1f(self.tween_location, self.tween_fraction)
        try:
            self.vbo.bind()
            try:
                G.glEnableVertexAttribArray(self.position_location)
                G.glEnableVertexAttribArray(self.tweened_location)
                G.glEnableVertexAttribArray(self.color_location)
                G.glVertexAttribPointer(
                    self.position_location,
                    3, G.GL_FLOAT, False, self.vbo_stride, self.vbo
                )
                G.glVertexAttribPointer(
                    self.tweened_location,
                    3, G.GL_FLOAT, False, self.vbo_stride, self.vbo + 12
                )
                G.glVertexAttribPointer(
                    self.color_location,
                    3, G.GL_FLOAT, False, self.vbo_stride, self.vbo + 24
                )
                G.glDrawArrays(G.GL_TRIANGLES, 0, 9)
            finally:
                G.glDisableVertexAttribArray(self.color_location)
                G.glDisableVertexAttribArray(self.tweened_location)
                G.glDisableVertexAttribArray(self.position_location)
                self.vbo.unbind()
        finally:
            shaders.glUseProgram(0)

    def OnTimerFraction(self, event):
        frac = event.fraction()
        if frac > .5:
            frac = 1.0-frac
        frac *= 2
        self.tween_fraction = frac
        self.triggerRedraw()


if __name__ == "__main__":
    TestContext.ContextMainLoop()
