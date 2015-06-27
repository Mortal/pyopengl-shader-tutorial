attribute vec3 Vertex_position;
attribute vec3 Vertex_normal;
void main() {
    gl_Position = gl_ModelViewProjectionMatrix * vec4(
        Vertex_position, 1.0
    );
    baseNormal = gl_NormalMatrix * normalize(Vertex_normal);
    light_preCalc(Vertex_position);
}
