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
