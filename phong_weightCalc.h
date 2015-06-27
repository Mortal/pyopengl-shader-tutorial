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
