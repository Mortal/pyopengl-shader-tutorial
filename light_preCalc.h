void light_preCalc( in vec3 vertex_position ) {
    // This function is dependent on the uniforms and
    // varying values we've been using, it basically
    // just iterates over the phong_lightCalc passing in
    // the appropriate pointers...
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
