const int nlights = NLIGHTS;
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
