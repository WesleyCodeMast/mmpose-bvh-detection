// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D texture0;

void main() {
    vec4 color = texture(texture0, TexCoord);

    if (color.a < 0.1){
        discard;
    }
    // Apply a smooth step for edges
    // float alpha = smoothstep(0.1, 0.5, color.a); // Adjust parameters as needed
    // FragColor = vec4(color.rgb, alpha); // Use the original color with modified alpha
    FragColor = color;
}
