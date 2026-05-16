// ===========================================================================
// LARS COLLAPSE — Quantum-Parallel Constraint Resolution, Visualized
// ===========================================================================
//
// Distilled from the GF(2) two-phase elimination research in lars-logic /
// schur-logic / wsrf / forensic-syndrome-decoder (William Rocha, /formac).
//
// One picture, one idea:
//   * Cells are bits (qubits) of a hidden state.
//   * Iridescent shimmer = undetermined  (superposition).
//   * Phase 1 sweep      = block-local elimination collapsing clusters.
//   * Phase 2 sweep      = cross-block bridges of light finalizing globals.
//   * Red flashes        = adversarial lies caught by the syndrome wavefront.
//
// Paste into the Image tab on shadertoy.com. No external textures, no buffers.
// Tested intent for WebGL2 / GLSL ES 3.00 (Shadertoy default).
// ===========================================================================

#define PI       3.14159265358979
#define TAU      6.28318530717959
#define LOOP     20.0          // seconds per full cycle
#define N_CLUST  9             // 3x3 cluster grid
#define HEX_R    vec2(1.0, 1.7320508)


// ---------- Hash & shimmer -------------------------------------------------
float hash11(float p){
    p = fract(p*0.1031); p *= p + 33.33; p *= p + p;
    return fract(p);
}
float hash21(vec2 p){
    vec3 p3 = fract(vec3(p.xyx)*0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y)*p3.z);
}
vec2 hash22(vec2 p){
    vec3 p3 = fract(vec3(p.xyx)*vec3(0.1031,0.1030,0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz)*p3.zy);
}

vec3 shimmer(vec2 id, float t){
    float ph = hash21(id)*TAU + t*0.55;
    return 0.55 + 0.45*cos(ph + vec3(0.0, 2.094, 4.188));
}


// ---------- Hex lattice ----------------------------------------------------
// Returns vec4( local_offset_within_cell, cell_center_in_world ).
vec4 hexGrid(vec2 p){
    vec2 h = HEX_R*0.5;
    vec2 a = mod(p,         HEX_R) - h;
    vec2 b = mod(p - h,     HEX_R) - h;
    vec2 gv = dot(a,a) < dot(b,b) ? a : b;
    return vec4(gv, p - gv);
}

// Signed distance to nearest hex edge of a unit hex.
float hexEdge(vec2 gv){
    gv = abs(gv);
    return 0.5 - max(gv.x*0.8660254 + gv.y*0.5, gv.y);
}


// ---------- Block (cluster) geometry --------------------------------------
vec2 clusterCenter(int i){
    vec2 base = vec2(float(i % 3), float(i / 3)) * 7.0 - vec2(7.0, 7.0);
    base += (hash22(vec2(float(i), 13.0)) - 0.5) * 1.6;
    return base;
}

// Returns (nearest_cluster_index, distance_to_it)
vec2 nearestCluster(vec2 cellId){
    float bestD = 1e9;
    float bestI = 0.0;
    for (int i = 0; i < N_CLUST; ++i){
        float d = length(cellId - clusterCenter(i));
        if (d < bestD){ bestD = d; bestI = float(i); }
    }
    return vec2(bestI, bestD);
}


// ---------- Constraint propagation (the heart) ----------------------------
// Returns (bitValue, certainty, isLie)
//   certainty: 0 = superposition, 1 = collapsed.
//   The two-phase schedule is encoded in `phase` ∈ [0,1].
vec3 cellState(vec2 cellId, float phase){
    float bit = step(0.5, hash21(cellId + 17.0));

    vec2 cl = nearestCluster(cellId);
    float dCluster = cl.y;

    // Phase 1: block-local elimination, wave from each cluster center.
    float p1 = smoothstep(0.08, 0.42, phase);
    float p1Front = p1 * 4.8;
    float c1 = smoothstep(p1Front + 0.6, p1Front - 0.6, dCluster);

    // Phase 2: cross-block bridges expanding from a global anchor.
    float p2 = smoothstep(0.42, 0.74, phase);
    vec2 anchor = clusterCenter(4); // middle cluster
    float dAnchor = length(cellId - anchor);
    float p2Front = p2 * 32.0;
    float c2 = smoothstep(p2Front + 1.0, p2Front - 1.0, dAnchor);

    float certainty = max(c1, c2);
    certainty = max(certainty, smoothstep(0.74, 0.80, phase));     // lock
    certainty *= 1.0 - smoothstep(0.95, 1.00, phase);              // fade

    // Sparse adversarial lies.
    float isLie = step(0.972, hash21(cellId + 71.0));
    return vec3(bit, certainty, isLie);
}


// ---------- Constraint edges ---------------------------------------------
// Sparse hash-picked edges in a 5-neighborhood; a "pulse" travels along each.
float drawEdges(vec2 worldP, vec2 cellId, float phase, out float carrier){
    float glow = 0.0;
    carrier = 0.0;
    for (int k = -1; k <= 1; ++k){
        for (int j = -1; j <= 1; ++j){
            if (j == 0 && k == 0) continue;
            vec2 nId = cellId + vec2(float(j), float(k));
            float pick = hash21(cellId + nId * 7.0);
            if (pick < 0.62) continue;
            vec2 a = cellId, b = nId;
            vec2 pa = worldP - a, ba = b - a;
            float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
            float d = length(pa - ba * h);
            float pulse_t = phase * 7.5 + hash21(cellId + nId + 5.0);
            float pulse   = exp(-32.0 * pow(h - fract(pulse_t), 2.0));
            float gA = cellState(a, phase).y;
            float gB = cellState(b, phase).y;
            float intensity = mix(0.35, 1.0, 0.5*(gA + gB));
            carrier += intensity;
            glow += smoothstep(0.022, 0.0, d) * intensity * (0.45 + 0.55*pulse);
        }
    }
    return glow;
}


// ---------- Camera (mild 3D projection onto a tilted plane) ---------------
vec2 cameraProject(vec2 uv, float t){
    // Slow yaw + tilt for a slight floating-lattice feel.
    float yaw   = 0.10 * sin(t * 0.07);
    float tilt  = 0.55;            // fixed downward tilt
    vec3 ro = vec3(0.0, 2.5, -2.5);
    vec3 rd = normalize(vec3(uv.x, uv.y - tilt, 1.0));
    // Rotate around y for yaw
    float cs = cos(yaw), sn = sin(yaw);
    rd.xz = mat2(cs, -sn, sn, cs) * rd.xz;
    ro.xz = mat2(cs, -sn, sn, cs) * ro.xz;
    // Intersect plane y = 0
    float tt = -ro.y / rd.y;
    if (tt < 0.0) tt = 50.0;
    vec3 p = ro + rd * tt;
    return p.xz;
}


// ---------- Main entry ----------------------------------------------------
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 res = iResolution.xy;
    vec2 uv = (fragCoord - 0.5*res) / min(res.x, res.y);

    float t   = iTime;
    float ph  = mod(t, LOOP) / LOOP;

    // Project to lattice space
    vec2 worldP = cameraProject(uv, t) * 3.0;
    vec4 hg     = hexGrid(worldP);
    vec2 gv     = hg.xy;
    vec2 cellId = hg.zw;

    // Cell state
    vec3 st         = cellState(cellId, ph);
    float bit       = st.x;
    float certainty = st.y;
    float isLie     = st.z;

    // Palette
    vec3 cool      = vec3(0.10, 0.65, 1.00);   // bit = 0
    vec3 warm      = vec3(1.00, 0.62, 0.10);   // bit = 1
    vec3 collapsed = mix(cool, warm, bit);
    vec3 superpo    = shimmer(cellId, t);
    vec3 cellCol   = mix(superpo, collapsed, certainty);

    // Hex outline glow rises with certainty
    float edge   = hexEdge(gv);
    float rim    = smoothstep(0.00, 0.05, edge) * (0.35 + 0.65*certainty);
    float cellMask = smoothstep(0.06, 0.18, edge);

    // Lie sweep: a wavefront sweeps the field during the syndrome window.
    float sweep = smoothstep(0.68, 0.76, ph) * (1.0 - smoothstep(0.82, 0.90, ph));
    if (isLie > 0.5){
        cellCol = mix(cellCol, vec3(1.00, 0.12, 0.10), sweep*0.85);
        rim    += sweep * 1.4;
    }

    // Constraint edges
    float carrier;
    float edges = drawEdges(worldP, cellId, ph, carrier);
    vec3 edgeCol = vec3(0.85, 0.95, 1.20) * edges * 0.85;

    // Background nebula
    float vign = 1.0 - 0.55*dot(uv, uv);
    vec3 bg = mix(vec3(0.010, 0.012, 0.030),
                  vec3(0.050, 0.020, 0.085),
                  0.5 + 0.5*sin(uv.x*1.6 + uv.y*2.3 + t*0.08));
    bg *= vign;

    // Far-distance fade (cells far from camera melt into the void)
    float dist = length(worldP) * 0.05;
    float fade = exp(-dist * dist * 0.18);

    // Compose
    vec3 col = bg;
    col = mix(col, cellCol * (0.6 + 0.5*certainty), cellMask * 0.95 * fade);
    col += rim * collapsed * 0.50 * fade;
    col += edgeCol * fade;

    // Highlight bloom
    col += pow(max(col - 0.92, 0.0), vec3(2.0)) * 0.7;

    // Subtle grain
    col += (hash21(fragCoord + t*60.0) - 0.5) * 0.018;

    // Gamma curve
    col = pow(clamp(col, 0.0, 1.6), vec3(0.85));

    fragColor = vec4(col, 1.0);
}
