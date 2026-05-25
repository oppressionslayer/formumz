// ===========================================================================
// RULE 30 × WSRF — Parity-Detector Aurora (3D)
// ===========================================================================
//
// Lars-style visualization that pulls together three threads of /formac:
//
//   * Rule 30 cascade           (rule30_inversion_showcase.py)
//   * WSRF parity detector      (wsrf_parity_showdown.py / full_benchmark.py)
//   * Lars Logic forced bits    (ai_lie_detector_rule_system.py, lars_collapse_shadertoy.glsl)
//
// What you see, in one rotating shot:
//
//   * A finite-width Rule 30 cascade laid flat on a 3D plane (a "table").
//     The camera slowly orbits it; the floor tilts and turns.
//   * The center column of every row carries the canonical 4-cycle of
//     forced bits — those rise as a column of light at the axis of rotation
//     (the "Lars-forced ladder pillar").
//   * A teal aurora ribbon — the WSRF P2-boundary projection — drifts in
//     3D above the floor, pulsing brightest where it crosses the powers-of-2
//     projection axes.
//   * Background is a slow parallax starfield with a soft nebula gradient.
//   * Bloom + chromatic dispersion + tone-mapping on top.
//
// Paste into Shadertoy's Image tab. Targets WebGL2 / GLSL ES 3.00.
// ===========================================================================

#define PI       3.14159265358979
#define TAU     (2.0 * PI)

const int W = 64;       // cascade width in cells
const int H = 32;       // visible rows


// ---------- Rule 30 simulation (per-pixel, packed in uvec2 = 64 bits) ------
uvec2 shl1_64(uvec2 v) { return uvec2(v.x << 1u, (v.y << 1u) | (v.x >> 31u)); }
uvec2 shr1_64(uvec2 v) { return uvec2((v.x >> 1u) | ((v.y & 1u) << 31u), v.y >> 1u); }

uvec2 rule30Row(int targetRow) {
    uvec2 row = uvec2(0u);
    int c0 = W / 2;
    if (c0 < 32) row.x = 1u << uint(c0);
    else         row.y = 1u << uint(c0 - 32);
    for (int i = 0; i < 64; i++) {
        if (i >= targetRow) break;
        uvec2 L = shl1_64(row);
        uvec2 R = shr1_64(row);
        row = L ^ (row | R);
    }
    return row;
}

int rule30Bit(int r, int c) {
    if (r < 0 || r >= H || c < 0 || c >= W) return 0;
    uvec2 row = rule30Row(r);
    if (c < 32) return int((row.x >> uint(c)) & 1u);
    else        return int((row.y >> uint(c - 32)) & 1u);
}


// ---------- Hashes / noise -------------------------------------------------
float hash11(float p) {
    p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p);
}
float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float hash31(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}


// ---------- Camera ---------------------------------------------------------
struct Cam { vec3 ro; vec3 fwd; vec3 right; vec3 up; };

Cam makeCamera(float t) {
    float orbit = t * 0.18;
    float r     = 4.2 + 0.35 * sin(t * 0.27);
    float h     = 2.2 + 0.45 * sin(t * 0.21);
    vec3 ro     = vec3(sin(orbit) * r, h, cos(orbit) * r);
    vec3 target = vec3(0.0, 0.4 + 0.2 * sin(t * 0.13), 0.0);
    vec3 fwd    = normalize(target - ro);
    vec3 right  = normalize(cross(vec3(0.0, 1.0, 0.0), fwd));
    vec3 up     = normalize(cross(fwd, right));
    return Cam(ro, fwd, right, up);
}

vec3 viewRay(Cam c, vec2 ndc) {
    // 60° vertical FOV.
    float zoom = 1.0 / tan(radians(30.0));
    return normalize(c.fwd * zoom + c.right * ndc.x + c.up * ndc.y);
}


// ---------- Cascade plane intersection -------------------------------------
//
// The cascade lives on the world-space plane y = 0, occupying
// x ∈ [-CASCADE_W/2, CASCADE_W/2], z ∈ [-CASCADE_D/2, CASCADE_D/2].
// The plane spins slowly around the y axis so the cascade appears to turn.

const float CASCADE_W = 4.0;     // world units in x
const float CASCADE_D = 2.0;     // world units in z

bool castOntoFloor(vec3 ro, vec3 rd, out vec3 hit, out float dist) {
    if (rd.y >= -0.0001) return false;     // looking up or parallel
    float tHit = -ro.y / rd.y;
    if (tHit <= 0.0) return false;
    hit  = ro + rd * tHit;
    dist = tHit;
    return true;
}

vec2 floorToCascadeUV(vec3 hit, float spin) {
    // Apply inverse spin so the cascade rotates with time.
    float c = cos(spin), s = sin(spin);
    vec2 r = vec2(hit.x * c - hit.z * s, hit.x * s + hit.z * c);
    // Map to (col, row) space.
    float colF = (r.x / CASCADE_W + 0.5) * float(W);
    float rowF = (r.y / CASCADE_D + 0.5) * float(H);
    return vec2(colF, rowF);
}


// ---------- Cascade shading ------------------------------------------------
vec3 cellColor(int bit, float shimmer) {
    vec3 off = vec3(0.06, 0.05, 0.13) + 0.05 * shimmer * vec3(0.5, 0.3, 1.0);
    vec3 on  = vec3(1.05, 0.55, 0.16) + 0.15 * shimmer * vec3(1.0, 0.7, 0.3);
    return bit == 1 ? on : off;
}

float p2ColumnGlow(float colIdx) {
    float d = 1e9;
    for (int k = 0; k < 7; k++) {
        float p2 = float(1 << k);
        d = min(d, abs(colIdx - p2));
        d = min(d, abs(colIdx - (float(W) - p2)));
    }
    return exp(-d * 1.0);
}

vec3 shadeFloor(vec2 cascadeUV, vec3 hit, float t) {
    float colF = cascadeUV.x;
    float rowF = cascadeUV.y;
    int col = int(floor(colF));
    int row = int(floor(rowF));
    if (col < 0 || col >= W || row < 0 || row >= H) {
        // Margin glow around the cascade.
        float edge = exp(-2.0 * max(
            max(-colF, colF - float(W)) / float(W),
            max(-rowF, rowF - float(H)) / float(H)));
        return vec3(0.03, 0.05, 0.10) + edge * vec3(0.15, 0.25, 0.45);
    }

    int bit = rule30Bit(row, col);
    vec2  cellUV  = vec2(colF - float(col), rowF - float(row));
    float shimmer = hash21(vec2(float(col), float(row)) * 13.0 + t * 0.3);
    float bx = smoothstep(0.0, 0.06, cellUV.x) * smoothstep(0.0, 0.06, 1.0 - cellUV.x);
    float by = smoothstep(0.0, 0.06, cellUV.y) * smoothstep(0.0, 0.06, 1.0 - cellUV.y);
    float cellMask = bx * by;

    vec3 base = cellColor(bit, shimmer) * mix(0.55, 1.0, cellMask);

    // Aurora P2 axes + flowing band, projected onto floor space.
    float p2 = p2ColumnGlow(colF) * 0.55;
    float waveY =
        0.50 * sin(colF * 0.12 + t * 0.4) +
        0.28 * sin(colF * 0.21 - t * 0.27 + 1.7);
    float bandDist = abs(rowF / float(H) - 0.5 - 0.18 * waveY);
    float band = exp(-pow(bandDist / 0.10, 2.0));
    base += vec3(0.06, 0.50, 0.42) * (p2 + band * (0.30 + 0.35 * float(bit)));

    return base;
}


// ---------- Lars-forced ladder pillar --------------------------------------
//
// A vertical column of light at the spinning-cascade axis. The 4-cycle phase
// shows up as a slow brightness ripple climbing the pillar.

float pillarContribution(vec3 ro, vec3 rd, float t) {
    // Closest approach of the ray to the y-axis (x=0, z=0).
    // r(u) = ro + u*rd; minimize r.x^2 + r.z^2 → u* = -(ro·rd)_xz / (rd·rd)_xz
    float dxz = rd.x * rd.x + rd.z * rd.z;
    if (dxz < 1e-5) return 0.0;
    float uStar = -(ro.x * rd.x + ro.z * rd.z) / dxz;
    if (uStar < 0.0) return 0.0;
    vec3 p = ro + rd * uStar;
    if (p.y < 0.0 || p.y > 3.4) return 0.0;
    float d = length(p.xz);
    float falloff = exp(-d * 9.0);
    // 4-cycle modulation up the height.
    float climb  = 0.65 + 0.35 * sin(t * 1.7 + p.y * 4.2);
    float taper  = smoothstep(3.4, 0.6, p.y);   // brightest near base
    return falloff * climb * taper;
}


// ---------- 3D aurora ribbon ------------------------------------------------
//
// A flowing curve in 3D space, swept above the cascade. Each ray samples
// the ribbon by checking, at a few sampled heights, how close it passes
// to a sinusoidal curve drifting on the (x, y) plane.

float ribbonContribution(vec3 ro, vec3 rd, float t) {
    float acc = 0.0;
    // March along the ray over a small range of heights.
    for (int i = 0; i < 12; i++) {
        float h = 0.6 + 0.12 * float(i);             // y heights 0.6..2.1
        if (rd.y >= -0.0001 && rd.y <= 0.0001) continue;
        float u = (h - ro.y) / rd.y;
        if (u <= 0.0) continue;
        vec3 p = ro + rd * u;
        // Ribbon parametric in x: z = A*sin(...) and a fading envelope at large |x|.
        float ribbonZ = 0.55 * sin(p.x * 0.9 - t * 0.7)
                      + 0.22 * sin(p.x * 1.6 + t * 0.41 + 1.3);
        float dz = abs(p.z - ribbonZ);
        float dy = abs(p.y - (1.1 + 0.18 * sin(p.x * 0.5 + t * 0.6)));
        float envelope = exp(-(p.x * p.x) / 16.0);
        acc += envelope * exp(-pow(dz / 0.10, 2.0)) * exp(-pow(dy / 0.20, 2.0));
    }
    return acc * 0.18;
}


// ---------- Starfield + nebula --------------------------------------------
vec3 background(vec3 rd, float t) {
    // Nebula gradient.
    float ny = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    vec3  neb = mix(vec3(0.04, 0.02, 0.08), vec3(0.10, 0.04, 0.18), ny);
    neb += vec3(0.05, 0.10, 0.18) * pow(1.0 - abs(rd.y), 6.0);

    // Twinkling stars in the upper hemisphere.
    vec3 sd = normalize(rd) * 18.0;
    vec3 cell = floor(sd);
    float n = hash31(cell);
    float threshold = 0.985;
    float star = step(threshold, n);
    float twinkle = 0.5 + 0.5 * sin(t * 2.0 + n * 50.0);
    neb += star * twinkle * vec3(0.9, 0.95, 1.1) * max(0.0, rd.y + 0.1);

    // Far rim halo behind the cascade.
    neb += 0.10 * pow(max(0.0, -rd.y), 4.0) * vec3(0.30, 0.10, 0.45);
    return neb;
}


// ---------- Main ----------------------------------------------------------
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 res = iResolution.xy;
    vec2 ndc = (fragCoord - 0.5 * res) / min(res.x, res.y);
    float t  = iTime;

    Cam c = makeCamera(t);
    vec3 rd = viewRay(c, ndc);

    // Slight chromatic dispersion: shift the floor ray a hair per channel.
    vec3 col3 = vec3(0.0);

    // Background first (every pixel sees the sky / nebula).
    vec3 bg = background(rd, t);
    col3 = bg;

    // Cascade floor hit.
    vec3  hit;
    float floorDist;
    bool hitFloor = castOntoFloor(c.ro, rd, hit, floorDist);
    if (hitFloor) {
        float spin  = t * 0.20;
        vec2  cuv   = floorToCascadeUV(hit, spin);
        vec3  floor3 = shadeFloor(cuv, hit, t);

        // Distance-based atmospheric fade (rear of the cascade dims into haze).
        float fade = exp(-floorDist * 0.05);
        floor3 = mix(bg, floor3, clamp(fade, 0.0, 1.0));
        col3 = floor3;
    }

    // Glowing pillar at the spinning axis.
    float pillar = pillarContribution(c.ro, rd, t);
    col3 += vec3(0.55, 0.85, 1.0) * pillar * 1.6;

    // 3D aurora ribbon floating above the floor.
    float ribbon = ribbonContribution(c.ro, rd, t);
    col3 += vec3(0.20, 0.95, 0.78) * ribbon;

    // Subtle chromatic accent on the floor: re-cast slightly offset rays for R/B,
    // sample only the floor cell value, and shift the channel.
    if (hitFloor) {
        vec3 rdR = normalize(rd + 0.0015 * c.right);
        vec3 rdB = normalize(rd - 0.0015 * c.right);
        vec3 hR, hB; float dR, dB;
        if (castOntoFloor(c.ro, rdR, hR, dR)) {
            vec2 uvR = floorToCascadeUV(hR, t * 0.20);
            int  bR  = (uvR.x >= 0.0 && uvR.x < float(W) && uvR.y >= 0.0 && uvR.y < float(H))
                       ? rule30Bit(int(floor(uvR.y)), int(floor(uvR.x))) : 0;
            col3.r += 0.06 * float(bR);
        }
        if (castOntoFloor(c.ro, rdB, hB, dB)) {
            vec2 uvB = floorToCascadeUV(hB, t * 0.20);
            int  bB  = (uvB.x >= 0.0 && uvB.x < float(W) && uvB.y >= 0.0 && uvB.y < float(H))
                       ? rule30Bit(int(floor(uvB.y)), int(floor(uvB.x))) : 0;
            col3.b += 0.06 * float(bB);
        }
    }

    // Cheap bloom: lift bright values, lift again.
    vec3 bright = max(col3 - 0.7, 0.0);
    col3 += 0.6 * bright + 0.3 * bright * bright;

    // Vignette.
    float vig = length(ndc) * 0.7;
    col3 *= 1.0 - 0.30 * vig * vig;

    // Film grain.
    col3 += 0.012 * (hash21(fragCoord + t) - 0.5);

    // Tone-map (Reinhard-ish).
    col3 = col3 / (1.0 + col3);
    col3 = pow(col3, vec3(0.85));

    fragColor = vec4(col3, 1.0);
}
