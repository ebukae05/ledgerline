// Animated contour field for the hero: a slowly shifting "risk landscape"
// drawn as stepped contour rings with an iridescent fringe and film grain.
// One fragment shader, no libraries. Falls back to the CSS gradient on the
// canvas if WebGL is unavailable; holds still for prefers-reduced-motion.

const canvas = document.getElementById("field");
const gl = canvas.getContext("webgl", { antialias: false, premultipliedAlpha: false });

const VERT = `
attribute vec2 p;
void main() { gl_Position = vec4(p, 0.0, 1.0); }`;

const FRAG = `
precision highp float;
uniform vec2 res;
uniform float t;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x),
             mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++) { v += a * noise(p); p *= 2.03; a *= 0.5; }
  return v;
}

// Height of the landscape: a tilted elliptical basin, warped by slow noise.
float height(vec2 p) {
  vec2 c = vec2(0.42, 0.02);
  vec2 q = p - c;
  q *= mat2(0.92, -0.38, 0.38, 0.92);
  float d = length(q * vec2(0.85, 1.55));
  float warp = 0.16 * fbm(p * 1.4 + vec2(t * 0.025, -t * 0.018))
             + 0.05 * sin(2.6 * p.x - t * 0.12)
             + 0.04 * sin(3.4 * p.y + t * 0.09);
  return d + warp;
}

void main() {
  vec2 uv = gl_FragCoord.xy / res;
  vec2 p = (gl_FragCoord.xy - 0.5 * res) / res.y;

  float h = height(p);
  float bands = h * 6.5 - t * 0.05;
  float k = fract(bands);

  // Stepped relief: bright terrace, shadow cast just after each step.
  float lit = smoothstep(0.0, 0.10, k) * (1.0 - 0.45 * smoothstep(0.45, 1.0, k));
  vec3 paper = vec3(0.955, 0.952, 0.94);
  vec3 shade = vec3(0.47, 0.47, 0.46);
  vec3 col = mix(shade, paper, lit);

  // Iridescent fringe at each step edge, stronger toward the outer rings.
  // Broad bands, as if light splits across each step; mostly on the outer rings.
  float edge = exp(-k * 3.6) + 0.4 * exp(-(1.0 - k) * 6.0);
  vec3 iri = 0.5 + 0.5 * cos(6.28318 * (vec3(0.0, 0.33, 0.67) + 1.1 * k + 0.7 * h + 0.03 * t));
  iri = mix(vec3(dot(iri, vec3(0.333))), iri, 1.35);  // a touch more saturation
  float spread = smoothstep(0.25, 1.25, h);
  col = mix(col, clamp(iri, 0.0, 1.0) * 0.92 + 0.06, clamp(edge * 0.9 * spread, 0.0, 0.9));

  // Fade into the page background at the bottom and edges.
  vec3 bg = vec3(0.082);
  float fade = smoothstep(0.0, 0.62, uv.y) * smoothstep(1.35, 0.55, length((uv - vec2(0.62, 0.62)) * vec2(1.0, 1.25)));
  col = mix(bg, col, fade);

  // Film grain.
  col += (hash(gl_FragCoord.xy + fract(t) * 91.7) - 0.5) * 0.07;
  gl_FragColor = vec4(col, 1.0);
}`;

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

function start() {
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(program);
  gl.useProgram(program);

  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(program, "p");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  const resLoc = gl.getUniformLocation(program, "res");
  const tLoc = gl.getUniformLocation(program, "t");
  const still = matchMedia("(prefers-reduced-motion: reduce)");
  let visible = true;
  let frame = 0;

  function resize() {
    const scale = Math.min(window.devicePixelRatio || 1, 1.5);
    canvas.width = Math.round(canvas.clientWidth * scale);
    canvas.height = Math.round(canvas.clientHeight * scale);
    gl.viewport(0, 0, canvas.width, canvas.height);
  }

  function draw(ms) {
    gl.uniform2f(resLoc, canvas.width, canvas.height);
    gl.uniform1f(tLoc, still.matches ? 8.0 : ms / 1000);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  function loop(ms) {
    draw(ms);
    if (visible && !still.matches) frame = requestAnimationFrame(loop);
  }

  function restart() {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(loop);
  }

  new ResizeObserver(() => { resize(); draw(performance.now()); }).observe(canvas);
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (visible) restart();
  }).observe(canvas);
  still.addEventListener("change", restart);
  resize();
  restart();
}

if (gl) {
  try { start(); } catch (error) { console.warn("contour field disabled:", error); }
}
