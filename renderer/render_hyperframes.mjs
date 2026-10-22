#!/usr/bin/env node
// AgentReel HyperFrames render worker.
//
//   node renderer/render_hyperframes.mjs --in req.json --out resp.json
//   req:  {scene_path, out_path, layout_path?, mock?, workers?, crf?, keep_project?, ffmpeg_path?, ffprobe_path?}
//   resp: {status, error, outputs: {video_path, frames, layout_path, render_engine, project_dir},
//          units: {frames, render_s, render_fps, capture_s}, model, model_revision, licence}
//
// Composes ONE HyperFrames project per render (temp dir) from scene.json (core/scene.py): every shot is a
// `.clip` with data-start/data-duration filled from its template fragment (renderer/templates/<name>/index.html),
// one paused GSAP root timeline (template timeline.js + seeded Ken Burns + caption word highlights), a voice
// <audio> track, local fonts/GSAP. Text lines are pre-fitted in Python (scene.shots[].text) so the browser never
// re-wraps. Then `hyperframes render` -> MP4, and a headless-Chrome pass measures every text box -> layout.json.
// Deterministic: no Math.random / Date.now in the page; randomness only from the scene seed (mulberry32).
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..");
const TEMPLATES = path.join(HERE, "templates");
const HF_BIN = path.join(HERE, "node_modules", "hyperframes", "bin", "hyperframes.mjs");
const HF_VERSION = JSON.parse(fs.readFileSync(path.join(HERE, "node_modules", "hyperframes", "package.json"), "utf8")).version;
const GSAP_JS = path.join(HERE, "node_modules", "gsap", "dist", "gsap.min.js");

// ------------------------------------------------------------------ small utils
const resolveRepo = (p) => (path.isAbsolute(p) ? p : path.join(REPO, p));
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const r3 = (x) => Math.round(x * 1000) / 1000;

/** Seeded PRNG (mulberry32) — the only randomness source; used when a shot has no precomputed kenburns. */
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function kenburnsFallback(seed, shotId) {
  const rnd = mulberry32((seed * 2654435761 + shotId * 97) >>> 0);
  const zHi = 1.08 + rnd() * 0.08, zoomIn = rnd() < 0.6;
  const dx = (rnd() < 0.5 ? -1 : 1) * (0.3 + rnd() * 0.5), dy = (rnd() < 0.5 ? -1 : 1) * (0.2 + rnd() * 0.4);
  return { zoom_from: zoomIn ? 1 : zHi, zoom_to: zoomIn ? zHi : 1, pan_x_from: -dx / 2, pan_x_to: dx / 2, pan_y_from: -dy / 2, pan_y_to: dy / 2 };
}

/** Mustache-lite: {{#a}}..{{/a}} (list/truthy), {{^a}}..{{/a}}, {{{raw}}}, {{escaped}}, dotted names, {{.}}. */
function renderTpl(tpl, stack) {
  const lookup = (name) => {
    if (name === ".") return stack[stack.length - 1];
    for (let i = stack.length - 1; i >= 0; i--) {
      let v = stack[i];
      const parts = name.split(".");
      let ok = true;
      for (const p of parts) {
        if (v !== null && typeof v === "object" && p in v) v = v[p];
        else { ok = false; break; }
      }
      if (ok) return v;
    }
    return undefined;
  };
  const sec = /\{\{([#^])\s*([\w.]+)\s*\}\}([\s\S]*?)\{\{\/\s*\2\s*\}\}/;
  let out = tpl, m;
  while ((m = sec.exec(out))) {
    const [all, kind, name, inner] = m;
    const v = lookup(name);
    const truthy = Array.isArray(v) ? v.length > 0 : !!v;
    let rep = "";
    if (kind === "^") rep = truthy ? "" : renderTpl(inner, stack);
    else if (Array.isArray(v)) rep = v.map((it) => renderTpl(inner, [...stack, it])).join("");
    else if (truthy) rep = renderTpl(inner, typeof v === "object" ? [...stack, v] : stack);
    out = out.slice(0, m.index) + rep + out.slice(m.index + all.length);
  }
  out = out.replace(/\{\{\{\s*([\w.]+)\s*\}\}\}/g, (_, n) => { const v = lookup(n); return v === undefined || v === null ? "" : String(v); });
  return out.replace(/\{\{\s*([\w.]+)\s*\}\}/g, (_, n) => { const v = lookup(n); return v === undefined || v === null ? "" : esc(v); });
}

/** Adds `html` (escaped lines joined by <br>) to every fitted-text dict in shot.text. */
function withHtml(node) {
  if (Array.isArray(node)) return node.map((x, i) => (x && typeof x === "object" ? { n: i + 1, ...withHtml(x) } : x));
  if (node && typeof node === "object") {
    const o = {};
    for (const [k, v] of Object.entries(node)) o[k] = k === "lines" ? v : withHtml(v);
    if (Array.isArray(node.lines)) o.html = node.lines.map(esc).join("<br>");
    return o;
  }
  return node;
}

function copyInto(projDir, src, rel) {
  const dst = path.join(projDir, rel);
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.copyFileSync(src, dst);
  return rel.split(path.sep).join("/");
}

// ------------------------------------------------------------------ composition
export function composeProject(scene, projDir) {
  const W = scene.width, H = scene.height, T = scene.duration_s, aspect = scene.aspect;
  fs.mkdirSync(projDir, { recursive: true });
  copyInto(projDir, GSAP_JS, "assets/gsap.min.js");
  const fontsDir = resolveRepo(scene.fonts_dir || "assets/fonts");
  for (const f of fs.readdirSync(fontsDir)) if (/\.(ttf|otf|woff2?)$/i.test(f)) copyInto(projDir, path.join(fontsDir, f), path.join("assets", "fonts", f));

  const used = [...new Set(scene.shots.map((s) => s.template))];
  let css = fs.readFileSync(path.join(TEMPLATES, "_shared", "base.css"), "utf8");
  let tplJs = "";
  const frags = {};
  for (const t of used) {
    const dir = path.join(TEMPLATES, t);
    if (!fs.existsSync(path.join(dir, "index.html"))) throw new Error(`unknown template ${t}`);
    frags[t] = fs.readFileSync(path.join(dir, "index.html"), "utf8");
    if (fs.existsSync(path.join(dir, "style.css"))) css += `\n/* ${t} */\n` + fs.readFileSync(path.join(dir, "style.css"), "utf8");
    tplJs += `\n/* ${t} */\n` + fs.readFileSync(path.join(dir, "timeline.js"), "utf8");
  }

  const logoExt = path.extname(scene.brand.logo_path) || ".svg";
  const logoRel = copyInto(projDir, resolveRepo(scene.brand.logo_path), `assets/logo${logoExt}`);
  const z = scene.safe;
  const zoneStyle = (b) => `left:${b.x0}px;top:${b.y0}px;width:${b.x1 - b.x0}px;height:${b.y1 - b.y0}px`;

  const shotsHtml = [];
  const runtimeShots = [];
  for (const s of scene.shots) {
    let img = "";
    if (s.image) {
      const ext = path.extname(s.image) || ".png";
      const rel = copyInto(projDir, resolveRepo(s.image), `assets/img/shot_${s.shot_id}${ext}`);
      img = `<img class="ar-kb" src="${rel}" alt="" />`;
    }
    const view = { ...withHtml(s.text || {}), logo_src: logoRel };
    const frag = renderTpl(frags[s.template], [view]);
    const zone = s.zone === "end" ? z.end : z.content;
    const dur = r3(s.end - s.start);
    shotsHtml.push(
      `<div id="shot-${s.shot_id}" class="clip ar-shot t-${s.template}${s.image ? " has-image" : ""}" data-shot="${s.shot_id}" ` +
      `data-start="${r3(s.start)}" data-duration="${dur}" data-track-index="0">` +
      `<div class="ar-bg">${img || '<div class="ar-accent"></div>'}</div><div class="ar-scrim"></div>` +
      `<div class="ar-zone" style="${zoneStyle(zone)}">${frag}</div></div>`);
    runtimeShots.push({ id: s.shot_id, template: s.template, start: r3(s.start), end: r3(s.end),
      kb: s.image ? (s.kenburns || kenburnsFallback(scene.seed, s.shot_id)) : null });
  }

  const capHtml = [];
  const runtimeCaps = [];
  const cz = z.caption;
  for (const c of scene.captions || []) {
    const words = [];
    const lines = c.lines.map((line) => line.map((w) => {
      words.push([r3(w.start), r3(w.end)]);
      return `<span class="w"><span class="hl"></span><span class="tx">${esc(w.text)}</span></span>`;
    }).join(" "));
    capHtml.push(`<div id="cap-${c.index}" class="clip ar-cap" data-cap="${c.index}" data-start="${r3(c.start)}" ` +
      `data-duration="${r3(c.end - c.start)}" data-track-index="1" style="font-size:${c.px || 64}px;line-height:${scene.caption_style.line_height || 1.25}">` +
      lines.map((l) => `<div class="ar-cap-line" data-text="caption">${l}</div>`).join("") + `</div>`);
    runtimeCaps.push({ i: c.index, start: r3(c.start), end: r3(c.end), words });
  }

  let audioHtml = "";
  if (scene.audio && scene.audio.voice_path) {
    const ext = path.extname(scene.audio.voice_path) || ".wav";
    const rel = copyInto(projDir, resolveRepo(scene.audio.voice_path), `assets/voice${ext}`);
    const ad = r3(Math.min(scene.audio.duration_s, T));
    audioHtml = `<audio id="vo" src="${rel}" data-start="0" data-duration="${ad}" data-track-index="2" data-volume="${scene.audio.volume ?? 1}"></audio>`;
  }

  const runtime = JSON.stringify({ T, W, H, aspect, shots: runtimeShots, caps: runtimeCaps }).replace(/</g, "\\u003c");
  const cs = scene.caption_style || {};
  const outline = cs.outline_px || 6;
  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=${W}, height=${H}" />
<title>AgentReel ${esc(scene.brief_id)}</title>
<script src="assets/gsap.min.js"></script>
<style>
:root { --w: ${W}px; --h: ${H}px; --cap-outline: ${outline}px; --cap-color: ${cs.color || "#FFFFFF"};
        --cap-outline-color: ${cs.outline_color || "#110F10"}; --cap-highlight: ${cs.highlight || "#B7202E"}; }
${css}
</style>
</head>
<body>
<div id="root" class="ar-${aspect}" data-composition-id="main" data-start="0" data-duration="${r3(T)}" data-width="${W}" data-height="${H}">
${shotsHtml.join("\n")}
<div class="ar-captions" style="${zoneStyle(cz)}">
${capHtml.join("\n")}
</div>
${audioHtml}
</div>
<script>
window.AR = { templates: {}, register: function (name, fn) { this.templates[name] = fn; } };
</script>
<script>
${tplJs}
</script>
<script>
(function () {
  var S = ${runtime};
  var tl = gsap.timeline({ paused: true });
  S.shots.forEach(function (s) {
    var el = document.getElementById("shot-" + s.id);
    if (s.kb) {
      var img = el.querySelector(".ar-kb"), k = s.kb;
      var hx = function (z) { return (z - 1) * S.W / 2; }, hy = function (z) { return (z - 1) * S.H / 2; };
      tl.fromTo(img, { scale: k.zoom_from, x: k.pan_x_from * hx(k.zoom_from), y: k.pan_y_from * hy(k.zoom_from) },
                     { scale: k.zoom_to, x: k.pan_x_to * hx(k.zoom_to), y: k.pan_y_to * hy(k.zoom_to),
                       duration: Math.max(0.04, s.end - s.start), ease: "none" }, s.start);
    }
    var fn = window.AR.templates[s.template];
    if (fn) fn(tl, el.querySelector(".ar-zone"), s, { aspect: S.aspect, width: S.W, height: S.H });
  });
  S.caps.forEach(function (c) {
    var hls = document.getElementById("cap-" + c.i).querySelectorAll(".hl");
    c.words.forEach(function (w, j) {
      var off = j + 1 < c.words.length ? Math.max(w[0], c.words[j + 1][0]) : c.end;
      tl.set(hls[j], { opacity: 1 }, w[0]);
      tl.set(hls[j], { opacity: 0 }, Math.max(off, w[0] + 0.01));
    });
  });
  tl.set({}, {}, S.T);
  window.__timelines = window.__timelines || {};
  window.__timelines["main"] = tl;
  window.__AR_SCENE = S;
})();
</script>
</body>
</html>
`;
  fs.writeFileSync(path.join(projDir, "index.html"), html, "utf8");
  return path.join(projDir, "index.html");
}

// ------------------------------------------------------------------ browser helpers
function browserPath(env) {
  if (process.env.HYPERFRAMES_BROWSER_PATH && fs.existsSync(process.env.HYPERFRAMES_BROWSER_PATH)) return process.env.HYPERFRAMES_BROWSER_PATH;
  const r = spawnSync(process.execPath, [HF_BIN, "browser", "path"], { env, encoding: "utf8" });
  const lines = (r.stdout || "").trim().split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const p = lines[lines.length - 1];
  if (!p || !fs.existsSync(p)) throw new Error(`HyperFrames Chrome not found (run: npx hyperframes browser ensure). ${r.stderr || ""}`);
  return p;
}

function serveDir(dir) {
  const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".ttf": "font/ttf", ".otf": "font/otf",
    ".woff2": "font/woff2", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".svg": "image/svg+xml", ".wav": "audio/wav", ".mp3": "audio/mpeg" };
  const server = http.createServer((req, res) => {
    const rel = decodeURIComponent((req.url || "/").split("?")[0]).replace(/^\/+/, "") || "index.html";
    const file = path.join(dir, rel);
    if (!file.startsWith(dir) || !fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { "Content-Type": types[path.extname(file).toLowerCase()] || "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((ok) => server.listen(0, "127.0.0.1", () => ok(server)));
}

/** Loads the composed page in the HyperFrames Chrome and measures every [data-text] box per shot + captions. */
export async function measureLayout(projDir, scene, env) {
  const { default: puppeteer } = await import("puppeteer-core");
  const server = await serveDir(projDir);
  const browser = await puppeteer.launch({
    executablePath: browserPath(env), headless: true,
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--font-render-hinting=none", "--hide-scrollbars", "--mute-audio"],
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: scene.width, height: scene.height, deviceScaleFactor: 1 });
    await page.goto(`http://127.0.0.1:${server.address().port}/index.html`, { waitUntil: "load" });
    await page.evaluate(() => document.fonts.ready);
    return await page.evaluate(() => {
      const S = window.__AR_SCENE, tl = window.__timelines.main;
      const box = (r) => ({ x0: Math.floor(r.left), y0: Math.floor(r.top), x1: Math.ceil(r.right), y1: Math.ceil(r.bottom) });
      const hex = (c) => { const m = /rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)/.exec(c || ""); return m ? "#" + [m[1], m[2], m[3]].map((v) => Math.round(+v).toString(16).padStart(2, "0")).join("").toUpperCase() : ""; };
      const union = (a, b) => (a ? { x0: Math.min(a.x0, b.x0), y0: Math.min(a.y0, b.y0), x1: Math.max(a.x1, b.x1), y1: Math.max(a.y1, b.y1) } : b);
      const shots = S.shots.map((s) => {
        const el = document.getElementById("shot-" + s.id);
        const d = s.end - s.start;
        const times = [Math.min(s.start + 1.2, s.end - 0.04), s.start + d * 0.6, s.end - 0.04].filter((t) => t >= s.start);
        const items = Array.from(el.querySelectorAll("[data-text]")).map((n) => ({ n, b: null }));
        const logos = Array.from(el.querySelectorAll("[data-logo]")).map((n) => ({ n, b: null }));
        for (const t of times) {
          tl.seek(t, false);
          for (const it of [...items, ...logos]) {
            const r = it.n.getBoundingClientRect();
            if (r.width > 0 && r.height > 0) it.b = union(it.b, box(r));
          }
        }
        return {
          shot_id: s.id, template: s.template, start: s.start, end: s.end, measured_at: times,
          items: items.filter((it) => it.b).map((it) => {
            const cs = getComputedStyle(it.n);
            const sw = parseFloat(cs.webkitTextStrokeWidth) || 0;
            return { kind: "text", slot: it.n.getAttribute("data-text"), text: it.n.innerText.replace(/\s+/g, " ").trim(),
              font_px: parseFloat(cs.fontSize), line_height_px: Math.round(parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.2),
              lines: Math.max(1, it.n.querySelectorAll("br").length + 1), color: hex(cs.color),
              stroke_color: sw ? hex(cs.webkitTextStrokeColor) : "", stroke_px: sw / 2, ...it.b };
          }),
          logos: logos.filter((it) => it.b).map((it) => ({ kind: "logo", ...it.b })),
        };
      });
      const captions = S.caps.map((c) => {
        tl.seek((c.start + c.end) / 2, false);
        const el = document.getElementById("cap-" + c.i);
        let b = null;
        const lines = Array.from(el.querySelectorAll(".ar-cap-line"));
        lines.forEach((ln) => {
          const spans = ln.querySelectorAll(".tx");
          if (!spans.length) return;
          const a = spans[0].getBoundingClientRect(), z = spans[spans.length - 1].getBoundingClientRect();
          b = union(b, { x0: Math.floor(a.left), y0: Math.floor(Math.min(a.top, z.top)), x1: Math.ceil(z.right), y1: Math.ceil(Math.max(a.bottom, z.bottom)) });
        });
        const tx = el.querySelector(".tx"), tcs = tx ? getComputedStyle(tx) : null;
        const sw = tcs ? parseFloat(tcs.webkitTextStrokeWidth) || 0 : 0;
        return { index: c.i, start: c.start, end: c.end, text: lines.map((l) => l.innerText.replace(/\s+/g, " ").trim()),
          font_px: parseFloat(getComputedStyle(el).fontSize), lines: lines.length, color: tcs ? hex(tcs.color) : "",
          stroke_color: sw ? hex(tcs.webkitTextStrokeColor) : "", stroke_px: sw / 2, ...(b || { x0: 0, y0: 0, x1: 0, y1: 0 }) };
      });
      return { shots, captions };
    });
  } finally {
    await browser.close();
    server.close();
  }
}

/** Flat list for the critic: {shot_id, kind: text|caption|logo, text, x0..y1, font_px, lines, color, stroke_*}. */
export function flatBoxes(layout) {
  const out = [];
  for (const s of layout.shots) {
    for (const b of s.items) out.push({ shot_id: s.shot_id, kind: "text", text: b.text, x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1,
      font_px: b.font_px, lines: b.lines, color: b.color, stroke_color: b.stroke_color, stroke_px: b.stroke_px, slot: b.slot });
    for (const b of s.logos) out.push({ shot_id: s.shot_id, kind: "logo", text: "", x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1,
      font_px: 0, lines: 0, color: "", stroke_color: "", stroke_px: 0 });
  }
  for (const c of layout.captions) {
    const mid = (c.start + c.end) / 2;
    const sh = layout.shots.find((s) => mid >= s.start && mid < s.end) || layout.shots[layout.shots.length - 1];
    out.push({ shot_id: sh ? sh.shot_id : -1, kind: "caption", text: c.text.join(" / "), x0: c.x0, y0: c.y0, x1: c.x1, y1: c.y1,
      font_px: c.font_px, lines: c.lines, color: c.color, stroke_color: c.stroke_color, stroke_px: c.stroke_px,
      caption_index: c.index, start: c.start, end: c.end });
  }
  return out;
}

// ------------------------------------------------------------------ render
function runHyperframes(projDir, outPath, opts, env) {
  const args = [HF_BIN, "render", projDir, "-o", outPath, "--fps", String(opts.fps || 30), "--crf", String(opts.crf ?? 16),
    "--workers", String(opts.workers || "auto"), "--no-browser-gpu"];
  return new Promise((ok, fail) => {
    const t0 = Date.now();
    const p = spawn(process.execPath, args, { cwd: projDir, env, stdio: ["ignore", "pipe", "pipe"] });
    let log = "";
    p.stdout.on("data", (d) => { log += d; });
    p.stderr.on("data", (d) => { log += d; });
    p.on("error", fail);
    p.on("close", (code) => {
      fs.writeFileSync(path.join(projDir, "render.log"), log, "utf8");
      const secs = (Date.now() - t0) / 1000;
      if (code !== 0 || !fs.existsSync(outPath)) return fail(new Error(`hyperframes render exited ${code}: ${log.slice(-2500)}`));
      const cap = /capture ([\d.]+)s/.exec(log);
      ok({ wall_s: secs, capture_s: cap ? parseFloat(cap[1]) : null, args: args.slice(1) });
    });
  });
}

function mockRender(scene, outPath, ffmpeg) {
  // Placeholder (no Chrome): solid brand-gray video of the right size/fps/duration + the voice track.
  const args = ["-y", "-loglevel", "error", "-f", "lavfi", "-i", `color=c=0x2b2728:s=${scene.width}x${scene.height}:r=${scene.fps}:d=${scene.duration_s}`];
  const voice = scene.audio && scene.audio.voice_path ? resolveRepo(scene.audio.voice_path) : null;
  if (voice && fs.existsSync(voice)) args.push("-i", voice, "-af", "apad", "-t", String(scene.duration_s), "-c:a", "aac", "-ar", "48000", "-ac", "2");
  args.push("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", outPath);
  const r = spawnSync(ffmpeg, args, { encoding: "utf8" });
  if (r.status !== 0) throw new Error(`mock ffmpeg failed: ${r.stderr}`);
}

export async function handle(req) {
  const scene = JSON.parse(fs.readFileSync(req.scene_path, "utf8"));
  const outPath = path.resolve(req.out_path);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  const layoutPath = path.resolve(req.layout_path || path.join(path.dirname(outPath), "layout.json"));
  const frames = Math.round(scene.duration_s * scene.fps);
  const env = { ...process.env, HYPERFRAMES_NO_TELEMETRY: "1", DO_NOT_TRACK: "1", HYPERFRAMES_NO_UPDATE_CHECK: "1",
    HYPERFRAMES_NO_AUTO_INSTALL: "1", HYPERFRAMES_NO_OPTIONAL_INSTALL: "1", HYPERFRAMES_SKIP_SKILLS: "1" };
  if (req.ffmpeg_path) env.HYPERFRAMES_FFMPEG_PATH = req.ffmpeg_path;
  if (req.ffprobe_path) env.HYPERFRAMES_FFPROBE_PATH = req.ffprobe_path;
  const base = { model: `hyperframes@${HF_VERSION}`, model_revision: HF_VERSION, licence: "Apache-2.0 (HyperFrames); GSAP Standard No-Charge License" };
  const layoutBase = { engine: "hyperframes", width: scene.width, height: scene.height, aspect: scene.aspect, fps: scene.fps, safe: scene.safe };

  if (req.mock) {
    mockRender(scene, outPath, req.ffmpeg_path || env.HYPERFRAMES_FFMPEG_PATH || "ffmpeg");
    fs.writeFileSync(layoutPath, JSON.stringify({ ...layoutBase, mock: true, boxes: [], shots: [], captions: [] }, null, 1));
    return { ...base, outputs: { video_path: outPath, frames, layout_path: layoutPath, render_engine: "hyperframes-mock" }, units: { frames, mock: true } };
  }

  const projDir = fs.mkdtempSync(path.join(req.work_dir || os.tmpdir(), "agentreel-hf-"));
  try {
    composeProject(scene, projDir);
    const t0 = Date.now();
    const layout = await measureLayout(projDir, scene, env);
    const layoutS = (Date.now() - t0) / 1000;
    fs.writeFileSync(layoutPath, JSON.stringify({ ...layoutBase, mock: false, boxes: flatBoxes(layout), ...layout }, null, 1));
    const r = await runHyperframes(projDir, outPath, { fps: scene.fps, crf: req.crf, workers: req.workers }, env);
    return { ...base,
      outputs: { video_path: outPath, frames, layout_path: layoutPath, render_engine: "hyperframes", project_dir: req.keep_project ? projDir : "" },
      units: { frames, render_s: r3(r.wall_s), render_fps: r3(frames / r.wall_s), capture_s: r.capture_s,
               capture_fps: r.capture_s ? r3(frames / r.capture_s) : null, layout_s: r3(layoutS), hf_args: r.args.join(" ") } };
  } finally {
    if (!req.keep_project) fs.rmSync(projDir, { recursive: true, force: true });
  }
}

// ------------------------------------------------------------------ CLI (worker contract)
async function main() {
  const argv = process.argv.slice(2);
  const get = (flag) => { const i = argv.indexOf(flag); return i >= 0 ? argv[i + 1] : undefined; };
  const inp = get("--in"), out = get("--out");
  if (!inp || !out) { console.error("usage: node render_hyperframes.mjs --in req.json --out resp.json"); process.exit(2); }
  let resp;
  try {
    const req = JSON.parse(fs.readFileSync(inp, "utf8"));
    resp = { status: "ok", error: "", ...(await handle(req)) };
  } catch (e) {
    resp = { status: "error", error: `${e && e.stack ? e.stack : e}`.slice(0, 4000), outputs: {}, units: {} };
  }
  fs.writeFileSync(out, JSON.stringify(resp, null, 1), "utf8");
  process.exit(resp.status === "ok" ? 0 : 1);
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) main();
