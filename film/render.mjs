// Renders video.html frame by frame with the installed Chrome and pipes frames to ffmpeg.
//   node render.mjs stills 3.5 12 20         -> out/still-<t>.png
//   node render.mjs cues                     -> data/cues.json (sound-effect cue sheet)
//   node render.mjs video [fps] [workers]    -> out/video-silent.mp4
import { createServer } from 'node:http';
import { mkdir, readFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const ROOT = fileURLToPath(new URL('.', import.meta.url));
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg', '.woff2': 'font/woff2', '.svg': 'image/svg+xml' };

function serve() {
  const server = createServer(async (req, res) => {
    const path = normalize(join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname)));
    if (!path.startsWith(ROOT)) { res.writeHead(403).end(); return; }
    let body;
    try { body = await readFile(path); } catch { res.writeHead(404).end(); return; }
    res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream' }).end(body);
  });
  return new Promise(ok => server.listen(0, '127.0.0.1', () => ok(server)));
}

async function openPage(browser, port) {
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  page.on('pageerror', e => console.error('page error:', e.message));
  await page.goto(`http://127.0.0.1:${port}/video.html`);
  await page.waitForFunction(() => window.READY === true, null, { timeout: 60000 });
  return page;
}

// One browser per worker: headless Chrome only paints the front page of a browser.
async function renderRange(port, fps, from, to, file) {
  const own = await launch();
  const page = await openPage(own, port);
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'png', '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '12', '-pix_fmt', 'yuv420p', '-r', String(fps), file], { stdio: ['pipe', 'inherit', 'inherit'] });
  for (let f = from; f < to; f++) {
    await page.evaluate(t => window.seek(t), f / fps);
    const png = await page.screenshot({ type: 'png' });
    if (!ff.stdin.write(png)) await new Promise(ok => ff.stdin.once('drain', ok));
    if ((f - from) % 300 === 0) console.log(`${file}: frame ${f - from}/${to - from}`);
  }
  ff.stdin.end();
  await new Promise((ok, fail) => ff.on('close', c => (c ? fail(new Error('ffmpeg ' + c)) : ok())));
  await own.close();
}

const [mode = 'stills', ...args] = process.argv.slice(2);
await mkdir(join(ROOT, 'out'), { recursive: true });
const server = await serve();
const port = server.address().port;
const launch = () => chromium.launch({ executablePath: CHROME, args: ['--force-color-profile=srgb', '--font-render-hinting=none'] });
const browser = await launch();
try {
  if (mode === 'stills') {
    const page = await openPage(browser, port);
    for (const t of args.map(Number)) {
      await page.evaluate(t => window.seek(t), t);
      await page.screenshot({ path: join(ROOT, `out/still-${t}.png`) });
      console.log(`out/still-${t}.png`);
    }
  } else if (mode === 'cues') {
    // the sound-effect cue sheet, for audio.py sfx
    const page = await openPage(browser, port);
    const sheet = await page.evaluate(() => ({ total: window.TOTAL, music_at: window.MUSIC_AT, cues: window.CUES }));
    await (await import('node:fs/promises')).writeFile(join(ROOT, 'data/cues.json'), JSON.stringify(sheet, null, 1));
    console.log(`data/cues.json ${sheet.cues.length} cues`);
  } else if (mode === 'video') {
    const fps = Number(args[0] || 60), workers = Number(args[1] || 4);
    const page = await openPage(browser, port);
    const total = Number(process.env.LIMIT_SECONDS) || await page.evaluate(() => window.TOTAL);
    await page.close();
    const frames = Math.round(total * fps), per = Math.ceil(frames / workers);
    const parts = [...Array(workers)].map((_, i) => join(ROOT, `out/part-${i}.mp4`));
    await Promise.all(parts.map((file, i) => renderRange(port, fps, i * per, Math.min(frames, (i + 1) * per), file)));
    const list = join(ROOT, 'out/parts.txt');
    await (await import('node:fs/promises')).writeFile(list, parts.map(p => `file '${p}'`).join('\n'));
    await new Promise((ok, fail) => spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', join(ROOT, 'out/video-silent.mp4')], { stdio: 'inherit' })
      .on('close', c => (c ? fail(new Error('concat ' + c)) : ok())));
    console.log(`out/video-silent.mp4 ${total.toFixed(2)} s at ${fps} fps`);
  }
} finally {
  await browser.close();
  server.close();
}
