"""Orientation labelling for real captures (benchmark A1).

Label = the clockwise angle by which the page is turned, the same convention as ODB-any: an upright page is 0,
a page turned a quarter turn clockwise (text top pointing right) is 90, upside down is 180.

Labelling with a ruler: (1) turn the blue guide lines until they run along the rows of text (drag on the photo,
scroll, or [ ]); (2) press the arrow the top of the text points to; (3) check the corrected-page preview, Enter.
The ruler gives the tilt (lines carry no direction), the arrow picks which of the two directions is the top:
theta = the angle congruent to the ruler angle modulo 180 that is closest to the arrow's quarter turn.
Fully keyboard-driven: A / D turn the ruler 1 deg (hold to repeat; Shift 0.1, Alt 5), [ ] 0.5 deg, 0 reset,
arrows pick the top of the text, Enter or Space save and next, X skip (blank or illegible), V vertical script,
M mixed orientations, Backspace or B back, Tab or N next unlabelled, G guide lines on or off.
Labels are appended to a JSONL file; the last label per image wins, so relabelling is safe.
Usage: python label_tool.py --images DIR --out labels.jsonl [--port 8777]   then open http://127.0.0.1:8777
"""
import argparse, json, math, mimetypes, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>LevelHead labels</title><style>
:root{--ink:#1E2A38;--muted:#5B6B7F;--blue:#1F5FAD;--soft:#EAF1FB;--line:#C9D9EE;--bg:#F6F8FB;--good:#1E7B3C;--accent:#E07A1F}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 -apple-system,"Helvetica Neue",Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink);height:100vh;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:16px;padding:10px 18px;background:#fff;border-bottom:1px solid var(--line)}
header h1{font-size:15px;margin:0;font-weight:650}#bar{flex:1;height:8px;background:var(--soft);border-radius:4px;overflow:hidden}#fill{height:100%;background:var(--blue);width:0}
.stat{color:var(--muted);white-space:nowrap}main{flex:1;display:flex;min-height:0}
#left{flex:1;position:relative;display:flex;align-items:center;justify-content:center;padding:14px;min-width:0}
#left canvas{max-width:100%;max-height:100%;background:#fff;box-shadow:0 1px 6px #0002;border-radius:4px;cursor:ew-resize;touch-action:none}
aside{width:360px;background:#fff;border-left:1px solid var(--line);display:flex;flex-direction:column;gap:12px;padding:14px;overflow:auto}
.card{border:1px solid var(--line);border-radius:10px;padding:12px}.card h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 8px;font-weight:650}
#prev{width:100%;background:#fff;border-radius:6px;border:1px solid var(--line)}
#angle{font-size:32px;font-weight:650;font-variant-numeric:tabular-nums}#angle small{font-size:13px;color:var(--muted);font-weight:500;margin-left:6px}
#state{font-weight:600}#state.ok{color:var(--good)}#state.no{color:var(--muted)}
ol{margin:0;padding-left:20px}ol li{margin-bottom:6px}.keys{display:grid;grid-template-columns:auto 1fr;gap:5px 10px;align-items:center;margin-top:8px}
kbd{font:12px ui-monospace,Menlo,monospace;background:var(--bg);border:1px solid var(--line);border-bottom-width:2px;border-radius:5px;padding:1px 6px;text-align:center;min-width:24px;display:inline-block}
.arrows{display:grid;grid-template-columns:repeat(3,44px);grid-template-rows:repeat(2,36px);gap:4px;justify-content:center;margin:6px 0}
.arrows button{font-size:18px;padding:0}.arrows .on{background:var(--blue);color:#fff;border-color:var(--blue)}
.btns{display:flex;gap:8px;flex-wrap:wrap}button{font:inherit;border:1px solid var(--line);background:#fff;border-radius:7px;padding:6px 10px;cursor:pointer}button.primary{background:var(--blue);color:#fff;border-color:var(--blue)}
#name{font:12px ui-monospace,Menlo,monospace;color:var(--muted);word-break:break-all}#toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:var(--ink);color:#fff;padding:7px 14px;border-radius:8px;opacity:0;transition:opacity .2s}
</style></head><body>
<header><h1>LevelHead orientation labels</h1><span class="stat" id="pos"></span><div id="bar"><div id="fill"></div></div><span class="stat" id="speed"></span></header>
<main><div id="left"><canvas id="c"></canvas></div>
<aside>
 <div class="card"><h2>Corrected page: must read upright, rows on the lines</h2><canvas id="prev"></canvas></div>
 <div class="card"><h2>Label</h2><div id="angle">0.0&deg;<small></small></div><div id="state" class="no">not labelled</div><div id="name"></div></div>
 <div class="card"><h2>Three steps</h2><ol>
  <li><b>Align the ruler.</b> Turn the blue guide lines until they run <i>along the rows of text</i>: <kbd>A</kbd> <kbd>D</kbd> on the keyboard, or drag left/right on the photo, or scroll. Already level? Skip this.</li>
  <li><b>Where is the top of the text?</b> Press the arrow it points to.</li>
  <li><b>Check the preview</b>, then <kbd>Enter</kbd>.</li></ol>
  <div class="arrows"><span></span><button id="a0" onclick="setQ(0)">&uarr;</button><span></span><button id="a270" onclick="setQ(270)">&larr;</button><button id="a180" onclick="setQ(180)">&darr;</button><button id="a90" onclick="setQ(90)">&rarr;</button></div>
  <div class="keys"><span><kbd>A</kbd><kbd>D</kbd></span><span>turn ruler &minus;/+1&deg; (hold to repeat; Shift 0.1&deg;, Alt 5&deg;)</span>
  <span><kbd>[</kbd><kbd>]</kbd></span><span>turn ruler 0.5&deg;</span><kbd>0</kbd><span>reset ruler to level</span>
  <span><kbd>&uarr;</kbd><kbd>&rarr;</kbd><kbd>&darr;</kbd><kbd>&larr;</kbd></span><span>top of text points up / right / down / left</span>
  <span><kbd>Enter</kbd></span><span>save and next (or Space)</span><kbd>X</kbd><span>skip: blank or illegible</span><kbd>V</kbd><span>vertical script</span><kbd>M</kbd><span>mixed orientations</span>
  <kbd>&#9003;</kbd><span>back (Backspace or B)</span><kbd>Tab</kbd><span>next unlabelled (or N)</span><kbd>G</kbd><span>guide lines on / off</span></div></div>
 <div class="btns"><button onclick="back()">&larr; Back</button><button onclick="nextUnl()">Next unlabelled</button><button class="primary" onclick="save()">Save &amp; next</button></div>
</aside></main><div id="toast"></div><script>
let files=[],labels={},i=0,phi=0,q=0,grid=true,img=new Image(),t0=Date.now(),n0=0,drag=null;
const c=document.getElementById('c'),x=c.getContext('2d'),pv=document.getElementById('prev'),px=pv.getContext('2d');
const $=id=>document.getElementById(id),rad=d=>d*Math.PI/180,norm=a=>((a%360)+360)%360;
const dist=(a,b)=>{const d=Math.abs(norm(a)-norm(b));return Math.min(d,360-d)};
// ruler lines carry no direction: the label is the angle congruent to phi (mod 180) closest to the chosen quarter turn
const theta=()=>{const a=norm(phi),b=norm(phi+180);return Math.round((dist(a,q)<=dist(b,q)?a:b)*10)/10};
function toast(s){const t=$('toast');t.textContent=s;t.style.opacity=1;clearTimeout(t.h);t.h=setTimeout(()=>t.style.opacity=0,900)}
function drawMain(){c.width=img.width;c.height=img.height;x.drawImage(img,0,0);const W=img.width,H=img.height,L=Math.hypot(W,H),w=Math.max(2,W/450),step=H/14;
 x.save();x.translate(W/2,H/2);x.rotate(rad(phi));x.strokeStyle='rgba(31,95,173,.55)';x.lineWidth=w*.8;
 if(grid)for(let y=-L;y<=L;y+=step){x.beginPath();x.moveTo(-L,y);x.lineTo(L,y);x.stroke()}
 x.strokeStyle='#E07A1F';x.lineWidth=w*2;x.beginPath();x.moveTo(-L,0);x.lineTo(L,0);x.stroke();   // the ruler itself, with ticks
 x.lineWidth=w;for(let t=-L;t<=L;t+=step/2){const h=(Math.round(t/(step/2))%2)?w*4:w*8;x.beginPath();x.moveTo(t,-h);x.lineTo(t,h);x.stroke()}
 // the top-of-text marker: an arrow perpendicular to the ruler, pointing to the chosen top
 const up=rad(theta()-phi-90);x.fillStyle='#E07A1F';x.beginPath();x.arc(0,0,w*5,0,7);x.fill();x.lineWidth=w*2.5;x.beginPath();x.moveTo(0,0);x.lineTo(Math.cos(up)*step*1.6,Math.sin(up)*step*1.6);x.stroke();x.restore()}
function drawPrev(){const W=pv.clientWidth||330,a=rad(theta()),cw=img.width,ch=img.height;
 const bw=Math.abs(cw*Math.cos(a))+Math.abs(ch*Math.sin(a)),bh=Math.abs(cw*Math.sin(a))+Math.abs(ch*Math.cos(a)),s=W/Math.max(bw,bh*0.9);
 pv.width=W;pv.height=Math.min(bh*s,480);px.fillStyle='#fff';px.fillRect(0,0,pv.width,pv.height);px.save();px.translate(pv.width/2,pv.height/2);px.rotate(-a);px.scale(s,s);px.drawImage(img,-cw/2,-ch/2);px.restore();
 px.strokeStyle='rgba(31,95,173,.45)';px.lineWidth=1;for(let y=pv.height/12;y<pv.height;y+=pv.height/12){px.beginPath();px.moveTo(0,y);px.lineTo(pv.width,y);px.stroke()}}
function show(){const f=files[i],lab=labels[f];$('angle').innerHTML=theta().toFixed(1)+'&deg;<small>ruler '+(Math.round(phi*10)/10)+'&deg;</small>';
 [0,90,180,270].forEach(k=>$('a'+k).className=k==q?'on':'');
 $('state').textContent=lab?('saved: '+(lab.flag||lab.theta.toFixed(1)+'°')):'not labelled';$('state').className=lab?'ok':'no';$('name').textContent=f;
 const n=Object.keys(labels).length;$('pos').textContent=(i+1)+' / '+files.length;$('fill').style.width=(100*n/files.length)+'%';
 const rate=(n-n0)/((Date.now()-t0)/60000);$('speed').textContent=n+' labelled'+(rate>0.5?' · '+rate.toFixed(1)+'/min · ~'+Math.ceil((files.length-n)/rate)+' min left':'');drawMain();drawPrev()}
function load(){const lab=labels[files[i]];if(lab&&lab.theta!=null){phi=lab.phi??lab.theta;q=lab.q??0}else{phi=0;q=0}img.onload=show;img.src='/img/'+encodeURIComponent(files[i])}
function turn(d){phi=Math.max(-90,Math.min(90,phi+d));show()}
function setQ(k){q=k;show()}
c.onpointerdown=e=>{drag={x:e.clientX,phi};c.setPointerCapture(e.pointerId)};
c.onpointermove=e=>{if(!drag)return;const r=c.getBoundingClientRect();phi=Math.max(-90,Math.min(90,drag.phi+(e.clientX-drag.x)/r.width*90));show()};
c.onpointerup=()=>drag=null;
c.onwheel=e=>{e.preventDefault();turn((e.deltaY>0?1:-1)*(e.shiftKey?0.1:0.5))};
async function save(flag){const f=files[i];const rec={image:f,theta:flag?null:theta(),phi:Math.round(phi*10)/10,q,flag:flag||null};
 const r=await fetch('/label',{method:'POST',body:JSON.stringify(rec)});if(!r.ok){toast('save failed');return}
 labels[f]=rec;toast(flag?('saved: '+flag):('saved '+rec.theta+'°'));if(i<files.length-1)i++;load()}
function back(){if(i>0){i--;load()}}function nextUnl(){const j=files.findIndex((f,k)=>k>i&&!labels[f]);if(j>=0){i=j;load()}else toast('all labelled after this one')}
const Q={arrowup:0,arrowright:90,arrowdown:180,arrowleft:270};
const code=e=>e.code||'';   // physical keys, so Shift/Alt variants and non-US layouts still work
onkeydown=e=>{const k=e.key.toLowerCase(),c=code(e);if(k in Q){setQ(Q[k]);e.preventDefault();return}
 const step=e.altKey?5:e.shiftKey?0.1:null;
 if(c=='KeyA'||c=='KeyD'){turn((c=='KeyD'?1:-1)*(step??1));e.preventDefault()}
 else if(c=='BracketLeft'||c=='BracketRight'){turn((c=='BracketRight'?1:-1)*(step??0.5));e.preventDefault()}
 else if(c=='Digit0'||c=='Numpad0'){phi=0;show()}else if(k=='enter'||k==' '){save();e.preventDefault()}
 else if(c=='KeyX')save('skip');else if(c=='KeyV')save('vertical');else if(c=='KeyM')save('mixed');else if(c=='KeyG'){grid=!grid;show()}
 else if(k=='backspace'||c=='KeyB'){back();e.preventDefault()}else if(k=='tab'||c=='KeyN'){nextUnl();e.preventDefault()}};
onresize=()=>files.length&&show();
fetch('/state').then(r=>r.json()).then(s=>{files=s.files;labels=s.labels;n0=Object.keys(labels).length;i=Math.max(0,files.findIndex(f=>!labels[f]));if(i<0)i=0;load()});
</script></body></html>"""

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--images", required=True); ap.add_argument("--out", required=True); ap.add_argument("--port", type=int, default=8777)
    a = ap.parse_args(); root = Path(a.images).resolve(); out = Path(a.out)
    scan = lambda: sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    files = scan()   # rescanned on every page load, so labelling can start while a download is still running
    labels = {}
    if out.exists():
        for line in open(out): r = json.loads(line); labels[r["image"]] = r

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def send(self, body, ctype):
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.path == "/": return self.send(PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/state":
                files[:] = scan(); return self.send(json.dumps({"files": files, "labels": labels}).encode(), "application/json")
            if self.path.startswith("/img/"):
                name = unquote(self.path[5:])
                if name in files: return self.send((root / name).read_bytes(), mimetypes.guess_type(name)[0] or "image/png")
            self.send_error(404)
        def do_POST(self):
            if self.path != "/label": return self.send_error(404)
            rec = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if rec.get("image") not in files: return self.send_error(400)
            rec["t"] = time.time()
            with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
            labels[rec["image"]] = rec; self.send(b"ok", "text/plain")

    print(f"{len(files)} images, {len(labels)} labelled; open http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()

def _theta_check():   # mirror of the page's theta(): ruler angle phi (mod 180) closest to the arrow's quarter turn q
    def theta(phi, q):
        d = lambda a, b: min(abs(a - b) % 360, 360 - abs(a - b) % 360)
        a, b = phi % 360, (phi + 180) % 360; return a if d(a, q) <= d(b, q) else b
    assert theta(0, 0) == 0 and theta(10, 180) == 190 and theta(81, 90) == 81 and theta(-9, 270) == 351 and theta(-12, 0) == 348 and theta(30, 270) == 210

if __name__ == "__main__":
    _theta_check(); main()
