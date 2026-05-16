"""
Web interface for the Plato-MIDI Bridge.
Serves real-time PLATO room visualization + MIDI generation.
"""

import json
import http.server
import socketserver
import threading
from typing import Optional

PORT = 9710
HTML = None

def get_html() -> str:
    global HTML
    if HTML:
        return HTML
    
    HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Plato-MIDI Bridge</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { 
    background: #0a0a0f; color: #c0c0d0; font-family: 'Courier New', monospace;
    display: flex; flex-direction: column; height: 100vh;
  }
  .header { 
    padding: 12px 20px; border-bottom: 1px solid #2a2a3f;
    display: flex; justify-content: space-between; align-items: center;
  }
  .header h1 { font-size: 14px; font-weight: normal; color: #7070a0; }
  .header .status { font-size: 11px; color: #50c050; }
  .main { display: flex; flex: 1; overflow: hidden; }
  .canvas-panel { flex: 1; position: relative; background: #0d0d14; }
  canvas { width: 100%; height: 100%; display: block; }
  .side-panel { width: 320px; border-left: 1px solid #2a2a3f; overflow-y: auto; padding: 12px; }
  .room-card { 
    background: #12121a; border: 1px solid #2a2a3f; border-radius: 4px;
    padding: 10px; margin-bottom: 8px; cursor: pointer;
    transition: border-color 0.3s;
  }
  .room-card:hover { border-color: #5050a0; }
  .room-card.active { border-color: #50c050; }
  .room-name { font-size: 12px; color: #9090b0; margin-bottom: 4px; }
  .room-chamber { font-size: 18px; color: #d0d0f0; }
  .room-meta { font-size: 10px; color: #606080; margin-top: 4px; display: flex; gap: 12px; }
  .tminus-section { margin-top: 16px; }
  .tminus-section h3 { font-size: 11px; color: #606080; margin-bottom: 8px; }
  .event-card {
    background: #12121a; border: 1px solid #2a2a3f; border-radius: 4px;
    padding: 8px; margin-bottom: 6px; font-size: 11px;
  }
  .event-name { color: #9090b0; }
  .event-time { color: #606080; font-size: 10px; }
  .event-resolved { color: #50c050; }
  .event-pending { color: #c0c050; }
  .tension-bar { 
    height: 3px; background: #1a1a2a; border-radius: 2px; margin-top: 4px;
  }
  .tension-fill { height: 100%; border-radius: 2px; transition: width 0.5s; }
  .footer { 
    padding: 8px 20px; border-top: 1px solid #2a2a3f;
    font-size: 10px; color: #404060; display: flex; justify-content: space-between;
  }
  .piano-roll { 
    height: 80px; border-top: 1px solid #2a2a3f; position: relative;
    overflow: hidden; background: #08080f;
  }
  .piano-note {
    position: absolute; height: 6px; border-radius: 1px;
    transition: left 0.5s;
  }
  @keyframes pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
  }
  .listening { animation: pulse 2s infinite; }
</style>
</head>
<body>
<div class="header">
  <h1>plato-midi-bridge v0.1.0</h1>
  <span class="status" id="status">● connecting</span>
</div>
<div class="main">
  <div class="canvas-panel">
    <canvas id="lattice"></canvas>
  </div>
  <div class="side-panel" id="side-panel">
    <div style="font-size:11px;color:#606080;margin-bottom:12px;">ROOMS</div>
    <div id="room-list"></div>
    <div class="tminus-section">
      <h3>T-MINUS EVENTS (predictions → tension)</h3>
      <div id="event-list"></div>
      <div class="tension-bar">
        <div class="tension-fill" id="tension-fill" style="width:0%;background:#5050a0;"></div>
      </div>
    </div>
  </div>
</div>
<div class="piano-roll" id="piano-roll"></div>
<div class="footer">
  <span id="tick-count">tick: awaiting data</span>
  <span id="note-count">notes: 0</span>
  <span id="refresh-time"></span>
</div>

<script>
const canvas = document.getElementById('lattice');
const ctx = canvas.getContext('2d');
let animFrame = null;
let data = null;
let tick = 0;

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
}
window.addEventListener('resize', resize);
resize();

const CHAMBERS = [
  {x:0.5, y:0.05, label:'C'}, {x:0.8, y:0.15, label:'C#'}, {x:0.95, y:0.35, label:'D'},
  {x:0.95, y:0.6, label:'D#'}, {x:0.8, y:0.8, label:'E'}, {x:0.55, y:0.9, label:'F'},
  {x:0.3, y:0.85, label:'F#'}, {x:0.1, y:0.7, label:'G'}, {x:0.05, y:0.45, label:'G#'},
  {x:0.15, y:0.2, label:'A'}, {x:0.35, y:0.08, label:'A#'}, {x:0.55, y:0.5, label:'B'},
];

const CHAMBER_COLORS = [
  '#404080','#604070','#606040','#706040','#408050','#405080',
  '#804040','#408060','#604080','#508040','#804060','#60a0a0'
];

function drawLattice(w, h) {
  ctx.clearRect(0, 0, w, h);
  const cx = w/2, cy = h/2, r = Math.min(w, h) * 0.38;

  // Draw chamber nodes
  CHAMBERS.forEach((ch, i) => {
    const x = cx + (ch.x - 0.5) * r * 2;
    const y = cy + (ch.y - 0.5) * r * 2;
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = CHAMBER_COLORS[i];
    ctx.fill();
    ctx.fillStyle = '#8080a0';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(ch.label, x, y - 14);
  });

  // Draw rooms as orbiting points
  if (data && data.rooms) {
    data.rooms.forEach((room, i) => {
      const ch = room.chamber || 0;
      const angle = (ch / 12) * Math.PI * 2 - Math.PI/2;
      const dist = r * 0.85;
      const x = cx + Math.cos(angle) * dist;
      const y = cy + Math.sin(angle) * dist;
      const vel = (room.velocity || 60) / 127;
      
      // Orbital trail
      ctx.beginPath();
      ctx.arc(x, y, 4 + vel * 8, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${80+vel*175}, ${80+vel*100}, ${160+vel*95}, ${0.3+vel*0.5})`;
      ctx.fill();
      
      // Room name
      ctx.fillStyle = '#9090b0';
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(room.name.replace('oracle1-','o1-').replace('forgemaster-','fm-'), x, y + vel*12 + 14);
    });
  }
}

function updateUI() {
  const status = document.getElementById('status');
  const roomList = document.getElementById('room-list');
  const eventList = document.getElementById('event-list');
  const tensionFill = document.getElementById('tension-fill');
  const tickCount = document.getElementById('tick-count');
  const noteCount = document.getElementById('note-count');
  const refreshTime = document.getElementById('refresh-time');
  const pianoRoll = document.getElementById('piano-roll');

  if (!data) {
    status.textContent = '● waiting for data';
    status.className = 'status';
    return;
  }

  status.textContent = '● listening';
  status.className = 'status listening';
  tickCount.textContent = `tick: ${data.rooms} rooms, ${data.t_minus_events?.length || 0} events`;
  noteCount.textContent = `notes: ${data.midi_notes || 0}`;
  refreshTime.textContent = new Date().toLocaleTimeString();

  // Tension bar
  if (data.tension !== undefined) {
    const pct = Math.min(100, data.tension * 100);
    tensionFill.style.width = pct + '%';
    tensionFill.style.background = pct > 50 ? '#c05050' : pct > 25 ? '#c0c050' : '#50c050';
  }

  // Room cards
  roomList.innerHTML = '';
  (data.room_details || []).forEach(room => {
    const card = document.createElement('div');
    card.className = 'room-card';
    card.innerHTML = `
      <div class="room-name">${room.name}</div>
      <div class="room-chamber">♯ ${room.chamber}</div>
      <div class="room-meta">
        <span>vel: ${room.velocity}</span>
        <span>gap: ${room.gap}</span>
      </div>
    `;
    roomList.appendChild(card);
  });

  // Event cards
  eventList.innerHTML = '';
  (data.t_minus_events || []).forEach(ev => {
    const card = document.createElement('div');
    card.className = 'event-card';
    const isResolved = ev.status === 'resolved';
    card.innerHTML = `
      <div class="event-name">${ev.name}</div>
      <div class="event-time">
        predicted: T-${ev.predicted}h 
        ${isResolved ? `<span class="event-resolved">→ actual: ${ev.actual}h ✓</span>` 
                     : `<span class="event-pending">→ pending</span>`}
      </div>
    `;
    eventList.appendChild(card);
  });

  // Piano roll
  pianoRoll.innerHTML = '';
  const pw = pianoRoll.offsetWidth || 400;
  for (let i = 0; i < 20; i++) {
    const note = document.createElement('div');
    note.className = 'piano-note';
    const x = Math.random() * pw;
    const y = Math.random() * 64;
    const w = Math.random() * 60 + 10;
    note.style.cssText = `left:${x}px;top:${y}px;width:${w}px;background:hsl(${Math.random()*360},50%,50%)`;
    pianoRoll.appendChild(note);
  }
}

function fetchData() {
  fetch('/api/state')
    .then(r => r.json())
    .then(d => { data = d; tick++; updateUI(); })
    .catch(() => { status.textContent = '● disconnected'; status.className = 'status'; });
}

function animate() {
  const rect = canvas.parentElement.getBoundingClientRect();
  drawLattice(rect.width, rect.height);
  animFrame = requestAnimationFrame(animate);
}
animate();
setInterval(fetchData, 3000);
fetchData();
</script>
</body>
</html>"""
    return HTML


class WebHandler(http.server.SimpleHTTPRequestHandler):
    engine = None

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(get_html().encode())
        elif self.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            if self.engine:
                s = self.engine.summary()
                self.wfile.write(json.dumps(s).encode())
            else:
                self.wfile.write(json.dumps({"status": "no engine"}).encode())
        elif self.path == '/api/export':
            self.send_response(200)
            self.send_header('Content-Type', 'audio/midi')
            self.send_header('Content-Disposition', 'attachment; filename="plato-current-song.mid"')
            self.end_headers()
            if self.engine and self.engine.current_midi:
                self.wfile.write(self.engine.current_midi.to_bytes(include_header=True))
            else:
                self.wfile.write(b'')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # quiet


def serve_web_interface(engine, port: int = PORT):
    """Start the web server in a background thread."""
    WebHandler.engine = engine
    server = socketserver.TCPServer(("", port), WebHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[web] Plato-MIDI Bridge at http://localhost:{port}")
    return server
