"""
flask_viewer.py — Colab インライン リアルタイムビューアー
                  Flask + Colab proxyPort を使った MuJoCo 画像ストリーミング

仕組み:
  1. Flask サーバーがバックグラウンドスレッドで動作
  2. MuJoCo Renderer がフレームを JPEG 化して /frame エンドポイントで配信
  3. JS が 50ms ごとに /frame を取得してブラウザに表示（JPEG ポーリング）
  4. Colab proxyPort で iframe URL を生成 → ngrok 不要

エンドポイント:
  GET  /          HTML ページ（カメラ映像 + コントロール）
  GET  /frame     現在フレームの JPEG 画像
  GET  /status    JSON 状態（生存時間・距離・etc）
  POST /cmd       JSON コマンド {"action": "play"|"pause"|"reset"|"vel", ...}
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from typing import Optional

import mujoco
import numpy as np
from PIL import Image

# ── パス ──────────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(__file__)
_ROOT     = os.path.join(_HERE, "..", "..")
ARENA_XML = os.path.abspath(os.path.join(_ROOT, "models", "go2", "arena.xml"))
WALK_XML  = os.path.abspath(os.path.join(_ROOT, "models", "go2", "walk_scene.xml"))

# ── 制御定数 ──────────────────────────────────────────────────────────────────
STANDING_POS = np.array([
    0, 0.9, -1.8,  # FR
    0, 0.9, -1.8,  # FL
    0, 0.9, -1.8,  # RR
    0, 0.9, -1.8,  # RL
], dtype=np.float64)
ACTION_SCALE = 0.3
KP, KD = 20.0, 0.5
MAX_TIME  = 60.0
TAG_DIST  = 0.65
ARENA_HALF = 2.35
ONI_START  = (1.5, 1.5)


class FlaskViewer:
    """MuJoCo をリアルタイムで Colab セル内に表示するビューアー。

    Parameters
    ----------
    xml_path    : MuJoCo XML ファイルパス
    camera      : カメラ名（例: 'top_view', 'side_cam', 'track_cam'）
    render_wh   : レンダリング解像度 (width, height)
    sim_fps     : シミュレーション更新頻度 (Hz)
    render_fps  : 映像フレームレート (fps)
    oni_speed   : 鬼の移動速度 (m/step)
    with_oni    : 鬼の表示・追跡を有効にする
    policy      : SB3 PPO 高レベルポリシー（省略時 = 手動操作）
    """

    def __init__(
        self,
        xml_path: str = ARENA_XML,
        camera: str = "top_view",
        render_wh: tuple[int, int] = (640, 480),
        sim_fps: int = 50,
        render_fps: int = 25,
        oni_speed: float = 0.025,
        with_oni: bool = True,
        policy=None,
    ):
        self.xml_path   = xml_path
        self.camera_name = camera
        self.render_wh  = render_wh
        self.sim_fps    = sim_fps
        self.render_fps = render_fps
        self.oni_speed  = oni_speed
        self.with_oni   = with_oni
        self.policy     = policy

        # MuJoCo モデル / データ
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # カメラ ID
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
        )
        if self._cam_id < 0:
            print(f"⚠  カメラ '{camera}' が見つかりません。デフォルトカメラを使用します。")

        # 共有状態（スレッド間）
        self._lock       = threading.Lock()
        self._frame_jpg  = b""          # 最新 JPEG バイト列
        self._running    = False
        self._vel_cmd    = np.zeros(3)
        self._last_action = np.zeros(12)
        self._sim_time   = 0.0
        self._oni_x      = float(ONI_START[0])
        self._oni_y      = float(ONI_START[1])
        self._done       = False
        self._done_reason = ""

        self._reset_sim()

    # ── シミュレーション ──────────────────────────────────────────────────────

    def _reset_sim(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.qvel[:] = 0.0
        if self.with_oni:
            self._oni_x, self._oni_y = ONI_START
            if self.model.nq > 19:
                self.data.qpos[19] = self._oni_x
                self.data.qpos[20] = self._oni_y
        mujoco.mj_forward(self.model, self.data)
        self._vel_cmd     = np.zeros(3)
        self._last_action = np.zeros(12)
        self._sim_time    = 0.0
        self._done        = False
        self._done_reason = ""

    def _step_sim(self) -> None:
        """1 制御ステップ（5 物理ステップ）進める。"""
        if self.policy is not None:
            obs = self._build_policy_obs()
            action, _ = self.policy.predict(obs, deterministic=True)
            self._vel_cmd = np.clip(action.astype(np.float64), -1, 1)

        # PD 制御
        q_target = STANDING_POS + self._last_action * ACTION_SCALE
        q_cur    = self.data.qpos[7:19]
        dq_cur   = self.data.qvel[6:18]
        tau      = KP * (q_target - q_cur) - KD * dq_cur
        lim      = self.model.actuator_ctrlrange
        self.data.ctrl[:] = np.clip(tau, lim[:, 0], lim[:, 1])

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        # 鬼を動かす
        if self.with_oni and self.model.nq > 19:
            rx, ry = self.data.qpos[0], self.data.qpos[1]
            dx = rx - self._oni_x
            dy = ry - self._oni_y
            dist = float(np.sqrt(dx**2 + dy**2))
            if dist > 0.01:
                self._oni_x = float(np.clip(
                    self._oni_x + (dx / dist) * self.oni_speed, -ARENA_HALF, ARENA_HALF))
                self._oni_y = float(np.clip(
                    self._oni_y + (dy / dist) * self.oni_speed, -ARENA_HALF, ARENA_HALF))
            self.data.qpos[19] = self._oni_x
            self.data.qpos[20] = self._oni_y
            mujoco.mj_forward(self.model, self.data)

            # 終了判定
            if dist < TAG_DIST:
                self._done, self._done_reason = True, "tagged"
            elif float(self.data.qpos[2]) < 0.15:
                self._done, self._done_reason = True, "fallen"
            elif self._sim_time >= MAX_TIME:
                self._done, self._done_reason = True, "escaped"
        else:
            if float(self.data.qpos[2]) < 0.15:
                self._done, self._done_reason = True, "fallen"

        self._sim_time += 1 / self.sim_fps

    def _build_policy_obs(self) -> np.ndarray:
        rx, ry = self.data.qpos[0], self.data.qpos[1]
        dx = self._oni_x - rx
        dy = self._oni_y - ry
        dist = float(np.sqrt(dx**2 + dy**2))
        ang_vel = self.data.qvel[3:6].copy()
        proj_grav = self._proj_gravity()
        time_left = (MAX_TIME - self._sim_time) / MAX_TIME
        return np.concatenate([[dx, dy, dist], ang_vel, proj_grav, [time_left]]).astype(np.float32)

    def _proj_gravity(self) -> np.ndarray:
        w, qx, qy, qz = self.data.qpos[3:7]
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy+w*qz),   2*(qx*qz-w*qy)],
            [2*(qx*qy-w*qz),   1-2*(qx**2+qz**2),  2*(qy*qz+w*qx)],
            [2*(qx*qz+w*qy),   2*(qy*qz-w*qx),    1-2*(qx**2+qy**2)],
        ])
        return (R.T @ np.array([0, 0, -1])).astype(np.float64)

    def _get_status(self) -> dict:
        rx, ry = float(self.data.qpos[0]), float(self.data.qpos[1])
        dist = float(np.sqrt((rx - self._oni_x)**2 + (ry - self._oni_y)**2))
        return {
            "time":    round(self._sim_time, 2),
            "rz":      round(float(self.data.qpos[2]), 3),
            "dist":    round(dist, 3),
            "done":    self._done,
            "reason":  self._done_reason,
            "running": self._running,
        }

    # ── バックグラウンドスレッド ───────────────────────────────────────────────

    def _sim_loop(self) -> None:
        """シミュレーション更新スレッド（独立した制御周期で動作）。"""
        interval = 1.0 / self.sim_fps
        while True:
            t0 = time.perf_counter()
            with self._lock:
                if self._running and not self._done:
                    self._step_sim()
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, interval - elapsed))

    def _render_loop(self) -> None:
        """レンダリングスレッド（JPEG フレーム生成）。"""
        interval = 1.0 / self.render_fps
        renderer = mujoco.Renderer(
            self.model,
            height=self.render_wh[1],
            width=self.render_wh[0],
        )
        while True:
            t0 = time.perf_counter()
            with self._lock:
                try:
                    if self._cam_id >= 0:
                        renderer.update_scene(self.data, camera=self._cam_id)
                    else:
                        renderer.update_scene(self.data)
                    rgb = renderer.render()
                except Exception:
                    pass
                else:
                    buf = io.BytesIO()
                    Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
                    self._frame_jpg = buf.getvalue()
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, interval - elapsed))

    # ── Flask サーバー ────────────────────────────────────────────────────────

    def _build_flask_app(self):
        from flask import Flask, Response, request, jsonify
        app = Flask(__name__)
        app.logger.disabled = True

        # ── silence Flask console logs ──
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        viewer = self  # closure

        @app.route("/")
        def index():
            return viewer._html_page()

        @app.route("/frame")
        def frame():
            jpg = viewer._frame_jpg
            return Response(jpg, mimetype="image/jpeg",
                            headers={"Cache-Control": "no-store"})

        @app.route("/status")
        def status():
            with viewer._lock:
                st = viewer._get_status()
            return jsonify(st)

        @app.route("/cmd", methods=["POST"])
        def cmd():
            data = request.get_json(force=True, silent=True) or {}
            action = data.get("action", "")
            with viewer._lock:
                if action == "play":
                    viewer._running = True
                elif action == "pause":
                    viewer._running = False
                elif action == "reset":
                    viewer._reset_sim()
                    viewer._running = True
                elif action == "vel":
                    viewer._vel_cmd = np.array([
                        float(data.get("vx",    0)),
                        float(data.get("vy",    0)),
                        float(data.get("omega", 0)),
                    ])
                elif action == "oni_speed":
                    viewer.oni_speed = float(data.get("value", viewer.oni_speed))
            return jsonify({"ok": True})

        return app

    def _start_server(self, port: int) -> None:
        app = self._build_flask_app()
        app.run(host="0.0.0.0", port=port, threaded=True,
                use_reloader=False, debug=False)

    # ── HTML ページ ────────────────────────────────────────────────────────────

    def _html_page(self) -> str:
        has_oni = "true" if (self.with_oni and self.model.nq > 19) else "false"
        has_policy = "true" if self.policy else "false"
        return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0d0d14; color:#e0e0f0;
       font-family:'Segoe UI',system-ui,sans-serif;
       display:flex; flex-direction:column; height:100vh; overflow:hidden; }}
#topbar {{ background:#13131f; border-bottom:1px solid #2a2a40;
           padding:6px 12px; display:flex; align-items:center; gap:12px; flex-shrink:0; }}
#topbar h2 {{ font-size:.9rem; color:#a0c0ff; }}
#topbar .badge {{ background:#1e3060; color:#7090e0; padding:2px 8px;
                  border-radius:10px; font-size:.72rem; }}
#fps-disp {{ margin-left:auto; font-size:.72rem; color:#556; }}
#main {{ display:flex; flex:1; overflow:hidden; }}
#video-wrap {{ flex:1; display:flex; align-items:center; justify-content:center;
               background:#080810; position:relative; }}
#sim-img {{ max-width:100%; max-height:100%; object-fit:contain; display:block; }}
#result-overlay {{
  position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
  background:rgba(10,10,20,.92); border:2px solid #4080ff; border-radius:12px;
  padding:22px 36px; text-align:center; display:none;
}}
#result-overlay.show {{ display:block; }}
#result-title {{ font-size:1.6rem; font-weight:700; }}
#result-sub   {{ font-size:.85rem; color:#8899bb; margin-top:6px; }}
#panel {{ width:200px; background:#13131f; border-left:1px solid #2a2a40;
          display:flex; flex-direction:column; overflow-y:auto; flex-shrink:0; }}
.sec {{ padding:10px; border-bottom:1px solid #1e1e30; }}
.sec h3 {{ font-size:.65rem; text-transform:uppercase; letter-spacing:.08em;
           color:#4466aa; margin-bottom:8px; }}
.stat {{ display:flex; justify-content:space-between; font-size:.78rem; margin-bottom:4px; }}
.stat .lbl {{ color:#778; }}
.stat .val {{ color:#c0d8ff; font-weight:600; font-variant-numeric:tabular-nums; }}
.val.danger {{ color:#ff6060; }}
.val.good   {{ color:#60e060; }}
.pbar {{ background:#1e1e30; border-radius:3px; height:5px; margin-bottom:6px; overflow:hidden; }}
#timebar {{ height:100%; background:linear-gradient(90deg,#4080ff,#40c0ff); transition:width .12s; }}
.btn {{ width:100%; padding:7px 0; border:1px solid #2a3a5a; border-radius:5px;
        background:#1a2440; color:#a0c0ff; cursor:pointer; font-size:.82rem;
        margin-bottom:5px; transition:background .12s; }}
.btn:hover {{ background:#22305a; }}
.btn.primary {{ background:#1e4080; border-color:#4080ff; font-weight:600; }}
.btn.primary:hover {{ background:#2a52a0; }}
.btn-row {{ display:flex; gap:4px; }}
.btn-row .btn {{ flex:1; margin-bottom:5px; }}
.dpad {{ display:grid; grid-template-columns:repeat(3,1fr); gap:3px; margin-bottom:6px; }}
.dpad .btn {{ margin-bottom:0; padding:8px 0; }}
.slider-wrap label {{
  display:flex; justify-content:space-between; font-size:.72rem; color:#889; margin-bottom:2px;
}}
.slider-wrap label span {{ color:#aac; font-weight:600; }}
input[type=range] {{ width:100%; accent-color:#4080ff; }}
</style>
</head>
<body>
<div id="topbar">
  <h2>🐾 RoboQuest 2026</h2>
  <span class="badge" id="mode-badge">{('🤖 AI モード' if self.policy else '🕹 手動モード')}</span>
  <span id="fps-disp">-- fps</span>
</div>
<div id="main">
  <div id="video-wrap">
    <img id="sim-img" src="/frame" alt="loading...">
    <div id="result-overlay">
      <div id="result-title">--</div>
      <div id="result-sub">--</div>
    </div>
  </div>
  <div id="panel">

    <div class="sec">
      <h3>📊 ステータス</h3>
      <div class="stat"><span class="lbl">生存時間</span><span class="val" id="v-time">0.0 s</span></div>
      <div class="pbar"><div id="timebar" style="width:0%"></div></div>
      <div class="stat"><span class="lbl">鬼との距離</span><span class="val" id="v-dist">-- m</span></div>
      <div class="stat"><span class="lbl">高さ</span><span class="val" id="v-height">-- m</span></div>
      <div class="stat"><span class="lbl">状態</span><span class="val" id="v-state">待機中</span></div>
    </div>

    <div class="sec">
      <h3>⏯ シミュレーション</h3>
      <div class="btn-row">
        <button class="btn primary" id="btn-play">▶ 開始</button>
        <button class="btn" id="btn-reset">↺</button>
      </div>
      <div class="slider-wrap">
        <label>表示fps上限 <span id="fps-lbl">25</span></label>
        <input type="range" id="fps-slider" min="5" max="30" step="5" value="25">
      </div>
    </div>

    <div class="sec" id="manual-sec" style="{('display:none' if self.policy else '')}">
      <h3>🕹 手動操作 (WASD)</h3>
      <div class="dpad">
        <div></div>
        <button class="btn" data-vx="1"  data-vy="0" data-om="0" >↑</button>
        <div></div>
        <button class="btn" data-vx="0"  data-vy="0" data-om="1" >←</button>
        <button class="btn" id="btn-stop">■</button>
        <button class="btn" data-vx="0"  data-vy="0" data-om="-1">→</button>
        <div></div>
        <button class="btn" data-vx="-1" data-vy="0" data-om="0" >↓</button>
        <div></div>
      </div>
    </div>

    <div class="sec" id="oni-sec" style="{('display:none' if not (self.with_oni and self.model.nq > 19) else '')}">
      <h3>👹 鬼の速度</h3>
      <div class="slider-wrap">
        <label>速度 <span id="oni-lbl">{self.oni_speed:.3f}</span></label>
        <input type="range" id="oni-slider"
               min="0.005" max="0.08" step="0.005" value="{self.oni_speed}">
      </div>
    </div>

  </div>
</div>

<script>
const MAX_TIME = {MAX_TIME};
const TAG_DIST = {TAG_DIST};
const HAS_POLICY = {has_policy};

let playing = false;
let frameInterval = null;
let statusInterval = null;
let frameDelay = Math.round(1000 / 25);  // ms
let frameCount = 0;
let lastFpsTime = Date.now();

// ── フレーム取得 ──────────────────────────────────────────────────────────────
function startStream() {{
  if (frameInterval) clearInterval(frameInterval);
  frameInterval = setInterval(fetchFrame, frameDelay);
}}

function fetchFrame() {{
  const img = document.getElementById('sim-img');
  const newSrc = '/frame?t=' + Date.now();
  const tmp = new Image();
  tmp.onload = () => {{
    img.src = tmp.src;
    frameCount++;
    const now = Date.now();
    if (now - lastFpsTime >= 1000) {{
      const fps = Math.round(frameCount * 1000 / (now - lastFpsTime));
      document.getElementById('fps-disp').textContent = fps + ' fps';
      frameCount = 0;
      lastFpsTime = now;
    }}
  }};
  tmp.src = newSrc;
}}

// ── ステータス取得 ────────────────────────────────────────────────────────────
statusInterval = setInterval(async () => {{
  try {{
    const res = await fetch('/status');
    const st  = await res.json();
    updateUI(st);
  }} catch(e) {{}}
}}, 200);

function updateUI(st) {{
  const pct = Math.min(100, st.time / MAX_TIME * 100);
  document.getElementById('v-time').textContent   = st.time.toFixed(1) + ' s';
  document.getElementById('timebar').style.width  = pct + '%';
  document.getElementById('v-dist').textContent   = st.dist.toFixed(2) + ' m';
  document.getElementById('v-height').textContent = st.rz.toFixed(3) + ' m';

  const distEl  = document.getElementById('v-dist');
  const stateEl = document.getElementById('v-state');
  distEl.className = 'val' + (st.dist < 1.0 ? ' danger' : st.dist > 2 ? ' good' : '');

  if (st.done) {{
    const ok = st.reason === 'escaped';
    document.getElementById('result-overlay').classList.add('show');
    document.getElementById('result-overlay').style.borderColor = ok ? '#40c060' : '#ff4040';
    document.getElementById('result-title').textContent =
        ok ? '🎉 逃げ切り成功！' : st.reason === 'tagged' ? '😱 タグされた！' : '😵 転倒！';
    document.getElementById('result-title').style.color = ok ? '#80ff80' : '#ff8080';
    document.getElementById('result-sub').textContent = '生存時間: ' + st.time.toFixed(1) + ' 秒';
    stateEl.textContent = ok ? '🎉 成功' : (st.reason === 'tagged' ? '😱 タグ' : '😵 転倒');
  }} else if (!st.running) {{
    stateEl.textContent = '停止中';
    stateEl.className = 'val';
  }} else if (st.dist < TAG_DIST * 1.5) {{
    stateEl.textContent = '🔴 危険！';
    stateEl.className = 'val danger';
  }} else {{
    stateEl.textContent = '🟢 逃走中';
    stateEl.className = 'val good';
  }}
}}

// ── コマンド送信 ─────────────────────────────────────────────────────────────
function sendCmd(obj) {{
  fetch('/cmd', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(obj),
  }}).catch(() => {{}});
}}

// ── ボタン ────────────────────────────────────────────────────────────────────
const playBtn = document.getElementById('btn-play');
playBtn.addEventListener('click', () => {{
  playing = !playing;
  playBtn.textContent = playing ? '⏸ 停止' : '▶ 開始';
  sendCmd({{action: playing ? 'play' : 'pause'}});
  document.getElementById('result-overlay').classList.remove('show');
}});

document.getElementById('btn-reset').addEventListener('click', () => {{
  playing = true;
  playBtn.textContent = '⏸ 停止';
  sendCmd({{action: 'reset'}});
  document.getElementById('result-overlay').classList.remove('show');
}});

document.getElementById('btn-stop')?.addEventListener('click', () => {{
  sendCmd({{action: 'vel', vx: 0, vy: 0, omega: 0}});
}});

// 方向ボタン
document.querySelectorAll('.dpad .btn[data-vx]').forEach(btn => {{
  const s = 0.8;
  const go = () => sendCmd({{
    action: 'vel',
    vx:    parseFloat(btn.dataset.vx) * s,
    vy:    parseFloat(btn.dataset.vy) * s,
    omega: parseFloat(btn.dataset.om) * s,
  }});
  const stop = () => sendCmd({{action: 'vel', vx: 0, vy: 0, omega: 0}});
  btn.addEventListener('pointerdown', go);
  btn.addEventListener('pointerup',   stop);
  btn.addEventListener('pointerleave',stop);
}});

// WASD キーボード
if (!HAS_POLICY) {{
  const keys = new Set();
  const applyKeys = () => {{
    const s = 0.8;
    sendCmd({{
      action: 'vel',
      vx:    (keys.has('w') ? s : 0) + (keys.has('s') ? -s : 0),
      vy:    (keys.has('q') ? s*0.5 : 0) + (keys.has('e') ? -s*0.5 : 0),
      omega: (keys.has('a') ? s : 0) + (keys.has('d') ? -s : 0),
    }});
  }};
  document.addEventListener('keydown', e => {{
    keys.add(e.key.toLowerCase()); applyKeys();
    if (e.key === ' ') {{ e.preventDefault(); playBtn.click(); }}
    if (e.key.toLowerCase() === 'r') document.getElementById('btn-reset').click();
  }});
  document.addEventListener('keyup', e => {{
    keys.delete(e.key.toLowerCase()); applyKeys();
  }});
}}

// fps スライダー
document.getElementById('fps-slider').addEventListener('input', e => {{
  const v = parseInt(e.target.value);
  document.getElementById('fps-lbl').textContent = v;
  frameDelay = Math.round(1000 / v);
  startStream();
}});

// 鬼速度スライダー
document.getElementById('oni-slider')?.addEventListener('input', e => {{
  const v = parseFloat(e.target.value);
  document.getElementById('oni-lbl').textContent = v.toFixed(3);
  sendCmd({{action: 'oni_speed', value: v}});
}});

// ── 起動 ─────────────────────────────────────────────────────────────────────
startStream();
fetchFrame(); // 初回即座に取得
</script>
</body>
</html>"""

    # ── 公開 API ──────────────────────────────────────────────────────────────

    def display(self, port: int = 7860, height: int = 540) -> None:
        """Colab セル内にビューアーを表示する。

        Parameters
        ----------
        port   : 希望ポート番号（使用中の場合は自動で次のポートを探す）
        height : IFrame の高さ (px)
        """
        import socket

        # ── 空きポートを自動検出 ────────────────────────────────────────────
        actual_port = None
        for p in range(port, port + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if s.connect_ex(('127.0.0.1', p)) != 0:   # 接続できない = 空き
                    actual_port = p
                    break
        if actual_port is None:
            print(f"❌ ポート {port}〜{port+19} がすべて使用中です。Colab ランタイムを再起動してください。")
            return

        if actual_port != port:
            print(f"⚠  ポート {port} は使用中のため {actual_port} を使用します。")

        # ── バックグラウンドスレッド起動 ────────────────────────────────────
        threading.Thread(target=self._sim_loop,    daemon=True).start()
        threading.Thread(target=self._render_loop, daemon=True).start()
        time.sleep(0.6)   # 最初のフレームが生成されるまで待機

        threading.Thread(
            target=self._start_server, args=(actual_port,), daemon=True
        ).start()
        time.sleep(1.2)   # Flask 起動待機

        # ── Colab proxy URL を取得して IFrame 表示 ──────────────────────────
        try:
            from google.colab.output import eval_js
            from IPython.display import IFrame, display as ipy_display
            proxy_url = eval_js(f'google.colab.kernel.proxyPort({actual_port})')
            print(f"✅ ビューアー起動 → {proxy_url}")
            print("   ↑ リンクを新しいタブで開くか、下の IFrame で確認できます")
            ipy_display(IFrame(src=proxy_url, width="100%", height=height))
        except ImportError:
            # Colab 外（ローカル実行）では URL を表示
            print(f"✅ ビューアー起動 → http://localhost:{actual_port}/")
            print("   ブラウザで上記URLを開いてください。")
