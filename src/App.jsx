import React, { useState, useEffect, useRef } from "react";

const API = "http://localhost:8000";

export default function App() {
  const [files, setFiles] = useState([]);
  const [logs, setLogs] = useState([]);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [completed, setCompleted] = useState([]);
  const [folderLabel, setFolderLabel] = useState("nhieu item");
  const [suffix, setSuffix] = useState("");
  const [noLimit, setNoLimit] = useState(false);

  const wsRef = useRef(null);

  useEffect(() => {
    fetchCompleted();
    // connect websocket
    const ws = new WebSocket("ws://localhost:8000/ws/progress");
    ws.onopen = () => console.log("WS open");
    ws.onmessage = (ev) => {
      try {
        const d = JSON.parse(ev.data);
        if (d.type === "progress") {
          setProgress({ current: d.current, total: d.total });
        } else if (d.type === "log") {
          setLogs((p) => [...p, d.message]);
        } else if (d.type === "start") {
          setProgress({ current: 0, total: d.total });
        } else if (d.type === "done") {
          setLogs((p) => [...p, `Done. moved ${d.moved}`]);
          fetchCompleted();
        } else if (d.type === "error") {
          setLogs((p) => [...p, `ERROR: ${d.message}`]);
        }
      } catch (e) {
        console.error("WS parse error", e);
      }
    };
    ws.onclose = () => console.log("WS closed");
    wsRef.current = ws;
    return () => {
      try { ws.close(); } catch {}
    };
  }, []);

  async function handleUpload(e) {
    const f = e.target.files;
    if (!f || !f.length) return;
    const fd = new FormData();
    for (let i = 0; i < f.length; i++) fd.append("files", f[i]);
    const res = await fetch(API + "/api/upload", { method: "POST", body: fd });
    const j = await res.json();
    setLogs((p) => [...p, `Uploaded ${j.saved.length} files`]);
    fetchUploads();
  }

  async function fetchUploads() {
    // list of uploads is local folder; backend does not expose listing uploads,
    // so we just show the selected files earlier. Skip.
  }

  async function queueUploads() {
    const form = new FormData();
    form.append("folder_label", folderLabel);
    form.append("suffix_text", suffix);
    form.append("no_limit", noLimit ? "true" : "false");
    const res = await fetch(API + "/api/queue_from_uploads", { method: "POST", body: form });
    const j = await res.json();
    setLogs((p) => [...p, `Queued ${j.queued} files`]);
  }

  async function startPipeline() {
    const form = new FormData();
    form.append("no_limit", noLimit ? "true" : "false");
    form.append("rotate_back", "true");
    form.append("check_rotate", "true");
    const res = await fetch(API + "/api/start_pipeline", { method: "POST", body: form });
    const j = await res.json();
    if (j.started) {
      setLogs((p) => [...p, "Pipeline started"]);
    }
  }

  async function fetchCompleted() {
    const res = await fetch(API + "/api/list_completed");
    const j = await res.json();
    setCompleted(j.files || []);
  }

  async function downloadFile(name) {
    const url = API + "/api/download/" + encodeURIComponent(name);
    window.open(url, "_blank");
  }

  return (
    <div style={{ padding: 20, fontFamily: "Arial, sans-serif" }}>
      <h2>TOOL DANH SỐ XƯỜNG — Web Demo</h2>

      <section style={{ marginBottom: 12 }}>
        <label>Chọn files để upload: </label>
        <input type="file" multiple onChange={handleUpload} />
      </section>

      <section style={{ marginBottom: 12 }}>
        <label>Folder label: </label>
        <select value={folderLabel} onChange={(e) => setFolderLabel(e.target.value)}>
          <option value="nhieu item">nhieu item</option>
          <option value="kidshirt">kidshirt</option>
          <option value="1item">1item</option>
          <option value="hieuungmautoi">hieuungmautoi</option>
          <option value="khachtusua">khachtusua</option>
        </select>
        <label style={{ marginLeft: 12 }}>Hậu tố: <input value={suffix} onChange={(e)=>setSuffix(e.target.value)} /></label>
        <label style={{ marginLeft: 12 }}>
          <input type="checkbox" checked={noLimit} onChange={(e)=>setNoLimit(e.target.checked)} /> Bỏ giới hạn chiều cao
        </label>
        <div style={{ marginTop: 8 }}>
          <button onClick={queueUploads}>Queue từ uploads → TU_DONG</button>
          <button onClick={startPipeline} style={{ marginLeft: 8 }}>Start pipeline</button>
          <button onClick={fetchCompleted} style={{ marginLeft: 8 }}>Refresh Completed</button>
        </div>
      </section>

      <section style={{ marginBottom: 12 }}>
        <h4>Progress</h4>
        <div> {progress.current} / {progress.total}</div>
        <progress value={progress.current} max={progress.total || 1} style={{ width: "100%" }} />
      </section>

      <section style={{ marginBottom: 12 }}>
        <h4>Logs</h4>
        <div style={{ height: 200, overflow: "auto", border: "1px solid #ccc", padding: 8, background: "#fafafa" }}>
          {logs.map((l, i) => <div key={i}>{l}</div>)}
        </div>
      </section>

      <section>
        <h4>Completed files</h4>
        <div>
          {completed.map((f) => (
            <div key={f} style={{ marginBottom: 6 }}>
              {f} <button onClick={()=>downloadFile(f)}>Download</button>
            </div>
          ))}
          {completed.length===0 && <div>Không có file hoàn tất</div>}
        </div>
      </section>
    </div>
  );
}
