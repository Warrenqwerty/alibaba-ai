from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import urlparse


APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Local Region BBox Annotator</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --blue: #2563eb;
      --ink: #172033;
      --muted: #667085;
      --line: #d8dee8;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 4;
    }
    h1 {
      font-size: 17px;
      margin: 0;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
      padding: 16px;
      min-height: calc(100vh - 58px);
    }
    .stage, .side {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .stage {
      min-height: 500px;
      display: grid;
      place-items: center;
      overflow: auto;
      padding: 12px;
    }
    .canvas-wrap {
      position: relative;
      max-width: 100%;
    }
    canvas {
      display: block;
      max-width: 100%;
      max-height: calc(100vh - 110px);
      cursor: crosshair;
      background: #111827;
    }
    .side {
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .label {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .query {
      font-size: 22px;
      line-height: 1.35;
      font-weight: 700;
    }
    .meta, .bbox, .path {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      line-height: 1.45;
      color: #344054;
      overflow-wrap: anywhere;
    }
    textarea {
      width: 100%;
      min-height: 72px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 9px 10px;
      font-weight: 650;
      cursor: pointer;
    }
    button.primary {
      background: var(--blue);
      border-color: var(--blue);
      color: white;
    }
    button.danger {
      color: var(--danger);
    }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .status {
      border-radius: 6px;
      background: #eef4ff;
      color: #1849a9;
      padding: 8px;
      font-size: 13px;
    }
    .status.warn {
      background: #fff1f3;
      color: var(--danger);
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .side { order: -1; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Local Region BBox Annotator</h1>
    <div id="progress" class="meta">Loading...</div>
  </header>
  <main>
    <section class="stage">
      <div class="canvas-wrap">
        <canvas id="canvas"></canvas>
      </div>
    </section>
    <aside class="side">
      <div>
        <div class="label">Query</div>
        <div id="query" class="query"></div>
      </div>
      <div>
        <div class="label">Target region</div>
        <div id="region" class="meta"></div>
      </div>
      <div>
        <div class="label">BBox [x1, y1, x2, y2]</div>
        <div id="bbox" class="bbox">null</div>
      </div>
      <div>
        <div class="label">Image</div>
        <div id="path" class="path"></div>
      </div>
      <div id="audit-panel" hidden>
        <div class="label">Audit instruction</div>
        <div id="audit-instruction" class="status"></div>
      </div>
      <div>
        <div class="label">Notes</div>
        <textarea id="notes" placeholder="Optional notes"></textarea>
      </div>
      <div id="status" class="status">Drag a box on the image.</div>
      <button id="save" class="primary">Save labeled bbox</button>
      <div class="row">
        <button id="prev">Previous</button>
        <button id="next">Next</button>
      </div>
      <div class="row">
        <button id="next-unlabeled">Next unlabeled</button>
        <button id="skip">Skip</button>
      </div>
      <button id="clear" class="danger">Clear bbox</button>
    </aside>
  </main>
  <script>
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const img = new Image();
    const state = {
      index: 0,
      count: 0,
      record: null,
      bbox: null,
      dragStart: null,
      display: {x: 0, y: 0, width: 0, height: 0, scale: 1},
    };

    async function api(path, options = {}) {
      const response = await fetch(path, options);
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || response.statusText);
      }
      return response.json();
    }

    function setStatus(text, warn = false) {
      const node = document.getElementById("status");
      node.textContent = text;
      node.className = warn ? "status warn" : "status";
    }

    function bboxText(bbox) {
      return bbox ? `[${bbox.map((v) => Math.round(v)).join(", ")}]` : "null";
    }

    function redraw() {
      const parentWidth = canvas.parentElement.clientWidth || img.naturalWidth || 900;
      const maxW = Math.min(parentWidth, img.naturalWidth || parentWidth);
      const maxH = window.innerHeight - 110;
      const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1);
      canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
      state.display = {x: 0, y: 0, width: canvas.width, height: canvas.height, scale};
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      drawBox(state.bbox, "#2563eb");
    }

    function drawBox(bbox, color) {
      if (!bbox) return;
      const scale = state.display.scale;
      const x = bbox[0] * scale;
      const y = bbox[1] * scale;
      const w = (bbox[2] - bbox[0]) * scale;
      const h = (bbox[3] - bbox[1]) * scale;
      ctx.lineWidth = 3;
      ctx.strokeStyle = color;
      ctx.fillStyle = "rgba(37, 99, 235, 0.16)";
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
    }

    function eventPoint(event) {
      const rect = canvas.getBoundingClientRect();
      const x = Math.max(0, Math.min(canvas.width, event.clientX - rect.left));
      const y = Math.max(0, Math.min(canvas.height, event.clientY - rect.top));
      return [x, y];
    }

    function displayToImageBox(start, end) {
      const scale = state.display.scale;
      const x1 = Math.min(start[0], end[0]) / scale;
      const y1 = Math.min(start[1], end[1]) / scale;
      const x2 = Math.max(start[0], end[0]) / scale;
      const y2 = Math.max(start[1], end[1]) / scale;
      return [
        Math.round(Math.max(0, Math.min(img.naturalWidth, x1))),
        Math.round(Math.max(0, Math.min(img.naturalHeight, y1))),
        Math.round(Math.max(0, Math.min(img.naturalWidth, x2))),
        Math.round(Math.max(0, Math.min(img.naturalHeight, y2))),
      ];
    }

    async function loadRecord(index) {
      const data = await api(`/api/record?index=${index}`);
      state.index = data.index;
      state.count = data.count;
      state.record = data.record;
      state.bbox = data.record.target_bbox || null;
      document.getElementById("query").textContent = data.record.query_text || "";
      const category = data.record.category_name ? ` | ${data.record.category_name}` : "";
      const item = data.record.source_item_key ? ` | ${data.record.source_item_key}` : "";
      document.getElementById("region").textContent = `${data.record.target_region || "unknown"}${category}${item}`;
      document.getElementById("path").textContent = data.record.image || "";
      document.getElementById("notes").value = data.record.notes || "";
      document.getElementById("bbox").textContent = bboxText(state.bbox);
      const auditPanel = document.getElementById("audit-panel");
      const auditInstruction = data.record.audit_instruction || "";
      auditPanel.hidden = !auditInstruction;
      document.getElementById("audit-instruction").textContent = auditInstruction;
      document.getElementById("progress").textContent =
        `${data.index + 1} / ${data.count} | labeled ${data.labeled_count} | skipped ${data.skipped_count}`;
      document.getElementById("prev").disabled = data.index <= 0;
      document.getElementById("next").disabled = data.index >= data.count - 1;
      img.onload = () => {
        redraw();
        if (auditInstruction) {
          setStatus("Review the existing bbox, adjust it, or skip this record.");
        } else {
          setStatus(data.record.label_status === "labeled" ? "Loaded existing bbox." : "Drag a box on the image.");
        }
      };
      img.src = `/api/image?index=${data.index}&t=${Date.now()}`;
    }

    async function saveRecord(labelStatus) {
      const notes = document.getElementById("notes").value;
      const payload = {
        index: state.index,
        target_bbox: labelStatus === "labeled" ? state.bbox : null,
        label_status: labelStatus,
        notes,
      };
      const result = await api("/api/save", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      setStatus(result.message);
      await loadRecord(Math.min(result.next_index, state.count - 1));
    }

    async function nextUnlabeled() {
      const data = await api(`/api/next_unlabeled?start=${state.index + 1}`);
      await loadRecord(data.index);
    }

    canvas.addEventListener("mousedown", (event) => {
      state.dragStart = eventPoint(event);
    });
    canvas.addEventListener("mousemove", (event) => {
      if (!state.dragStart) return;
      redraw();
      const tempBox = displayToImageBox(state.dragStart, eventPoint(event));
      drawBox(tempBox, "#f97316");
    });
    canvas.addEventListener("mouseup", (event) => {
      if (!state.dragStart) return;
      const box = displayToImageBox(state.dragStart, eventPoint(event));
      state.dragStart = null;
      if (box[2] <= box[0] || box[3] <= box[1]) {
        setStatus("Box is empty. Drag from one corner to the opposite corner.", true);
        redraw();
        return;
      }
      state.bbox = box;
      document.getElementById("bbox").textContent = bboxText(state.bbox);
      redraw();
      setStatus("BBox ready. Click Save labeled bbox.");
    });
    window.addEventListener("resize", () => {
      if (img.complete && img.naturalWidth) redraw();
    });
    document.getElementById("save").onclick = () => {
      if (!state.bbox) {
        setStatus("Draw a bbox before saving.", true);
        return;
      }
      saveRecord("labeled").catch((err) => setStatus(err.message, true));
    };
    document.getElementById("skip").onclick = () => saveRecord("skip").catch((err) => setStatus(err.message, true));
    document.getElementById("clear").onclick = () => saveRecord("unlabeled").catch((err) => setStatus(err.message, true));
    document.getElementById("prev").onclick = () => loadRecord(Math.max(0, state.index - 1)).catch((err) => setStatus(err.message, true));
    document.getElementById("next").onclick = () => loadRecord(Math.min(state.count - 1, state.index + 1)).catch((err) => setStatus(err.message, true));
    document.getElementById("next-unlabeled").onclick = () => nextUnlabeled().catch((err) => setStatus(err.message, true));
    loadRecord(0).catch((err) => setStatus(err.message, true));
  </script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a browser-based bbox annotator for 3.1.2 manual eval JSONL."
    )
    parser.add_argument("--manifest", required=True, help="Input manifest JSONL.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output labeled JSONL. Defaults to <manifest>_labeled.jsonl.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def default_output_path(manifest_path: str | Path) -> Path:
    """Return the default labeled JSONL path for a manifest."""
    path = Path(manifest_path)
    suffix = "".join(path.suffixes)
    if suffix:
        stem = path.name[: -len(suffix)]
        return path.with_name(f"{stem}_labeled{suffix}")
    return path.with_name(f"{path.name}_labeled.jsonl")


def load_annotation_records(
    manifest_path: str | Path,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load existing labeled output when present, otherwise load the manifest."""
    manifest = Path(manifest_path)
    output = Path(output_path) if output_path is not None else default_output_path(manifest)
    source = output if output.exists() else manifest
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    if not records:
        raise ValueError(f"No annotation records found in {source}")
    return records


def write_annotation_records(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Atomically write annotation records as JSONL."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output.parent,
        delete=False,
        prefix=f".{output.name}.",
        suffix=".tmp",
    ) as tmp:
        tmp_path = Path(tmp.name)
        for record in records:
            tmp.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(output)


def update_record(
    records: list[dict[str, Any]],
    index: int,
    *,
    target_bbox: list[float] | None,
    label_status: str,
    notes: str | None = None,
) -> None:
    """Update one annotation record in memory."""
    if index < 0 or index >= len(records):
        raise IndexError(f"index out of range: {index}")
    if label_status not in {"labeled", "unlabeled", "skip"}:
        raise ValueError(f"unsupported label_status: {label_status}")
    bbox = validate_bbox(target_bbox) if label_status == "labeled" else None
    record = records[index]
    record["target_bbox"] = bbox
    record["label_status"] = label_status
    if notes is not None:
        record["notes"] = notes


def validate_bbox(value: Any) -> list[int]:
    """Validate and normalize one xyxy bbox."""
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError("target_bbox must be [x1, y1, x2, y2]")
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("target_bbox must contain numbers") from exc
    if x2 <= x1 or y2 <= y1:
        raise ValueError("target_bbox must satisfy x2 > x1 and y2 > y1")
    return [x1, y1, x2, y2]


def count_status(records: list[dict[str, Any]], status: str) -> int:
    return sum(record.get("label_status") == status for record in records)


def next_unlabeled_index(records: list[dict[str, Any]], start: int = 0) -> int:
    """Return the next record whose label is not done, wrapping around."""
    if not records:
        raise ValueError("No records loaded")
    total = len(records)
    for offset in range(total):
        index = (start + offset) % total
        if records[index].get("label_status") not in {"labeled", "skip"}:
            return index
    return max(0, min(start, total - 1))


class AnnotationServer(BaseHTTPRequestHandler):
    records: list[dict[str, Any]] = []
    output_path: Path
    manifest_path: Path

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_bytes(APP_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/state":
                self._send_json(self._state_payload())
            elif parsed.path == "/api/record":
                index = self._query_index(parsed.query)
                self._send_json(self._record_payload(index))
            elif parsed.path == "/api/image":
                index = self._query_index(parsed.query)
                self._send_image(index)
            elif parsed.path == "/api/next_unlabeled":
                start = self._query_index(parsed.query, key="start", default=0)
                self._send_json({"index": next_unlabeled_index(self.records, start)})
            else:
                self.send_error(404, "Not found")
        except Exception as exc:  # pragma: no cover - exercised by browser use
            self.send_error(400, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path != "/api/save":
                self.send_error(404, "Not found")
                return
            payload = self._read_json_body()
            index = int(payload["index"])
            update_record(
                self.records,
                index,
                target_bbox=payload.get("target_bbox"),
                label_status=str(payload.get("label_status", "")),
                notes=payload.get("notes"),
            )
            write_annotation_records(self.records, self.output_path)
            self._send_json(
                {
                    "message": f"Saved to {self.output_path}",
                    "next_index": next_unlabeled_index(self.records, index + 1),
                    **self._state_payload(),
                }
            )
        except Exception as exc:  # pragma: no cover - exercised by browser use
            self.send_error(400, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _state_payload(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path),
            "output": str(self.output_path),
            "num_records": len(self.records),
            "labeled_count": count_status(self.records, "labeled"),
            "skipped_count": count_status(self.records, "skip"),
            "unlabeled_count": len(self.records)
            - count_status(self.records, "labeled")
            - count_status(self.records, "skip"),
        }

    def _record_payload(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.records):
            raise IndexError(f"index out of range: {index}")
        return {
            "index": index,
            "count": len(self.records),
            "record": self.records[index],
            "labeled_count": count_status(self.records, "labeled"),
            "skipped_count": count_status(self.records, "skip"),
        }

    def _query_index(
        self,
        query: str,
        *,
        key: str = "index",
        default: int | None = None,
    ) -> int:
        values = parse_qs(query).get(key)
        if not values:
            if default is None:
                raise ValueError(f"Missing query parameter: {key}")
            return default
        return int(values[0])

    def _send_image(self, index: int) -> None:
        record = self.records[index]
        image_path = Path(record["image"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        self._send_bytes(image_path.read_bytes(), content_type)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        data = self.rfile.read(length)
        return json.loads(data.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any]) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(
    manifest_path: str | Path,
    output_path: str | Path | None,
    *,
    host: str,
    port: int,
) -> None:
    manifest = Path(manifest_path)
    output = Path(output_path) if output_path is not None else default_output_path(manifest)
    records = load_annotation_records(manifest, output)
    AnnotationServer.records = records
    AnnotationServer.manifest_path = manifest
    AnnotationServer.output_path = output
    write_annotation_records(records, output)

    server = ThreadingHTTPServer((host, port), AnnotationServer)
    url = f"http://{host}:{port}"
    print(f"Annotator running at {url}")
    print(f"Manifest: {manifest}")
    print(f"Saving labels to: {output}")
    print("Open the URL in a browser, drag a box, then click Save labeled bbox.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped annotator.")


def main() -> None:
    args = parse_args()
    run_server(args.manifest, args.output, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
