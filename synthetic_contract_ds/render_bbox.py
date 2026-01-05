import json
import random
from pathlib import Path
from typing import Dict, Any
import time
import traceback
from PIL import Image, ImageEnhance, ImageFilter
from playwright.sync_api import sync_playwright

def apply_noise(img_path: Path, profile: dict, rng: random.Random):
    img = Image.open(img_path).convert("RGB")

    # blur
    b0, b1 = profile["blur"]
    blur_r = rng.uniform(b0, b1)
    if blur_r > 0.01:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_r))

    # contrast / brightness
    c0, c1 = profile["contrast"]
    br0, br1 = profile["brightness"]
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(c0, c1))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(br0, br1))

    # occlusion (遮挡不改变几何，不影响 bbox 对齐)
    o0, o1 = profile["occlusion"]
    n_occ = rng.randint(o0, o1)
    if n_occ > 0:
        w, h = img.size
        for _ in range(n_occ):
            rw = rng.randint(int(0.06 * w), int(0.18 * w))
            rh = rng.randint(int(0.03 * h), int(0.10 * h))
            x = rng.randint(0, max(0, w - rw))
            y = rng.randint(0, max(0, h - rh))
            patch = Image.new("RGB", (rw, rh), (rng.randint(180, 245),) * 3)
            img.paste(patch, (x, y))

    # JPEG recompress
    q0, q1 = profile["jpeg_q"]
    q = int(rng.uniform(q0, q1))
    tmp = img_path.with_suffix(".tmp.jpg")
    img.save(tmp, format="JPEG", quality=q)
    img = Image.open(tmp).convert("RGB")
    tmp.unlink(missing_ok=True)

    img.save(img_path)

def extract_nodes_from_dom(page, scale_factor: float) -> Dict[str, Any]:
    js = """
    () => {
      const selectors = [
        "[data-element-id]",
        "[data-mention-id]",
        "[data-value-id]",
        "[data-object-id]",
        "[data-ref-id]"
      ];
      const els = [];
      selectors.forEach(s => document.querySelectorAll(s).forEach(e => els.push(e)));

      function rectOf(el){
        const r = el.getBoundingClientRect();
        const x0 = r.left + window.scrollX;
        const y0 = r.top + window.scrollY;
        const x1 = r.right + window.scrollX;
        const y1 = r.bottom + window.scrollY;
        return [x0, y0, x1, y1];
      }

      function closestElementId(el){
        const p = el.closest("[data-element-id]");
        return p ? p.getAttribute("data-element-id") : null;
      }

      const out = [];
      for (const el of els){
        const bbox = rectOf(el);
        const txt = (el.innerText || el.textContent || "").trim();
        const d = el.dataset;

        let node_id = null;
        let kind = null;

        if (d.elementId){ node_id = d.elementId; kind = "block"; }
        if (d.mentionId){ node_id = d.mentionId; kind = "mention"; }
        if (d.valueId){ node_id = d.valueId; kind = "value"; }
        if (d.objectId){ node_id = d.objectId; kind = "object"; }
        if (d.refId){ node_id = d.refId; kind = "ref"; }

        out.push({
          node_id, kind,
          element_id: d.elementId || null,
          mention_id: d.mentionId || null,
          entity_id: d.entityId || null,
          value_id: d.valueId || null,
          norm_type: d.normType || null,
          norm_value: d.normValue || null,
          object_id: d.objectId || null,
          rel: d.rel || null,
          target_obj: d.targetObj || null,
          parent_element_id: closestElementId(el),
          text: txt,
          bbox
        });
      }

      return {
        devicePixelRatio: window.devicePixelRatio,
        scrollWidth: document.documentElement.scrollWidth,
        scrollHeight: document.documentElement.scrollHeight,
        nodes: out
      };
    }
    """
    res = page.evaluate(js)

    # bbox: CSS px -> screenshot px
    for n in res["nodes"]:
        x0, y0, x1, y1 = n["bbox"]
        n["bbox"] = [x0 * scale_factor, y0 * scale_factor, x1 * scale_factor, y1 * scale_factor]
    return res

def build_nodes_json(dom_res: Dict[str, Any], page_w_px: int, page_h_px: int) -> Dict[str, Any]:
    nodes = []
    for n in dom_res["nodes"]:
        x0, y0, x1, y1 = n["bbox"]
        page_idx = int(y0 // page_h_px) + 1
        nn = dict(n)
        nn["page_idx"] = page_idx
        nodes.append(nn)

    return {
      "page": { "width": page_w_px, "height": page_h_px },
      "document": { "scroll_width": dom_res["scrollWidth"], "scroll_height": dom_res["scrollHeight"] },
      "nodes": nodes
    }

def main():
    base = Path(".")
    cfg = json.loads((base / "config_v1_audit200.json").read_text(encoding="utf-8"))
    ds_dir = base / cfg["dataset_name"]

    scale_factor = cfg["noise"]["render_scale_factor"]
    page_w = cfg["page"]["width"]
    page_h = cfg["page"]["height"]
    profiles = cfg["noise"]["profiles"]

    page_w_px = int(page_w * scale_factor)
    page_h_px = int(page_h * scale_factor)

    doc_dirs = sorted(ds_dir.glob("doc_*"))
    total = len(doc_dirs)
    print(f"[INFO] Total docs to render: {total}", flush=True)

    errors = []
    t_all0 = time.time()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": page_w, "height": page_h},
            device_scale_factor=scale_factor
        )
        page = context.new_page()

        # 防止无限等待（可选但推荐）
        page.set_default_timeout(15000)
        page.set_default_navigation_timeout(15000)

        for idx, doc_dir in enumerate(doc_dirs, 1):
            t0 = time.time()
            html_path = (doc_dir / "doc.html").resolve()

            try:
                meta = json.loads((doc_dir / "meta.json").read_text(encoding="utf-8"))
                print(f"[{idx:>4}/{total}] start {doc_dir.name} | noise={meta.get('noise_level')} pages={meta.get('page_count')}", flush=True)

                # 更稳：不要用 networkidle，避免卡住
                page.goto(html_path.as_uri(), wait_until="load", timeout=15000)
                page.wait_for_timeout(50)  # 给布局一点时间

                png_path = doc_dir / "rendered.png"
                page.screenshot(path=str(png_path), full_page=True)

                dom_res = extract_nodes_from_dom(page, scale_factor=scale_factor)
                nodes = build_nodes_json(dom_res, page_w_px=page_w_px, page_h_px=page_h_px)
                (doc_dir / "nodes.json").write_text(json.dumps(nodes, ensure_ascii=False, indent=2), encoding="utf-8")

                # Apply photometric noise AFTER bbox export (几何不变，bbox 仍对齐)
                noise_level = meta["noise_level"]
                rng = random.Random(meta["seed"] + 99991)
                apply_noise(png_path, profiles[noise_level], rng)

                dt = time.time() - t0
                print(f"[{idx:>4}/{total}] done  {doc_dir.name} | {dt:.2f}s", flush=True)

            except Exception as e:
                dt = time.time() - t0
                err_msg = f"{type(e).__name__}: {e}"
                print(f"[{idx:>4}/{total}] ERROR {doc_dir.name} | {dt:.2f}s | {err_msg}", flush=True)
                errors.append({"doc": doc_dir.name, "error": err_msg})
                continue

        context.close()
        browser.close()

    # 输出错误清单
    if errors:
        (ds_dir / "render_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[WARN] {len(errors)} docs failed. See: {ds_dir / 'render_errors.json'}", flush=True)

    print(f"[OK] rendered.png + nodes.json generated for: {ds_dir} | total_time={time.time()-t_all0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
