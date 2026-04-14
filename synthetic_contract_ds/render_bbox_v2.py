import json
import argparse
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageEnhance, ImageFilter
from playwright.sync_api import sync_playwright


def overlay_paper_texture(img: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    if strength <= 1e-6:
        return img
    width, height = img.size
    noise = Image.effect_noise((width, height), rng.uniform(6, 18)).convert("L")
    noise = noise.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.6)))
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    return Image.blend(img, noise_rgb, alpha=float(strength))


def overlay_scanlines(img: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    if strength <= 1e-6:
        return img
    width, height = img.size
    lines = Image.new("L", (width, height), 255)
    step = rng.randint(6, 14)
    for y_pos in range(0, height, step):
        value = rng.randint(220, 245)
        Image.Image.paste(lines, Image.new("L", (width, 1), value), (0, y_pos))
    lines = lines.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
    lines_rgb = Image.merge("RGB", (lines, lines, lines))
    return Image.blend(img, lines_rgb, alpha=float(strength))


def apply_noise(img_path: Path, profile: dict, rng: random.Random) -> None:
    if not img_path.exists():
        return

    img = Image.open(img_path).convert("RGB")

    blur_min, blur_max = profile["blur"]
    blur_radius = rng.uniform(blur_min, blur_max)
    if blur_radius > 0.01:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    contrast_min, contrast_max = profile["contrast"]
    brightness_min, brightness_max = profile["brightness"]
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(contrast_min, contrast_max))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(brightness_min, brightness_max))

    paper_min, paper_max = profile.get("paper", [0.0, 0.0])
    scanline_min, scanline_max = profile.get("scanline", [0.0, 0.0])
    img = overlay_paper_texture(img, rng.uniform(paper_min, paper_max), rng)
    img = overlay_scanlines(img, rng.uniform(scanline_min, scanline_max), rng)

    occ_min, occ_max = profile["occlusion"]
    num_occ = rng.randint(occ_min, occ_max)
    if num_occ > 0:
        width, height = img.size
        for _ in range(num_occ):
            rect_w = rng.randint(int(0.06 * width), int(0.18 * width))
            rect_h = rng.randint(int(0.03 * height), int(0.10 * height))
            x_pos = rng.randint(0, max(0, width - rect_w))
            y_pos = rng.randint(0, max(0, height - rect_h))
            patch = Image.new("RGB", (rect_w, rect_h), (rng.randint(180, 245),) * 3)
            img.paste(patch, (x_pos, y_pos))

    quality_min, quality_max = profile["jpeg_q"]
    quality = int(rng.uniform(quality_min, quality_max))
    img.save(img_path, format="JPEG", quality=quality)


def extract_nodes_single_page(page) -> List[Dict[str, Any]]:
    js = r"""
    () => {
      const pageDiv = document.querySelector(".page");
      if (!pageDiv) {
        return [];
      }

      const selectors = [
        "[data-element-id]",
        "[data-mention-id]",
        "[data-value-id]",
        "[data-object-id]",
        "[data-ref-id]"
      ];

      const pageRect = pageDiv.getBoundingClientRect();
      const nodes = [];
      const seenNodeId = new Set();

      for (const selector of selectors) {
        const elements = Array.from(pageDiv.querySelectorAll(selector));
        for (const el of elements) {
          const d = el.dataset;

          let nodeId = d.elementId || "";
          let kind = "block";
          if (d.mentionId) { nodeId = d.mentionId; kind = "mention"; }
          if (d.valueId) { nodeId = d.valueId; kind = "value"; }
          if (d.objectId) { nodeId = d.objectId; kind = "object"; }
          if (d.refId) { nodeId = d.refId; kind = "ref"; }

          if (!nodeId || seenNodeId.has(nodeId)) {
            continue;
          }
          seenNodeId.add(nodeId);

          const rect = el.getBoundingClientRect();
          let x0 = Math.max(0, rect.left - pageRect.left);
          let y0 = Math.max(0, rect.top - pageRect.top);
          let x1 = Math.min(pageRect.width, rect.right - pageRect.left);
          let y1 = Math.min(pageRect.height, rect.bottom - pageRect.top);

          if (x1 <= x0 || y1 <= y0) {
            continue;
          }

          let parentElementId = null;
          const closestEl = el.closest("[data-element-id]");
          if (closestEl) {
            if (closestEl === el) {
              const parent = el.parentElement ? el.parentElement.closest("[data-element-id]") : null;
              parentElementId = parent ? (parent.dataset.elementId || null) : null;
            } else {
              parentElementId = closestEl.dataset.elementId || null;
            }
          }

          nodes.push({
            node_id: nodeId,
            kind: kind,
            page_idx: 1,
            bbox: [x0, y0, x1, y1],
            text: (el.innerText || el.textContent || "").trim(),
            parent_element_id: parentElementId,
            element_id: d.elementId || null,
            mention_id: d.mentionId || null,
            value_id: d.valueId || null,
            object_id: d.objectId || null,
            ref_id: d.refId || null,
            rel: d.rel || null,
            target_obj: d.targetObj || null,
            entity_id: d.entityId || null,
            norm_type: d.normType || null,
            norm_value: d.normValue || null,
            tag: d.tag || null,
            page_width: pageRect.width,
            page_height: pageRect.height
          });
        }
      }

      return nodes;
    }
    """
    return page.evaluate(js)


def build_normalized_nodes(raw_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = []
    page_width = 1000
    page_height = 1000

    for node in raw_nodes:
        width = node.pop("page_width")
        height = node.pop("page_height")
        if width <= 0 or height <= 0:
            continue

        x0, y0, x1, y1 = node["bbox"]
        nx0 = max(0, min(1000, int((x0 / width) * 1000)))
        ny0 = max(0, min(1000, int((y0 / height) * 1000)))
        nx1 = max(0, min(1000, int((x1 / width) * 1000)))
        ny1 = max(0, min(1000, int((y1 / height) * 1000)))
        if nx1 <= nx0 or ny1 <= ny0:
            continue
        node["bbox"] = [nx0, ny0, nx1, ny1]
        normalized.append(node)

    return {
        "document": {"page_count": 1},
        "page": {"width": page_width, "height": page_height},
        "nodes": normalized,
    }


def validate_nodes(nodes_json: Dict[str, Any], labels: Dict[str, Any]) -> None:
    nodes = nodes_json["nodes"]
    node_ids = [node["node_id"] for node in nodes]
    duplicates = [node_id for node_id, count in Counter(node_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate node ids: {duplicates[:5]}")

    for node in nodes:
        x0, y0, x1, y1 = node["bbox"]
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError(f"Invalid bbox: {node}")
        if node["page_idx"] != 1:
            raise ValueError(f"Unexpected page index: {node}")

    id_set = set(node_ids)
    for entity in labels["coref"]["entities"]:
        for mention_id in entity["mentions"]:
            if mention_id not in id_set:
                raise ValueError(f"Missing coref mention in nodes: {mention_id}")

    for relation in labels["relations"]:
        if relation["h"] not in id_set or relation["t"] not in id_set:
            raise ValueError(f"Relation points to missing node: {relation}")
        trigger = relation.get("trigger")
        if trigger and trigger not in id_set:
            raise ValueError(f"Relation trigger missing in nodes: {relation}")

    for item in labels["normalization"]:
        if item["value_id"] not in id_set:
            raise ValueError(f"Normalization value missing in nodes: {item}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v2_audit200.json")
    args = parser.parse_args()

    base = Path(".")
    cfg = json.loads((base / args.config).read_text(encoding="utf-8"))
    ds_dir = base / cfg["dataset_name"]

    scale_factor = cfg["noise"]["render_scale_factor"]
    page_width = cfg["page"]["width"]
    page_height = cfg["page"]["height"]
    profiles = cfg["noise"]["profiles"]

    doc_dirs = sorted(ds_dir.glob("doc_*"))
    total = len(doc_dirs)
    print(f"[INFO] Total docs to render: {total}", flush=True)

    errors = []
    start_time = time.time()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport={"width": page_width, "height": page_height},
            device_scale_factor=scale_factor,
        )
        page = context.new_page()
        page.set_default_timeout(15000)

        for idx, doc_dir in enumerate(doc_dirs, 1):
            step_start = time.time()
            html_path = (doc_dir / "doc.html").resolve()
            try:
                meta = json.loads((doc_dir / "meta.json").read_text(encoding="utf-8"))
                labels = json.loads((doc_dir / "labels.json").read_text(encoding="utf-8"))

                page.goto(html_path.as_uri(), wait_until="load")
                page.wait_for_timeout(50)

                raw_nodes = extract_nodes_single_page(page)
                nodes_json = build_normalized_nodes(raw_nodes)
                validate_nodes(nodes_json, labels)
                (doc_dir / "nodes.json").write_text(
                    json.dumps(nodes_json, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                page_el = page.query_selector(".page")
                if page_el is None:
                    raise ValueError("Missing .page element")

                image_path = doc_dir / "image_p1.jpg"
                page_el.screenshot(path=str(image_path), type="jpeg")

                rng = random.Random(meta["seed"] + 99991)
                apply_noise(image_path, profiles[meta["noise_level"]], rng)

                elapsed = time.time() - step_start
                print(f"[{idx:>4}/{total}] done  {doc_dir.name} | {elapsed:.2f}s", flush=True)
            except Exception as exc:
                errors.append({"doc": doc_dir.name, "error": str(exc)})
                elapsed = time.time() - step_start
                print(f"[{idx:>4}/{total}] ERROR {doc_dir.name} | {elapsed:.2f}s | {exc}", flush=True)

        context.close()
        browser.close()

    if errors:
        (ds_dir / "render_errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        (ds_dir / "render_errors.json").unlink(missing_ok=True)

    print(f"[OK] Render complete. Time: {time.time() - start_time:.2f}s", flush=True)


if __name__ == "__main__":
    main()
