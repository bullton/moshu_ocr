#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import cv2
import jieba
from paddleocr import PaddleOCR

jieba.setLogLevel(logging.WARNING)
# Speed up startup by skipping online hoster check when models are already cached.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _normalize_result_item(item: Any) -> Dict[str, Any]:
    """Normalize a PaddleOCR result item into a dict-like payload."""
    if isinstance(item, dict):
        return item

    # PaddleOCR result objects may expose json/json() in different forms.
    for attr in ("json", "to_dict"):
        if hasattr(item, attr):
            value = getattr(item, attr)
            try:
                value = value() if callable(value) else value
            except Exception:
                continue

            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    continue
            if isinstance(value, dict):
                return value

    return {}


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        try:
            out = value.tolist()
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))


def _split_text_to_words(text: str, segment_cn: bool) -> List[str]:
    # First split by common separators (space/punctuation/newline).
    chunks = [x.strip() for x in re.split(r"[\s,，。；;、|]+", text) if x.strip()]
    if not segment_cn:
        return chunks

    words: List[str] = []
    for chunk in chunks:
        if _contains_cjk(chunk):
            segs = [x.strip() for x in jieba.lcut(chunk) if x.strip()]
            words.extend(segs if segs else [chunk])
        else:
            words.append(chunk)
    return words


def _poly_to_xyh(poly: Any) -> Tuple[float, float, float]:
    pts = _to_list(poly)
    if not pts:
        return (0.0, 0.0, 0.0)
    xs = [float(p[0]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
    ys = [float(p[1]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not xs or not ys:
        return (0.0, 0.0, 0.0)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return (x_min, y_min, max(1.0, y_max - y_min))


def _sort_by_reading_order(rows: List[Tuple[str, float, float, float]]) -> List[str]:
    if not rows:
        return []
    hs = [h for _, _, _, h in rows if h > 0]
    line_tol = (median(hs) * 0.6) if hs else 20.0

    rows_sorted = sorted(rows, key=lambda r: (r[2], r[1]))  # by y then x
    lines: List[List[Tuple[str, float, float, float]]] = []
    line_ys: List[float] = []

    for row in rows_sorted:
        _, _, y, _ = row
        placed = False
        for idx, line_y in enumerate(line_ys):
            if abs(y - line_y) <= line_tol:
                lines[idx].append(row)
                new_avg = sum(r[2] for r in lines[idx]) / len(lines[idx])
                line_ys[idx] = new_avg
                placed = True
                break
        if not placed:
            lines.append([row])
            line_ys.append(y)

    ordered_words: List[str] = []
    for line in lines:
        for word, _, _, _ in sorted(line, key=lambda r: r[1]):  # left to right
            ordered_words.append(word)
    return ordered_words


def _load_variants(image_path: str, multi_pass: bool) -> List[Any]:
    if not multi_pass:
        return [image_path]

    img = cv2.imread(image_path)
    if img is None:
        return [image_path]

    variants: List[Any] = [img]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    up2 = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    variants.append(up2)
    variants.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
    return variants


def extract_words(
    image_path: str,
    lang: str = "ch",
    min_score: float = 0.0,
    segment_cn: bool = True,
    multi_pass: bool = True,
) -> List[str]:
    """Run PaddleOCR v5 pipeline and return recognized words as a list."""
    ocr = PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    words: List[str] = []
    vote_count: Dict[str, int] = {}
    first_seen: Dict[str, int] = {}
    base_words: List[str] = []
    word_pos: Dict[str, Tuple[float, float, float]] = {}
    for pass_idx, image_input in enumerate(_load_variants(image_path, multi_pass=multi_pass)):
        results = ocr.predict(image_input)
        pass_words: List[str] = []
        pass_positions: Dict[str, Tuple[float, float, float]] = {}

        for item in results:
            raw = _normalize_result_item(item)
            payload = raw.get("res", raw)

            rec_texts = _to_list(payload.get("rec_texts"))
            rec_scores = _to_list(payload.get("rec_scores"))
            dt_polys = _to_list(payload.get("dt_polys"))

            if not rec_texts:
                continue

            paired: List[Tuple[str, float]] = []
            for idx, text in enumerate(rec_texts):
                score = float(rec_scores[idx]) if idx < len(rec_scores) else 1.0
                paired.append((str(text).strip(), score))

            for idx, (text, score) in enumerate(paired):
                if text and score >= min_score:
                    split_words = _split_text_to_words(text, segment_cn=segment_cn)
                    pass_words.extend(split_words)
                    x, y, h = _poly_to_xyh(dt_polys[idx] if idx < len(dt_polys) else None)
                    for offset, sw in enumerate(split_words):
                        if sw not in pass_positions:
                            pass_positions[sw] = (x + 0.01 * offset, y, h)

        # Count each token at most once per pass.
        unique_pass_words = list(dict.fromkeys(pass_words))
        for word in unique_pass_words:
            vote_count[word] = vote_count.get(word, 0) + 1
            if word not in first_seen:
                first_seen[word] = pass_idx
            if word not in word_pos and word in pass_positions:
                word_pos[word] = pass_positions[word]
        if pass_idx == 0:
            base_words = unique_pass_words

    base_set = set(base_words)
    selected = set(base_words)
    single_char_added: List[str] = []

    for word in sorted(vote_count.keys(), key=lambda w: (first_seen[w], -vote_count[w], w)):
        if word in base_set:
            continue

        # Keep a missing single Chinese char only when it is not already part of base words.
        if len(word) == 1 and _contains_cjk(word):
            if any(word in bw for bw in base_words):
                continue
            selected.add(word)
            single_char_added.append(word)
            continue

        # For non-single-char additions, require >=2-pass agreement.
        if vote_count[word] >= 2:
            selected.add(word)

    rows: List[Tuple[str, float, float, float]] = []
    base_rows: List[Tuple[str, float, float, float]] = []
    for word in base_words:
        x, y, h = word_pos.get(word, (1e9, 1e9, 0.0))
        base_rows.append((word, x, y, h))

    # Heuristic for missing single-char recovery: place it before nearest base word on same line.
    for word in single_char_added:
        if word not in word_pos or not base_rows:
            continue
        x, y, h = word_pos[word]
        nearest = min(base_rows, key=lambda r: (abs(r[2] - y), abs(r[1] - x)))
        nx, ny, nh = nearest[1], nearest[2], nearest[3]
        line_tol = max(8.0, (nh if nh > 0 else h) * 0.8)
        if abs(y - ny) <= line_tol:
            word_pos[word] = (nx - 0.1, ny, h if h > 0 else nh)

    for word in selected:
        x, y, h = word_pos.get(word, (1e9, 1e9, 0.0))
        rows.append((word, x, y, h))
    words = _sort_by_reading_order(rows)

    # Final deterministic reorder: place recovered single-char token before nearest base token.
    for word in single_char_added:
        if word not in words or word not in word_pos or not base_rows:
            continue
        nearest_base = min(
            base_rows, key=lambda r: (abs(r[2] - word_pos[word][1]), abs(r[1] - word_pos[word][0]))
        )[0]
        if nearest_base not in words:
            continue
        i_word = words.index(word)
        i_base = words.index(nearest_base)
        if i_word > i_base:
            words.pop(i_word)
            words.insert(i_base, word)
    return words


def main() -> None:
    parser = argparse.ArgumentParser(
        description="识别图片中的手写词语，并输出 JSON 数组（PaddleOCR v5）。"
    )
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--lang", default="ch", help="OCR 语言，默认 ch")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="最低置信度阈值（0~1），低于该值的词语会被过滤",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="是否格式化输出 JSON",
    )
    parser.add_argument(
        "--segment-cn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否进行中文分词（默认开启，可用 --no-segment-cn 关闭）",
    )
    parser.add_argument(
        "--multi-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用多路预处理识别（默认开启，可用 --no-multi-pass 关闭）",
    )

    args = parser.parse_args()

    words = extract_words(
        args.image,
        lang=args.lang,
        min_score=args.min_score,
        segment_cn=args.segment_cn,
        multi_pass=args.multi_pass,
    )
    if args.pretty:
        print(json.dumps(words, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(words, ensure_ascii=False))


if __name__ == "__main__":
    main()
