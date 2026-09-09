from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_yaml(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml
    except ModuleNotFoundError:
        return _simple_yaml_load(text)

    return yaml.safe_load(text)


def _simple_yaml_load(text: str) -> Any:
    lines = text.splitlines()

    def strip_comment(value: str) -> str:
        in_single = False
        in_double = False
        for idx, char in enumerate(value):
            if char == "'" and not in_double:
                in_single = not in_single
            elif char == '"' and not in_single:
                in_double = not in_double
            elif char == "#" and not in_single and not in_double:
                return value[:idx].rstrip()
        return value.rstrip()

    cleaned: list[tuple[int, str]] = []
    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        if not raw.strip() or raw.lstrip().startswith("#"):
            idx += 1
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = strip_comment(raw.strip())
        if stripped.endswith(": |") or stripped.endswith(": >") or stripped in {"|", ">"}:
            key_part = stripped[:-2].rstrip()
            block_indent = None
            block: list[str] = []
            idx += 1
            while idx < len(lines):
                child = lines[idx]
                if not child.strip():
                    block.append("")
                    idx += 1
                    continue
                child_indent = len(child) - len(child.lstrip(" "))
                if child_indent <= indent:
                    break
                if block_indent is None:
                    block_indent = child_indent
                block.append(child[block_indent:])
                idx += 1
            scalar = "\\n".join(block) if stripped.endswith(": |") or stripped == "|" else " ".join(part.strip() for part in block)
            if key_part:
                cleaned.append((indent, f"{key_part}: {json.dumps(scalar)}"))
            else:
                cleaned.append((indent, json.dumps(scalar)))
            continue
        cleaned.append((indent, stripped))
        idx += 1

    def parse_scalar(value: str) -> Any:
        value = value.strip()
        if value == "":
            return ""
        if value in {"null", "Null", "NULL", "~"}:
            return None
        if value in {"true", "True", "TRUE"}:
            return True
        if value in {"false", "False", "FALSE"}:
            return False
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [parse_scalar(part.strip()) for part in inner.split(",")]
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value[1:-1]
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    def parse_block(pos: int, indent: int) -> tuple[Any, int]:
        if pos >= len(cleaned):
            return {}, pos
        is_list = cleaned[pos][0] == indent and cleaned[pos][1].startswith("- ")
        if is_list:
            out: list[Any] = []
            while pos < len(cleaned):
                line_indent, line = cleaned[pos]
                if line_indent != indent or not line.startswith("- "):
                    break
                item = line[2:].strip()
                pos += 1
                child = None
                if pos < len(cleaned) and cleaned[pos][0] > indent:
                    child, pos = parse_block(pos, cleaned[pos][0])
                if not item:
                    out.append(child)
                elif ": " in item or item.endswith(":"):
                    key, _, value = item.partition(":")
                    entry = {key.strip(): parse_scalar(value.strip()) if value.strip() else child}
                    if isinstance(child, dict) and value.strip():
                        entry.update(child)
                    out.append(entry)
                else:
                    out.append(parse_scalar(item))
            return out, pos

        out_dict: dict[str, Any] = {}
        while pos < len(cleaned):
            line_indent, line = cleaned[pos]
            if line_indent < indent:
                break
            if line_indent > indent:
                break
            if line.startswith("- "):
                break
            key, sep, value = line.partition(":")
            if not sep:
                return parse_scalar(line), pos + 1
            pos += 1
            if value.strip():
                out_dict[key.strip()] = parse_scalar(value.strip())
            elif pos < len(cleaned) and cleaned[pos][0] > indent:
                child, pos = parse_block(pos, cleaned[pos][0])
                out_dict[key.strip()] = child
            else:
                out_dict[key.strip()] = {}
        return out_dict, pos

    if not cleaned:
        return None
    data, _ = parse_block(0, cleaned[0][0])
    return data
