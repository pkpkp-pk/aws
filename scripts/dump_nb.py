"""Dump notebook cell outputs (stream/error/execute_result), ANSI-stripped.

Usage:
    python3 scripts/dump_nb.py test.ipynb          # all cells with outputs
    python3 scripts/dump_nb.py test.ipynb 3        # cell index 3 only
"""
import json
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
MAX_CHARS = 4000


def cell_output_text(cell: dict) -> str:
    parts = []
    for out in cell.get("outputs", []):
        t = out.get("output_type")
        if t == "stream":
            parts.append("".join(out.get("text", [])))
        elif t == "error":
            parts.append("\n".join(out.get("traceback", [])))
        elif t in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "text/plain" in data:
                parts.append("".join(data["text/plain"]))
    return ANSI.sub("", "\n".join(parts))


def main() -> None:
    path = sys.argv[1]
    only = int(sys.argv[2]) if len(sys.argv) > 2 else None
    nb = json.load(open(path))
    for i, cell in enumerate(nb["cells"]):
        if only is not None and i != only:
            continue
        text = cell_output_text(cell).strip()
        if not text:
            continue
        if len(text) > MAX_CHARS:
            text = text[:1500] + f"\n...[{len(text) - 3000} chars cut]...\n" + text[-1500:]
        print(f"===== cell {i} ({cell['cell_type']}) =====")
        print(text)
        print()


if __name__ == "__main__":
    main()
