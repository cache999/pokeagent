import os
import json
from pathlib import Path
from typing import Iterable, Union, Optional


def save_llm_responses(
    log_source: Union[str, Iterable[str]],
    output_dir: Union[str, Path] = "parsed_llm_logs",
    output_filename: Optional[str] = None
) -> Path:
    """
    Parse newline-delimited JSON LLM logs, extract the `response` field for each entry,
    and save them to a file under `parsed_llm_logs/`.

    Args:
        log_source: File path to the log or an iterable of lines.
        output_dir: Directory where parsed responses will be saved.
        output_filename: Optional custom output filename. If not provided and
                         log_source is a path, uses its stem with '.txt'. Otherwise, 'responses.txt'.

    Returns:
        Path to the written output file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output file name
    if output_filename:
        out_path = output_dir / output_filename
    elif isinstance(log_source, str):
        stem = Path(log_source).stem
        out_path = output_dir / f"{stem}_responses.txt"
    else:
        out_path = output_dir / "responses.txt"

    def _iter_lines(src):
        if isinstance(src, str):
            with open(src, "r", encoding="utf-8") as f:
                for line in f:
                    yield line
        else:
            for line in src:
                yield line

    count = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for raw in _iter_lines(log_source):
            line = raw.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = entry.get("response")
            if resp is None:
                continue
            # Normalize non-string responses
            if not isinstance(resp, str):
                resp = json.dumps(resp, ensure_ascii=False)
            out_f.write(resp.rstrip() + "\n")
            count += 1

    # Optionally, create an index file with a summary
    index_path = output_dir / "index.txt"
    try:
        with open(index_path, "a", encoding="utf-8") as idx:
            idx.write(f"{out_path.name}\t{count} responses\n")
    except OSError:
        pass

    return out_path


# Example usage:
if __name__ == "__main__":
    save_llm_responses("llm_logs/llm_log_20250913_095536.jsonl")