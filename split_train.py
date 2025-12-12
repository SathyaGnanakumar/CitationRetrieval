import json
from pathlib import Path
from typing import Iterator, TextIO, Dict, Any


def iter_json_array(stream: TextIO) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a large JSON array without loading it fully."""
    decoder = json.JSONDecoder()
    buffer = ""

    while True:
        chunk = stream.read(1 << 16)  # 64 KiB per chunk
        if not chunk:
            break
        buffer += chunk

        while True:
            buffer = buffer.lstrip()
            if not buffer:
                break

            if buffer[0] == "[":
                buffer = buffer[1:]
                continue
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue
            if buffer[0] == "]":
                # End of array
                return

            try:
                item, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                # Need more data
                break

            yield item
            buffer = buffer[offset:]

    # Process any residual buffer after EOF
    buffer = buffer.lstrip()
    if buffer and buffer[0] == "]":
        return
    if buffer:
        item, _ = decoder.raw_decode(buffer)
        yield item


def split_dataset(
    source_path: Path,
    correct_path: Path,
    wrong_path: Path,
) -> None:
    with (
        source_path.open("r", encoding="utf-8") as src,
        correct_path.open("w", encoding="utf-8") as correct_f,
        wrong_path.open("w", encoding="utf-8") as wrong_f,
    ):

        for record in iter_json_array(src):
            base = {
                "example_id": record.get("example_id"),
                "paper_id": record.get("paper_id"),
                "citation_context": record.get("citation_context"),
            }

            candidates = record.get("candidates", [])
            gt_index = record.get("ground_truth_index")

            for idx, candidate in enumerate(candidates):
                output = {**base, "guess_index": idx, "guess": candidate}
                line = json.dumps(output, ensure_ascii=False)

                if idx == gt_index:
                    correct_f.write(line + "\n")
                else:
                    wrong_f.write(line + "\n")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "train.json"

    out_dir = project_root / "data_splits"
    out_dir.mkdir(exist_ok=True)

    correct_out = out_dir / "correct.jsonl"
    wrong_out = out_dir / "wrong.jsonl"

    split_dataset(source, correct_out, wrong_out)


if __name__ == "__main__":
    main()
