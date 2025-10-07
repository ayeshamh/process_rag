import os
import glob
import json
import argparse
from typing import Any, Dict


def find_latest_dump(log_dir: str) -> str | None:
    pattern = os.path.join(log_dir, "prompt_inputs_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def main():
    parser = argparse.ArgumentParser(description="Print the latest prompt inputs (context, entity_context, graph_neighbors)")
    parser.add_argument("--log_dir", default="logs", help="Directory where prompt input dumps are stored")
    parser.add_argument("--file", default=None, help="Explicit path to a prompt_inputs_*.json file to print")
    parser.add_argument("--keys", nargs="*", default=["context", "entity_context", "graph_neighbors", "chunk_context"],
                        help="Which keys to print from the dump (default: context, entity_context, graph_neighbors, chunk_context)")
    args = parser.parse_args()

    dump_path = args.file or find_latest_dump(args.log_dir)
    if not dump_path or not os.path.exists(dump_path):
        print("No prompt input dump found. Run a query first to generate logs.")
        print("Tip: run your app or the demo to trigger a QA call, then rerun this script.")
        return

    with open(dump_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # Only keep requested keys
    filtered = {k: data.get(k) for k in args.keys}
    print(json.dumps({"file": dump_path, **filtered}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


