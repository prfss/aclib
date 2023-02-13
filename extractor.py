#!/usr/bin/env python3
import argparse
import sys
import re
import json
from pathlib import Path
import fnmatch
from collections import deque
from typing import Optional


def main() -> None:
    args = parse_args()
    args.func(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rustコードにテキスト処理を施し,標準出力へ出力します.")

    subparser = parser.add_subparsers(required=True)

    # ドキュメンテーションコメント除去
    strip_parser = subparser.add_parser("strip", help="ドキュメンテーションコメントとテストコードを除去します.")
    strip_parser.add_argument("file", help="入力となるRustコードのファイルパス.")
    strip_parser.set_defaults(func=strip_command)

    # スニペット生成
    snippet_parser = subparser.add_parser(
        "snippet", help="(stripを施した上で)VSCodeのスニペットを生成します."
    )
    snippet_parser.add_argument("directory", help="入力となるRustコードを含むディレクトリパス.")
    snippet_parser.add_argument(
        "-e",
        "--exclude",
        nargs="*",
        metavar="pattern",
        default=[],
        help="無視するファイルのグロブパターンを指定します.",
    )

    module_option = parser.add_mutually_exclusive_group()
    module_option.add_argument(
        "-m", "--module", metavar="module_name", help="モジュール名を指定します."
    )
    module_option.add_argument(
        "--automodule", action="store_true", help="ファイル名をモジュール名とします."
    )
    snippet_parser.set_defaults(func=snippet_command)

    return parser.parse_args()


def strip_command(args: argparse.Namespace) -> None:
    file = Path(args.file).resolve()
    with open(file) as f:
        main_part = strip(f.read())
        module = args.module

        if args.automodule:
            module = file.stem

        if module is not None:
            main_part = f"mod {module} {{\n{main_part}\n}}"

        main_part = escape_dollar(main_part)

        print(main_part)


def snippet_command(args: argparse.Namespace) -> None:
    snippet = dict()

    directory_queue: deque[Path] = deque([Path(args.directory).resolve()])

    default_exclude = ["lib.rs", "mod.rs"]

    try:
        while len(directory_queue) > 0:
            directory = directory_queue.popleft()

            for path in directory.iterdir():
                if path.is_dir():
                    directory_queue.append(path)

                if (
                    not path.is_file()
                    or path.suffix != ".rs"
                    or any(
                        [
                            fnmatch.fnmatch(path.name, pattern)
                            for pattern in args.exclude + default_exclude
                        ]
                    )
                ):
                    continue

                with open(path, "r") as f:
                    name = path.stem

                    if args.automodule:
                        module = name
                    elif args.module is not None:
                        module = args.module
                    else:
                        module = None

                    if name in snippet:
                        print("f{name}は重複しています.新しいものは無視されます.", file=sys.stderr)
                        continue

                    text = strip(f.read())
                    s = to_snippet(text, name, module)
                    snippet[name] = s

        output = json.dumps(snippet, indent=4)

        print(output)

    except Exception as ex:
        print(ex, file=sys.stderr)
        sys.exit(1)


def strip(text: str) -> str:
    res = []
    for line in text.splitlines():
        if re.match(r"^\s*#\[cfg\(test\)\]", line) is not None:
            break

        if re.match(r"^\s*//(?:/|!)", line) is None:
            res.append(line)

    return "\n".join(res)


def to_snippet(text: str, name: str, module: Optional[str]) -> dict:
    if module is not None:
        text = f"mod {module} {{\n" + text + "\n}"

    text = escape_dollar(text)

    lined = [line for line in text.splitlines() if len(line) > 0]

    return dict({"scope": "rust", "prefix": name, "body": lined})


def escape_dollar(text: str) -> str:
    res = ""
    for c in text:
        if c == "$":
            res += "\$"
        else:
            res += c
    return res


# 単純に"#[cfg(test)]"以降を削除する
def strip_test_code(text: str) -> str:
    m = re.search(r"#\[cfg\(test\)\]", text)
    if m is not None:
        return text[: m.start()]
    else:
        return text


if __name__ == "__main__":
    main()
