# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""
One entry point for the repository (the entry point of the Docker image; it also works without Docker).

    python gridstudio.py                        generate the synthetic feeder and the interactive map   (= pipeline)
    python gridstudio.py pipeline [options]     the same, with options (e.g. --seed 7, --no-map, --help)
    python gridstudio.py compare [options]      compare the real feeder with 10 synthetic feeders (structure, cables, one real day)
    python gridstudio.py test                   run the automated tests
    python gridstudio.py help                   this text

Anything that is not a command is passed on to the pipeline, so `gridstudio.py --seed 7` works like `pipeline.py --seed 7`.
Paths: the ENEL data folder is read from ENEL_DATA_DIR (Docker: /data), results are written to OUTPUT_DIR (Docker: /output).
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = {
    "pipeline": ("pipeline.py", "generate the synthetic feeder and the interactive map from the real feeder (default)"),
    "compare": ("compare_real_fake.py", "compare the real feeder with 10 synthetic feeders: structure, cables, one real day (about 10-20 minutes)"),
}


def available() -> dict:
    cmds = {name: (HERE / path, text) for name, (path, text) in SCRIPTS.items() if (HERE / path).exists()}
    if (HERE / "tests").exists():
        cmds["test"] = (None, "run the automated tests")
    cmds["help"] = (None, "show this text")
    return cmds


def usage() -> str:
    cmds = available()
    width = max(len(name) for name in cmds)
    lines = ["usage: gridstudio.py [command] [options]", "", "commands:"]
    lines += [f"  {name:<{width}}  {text}" for name, (_, text) in cmds.items()]
    lines += ["", "Run `gridstudio.py <command> --help` for the options of a command; options without a command go to the pipeline."]
    return "\n".join(lines)


def run(args) -> int:
    return subprocess.call([sys.executable, *map(str, args)], cwd=os.getcwd())


def main(argv) -> int:
    cmds = available()
    cmd = argv[0] if argv else "pipeline"
    rest = argv[1:] if argv and argv[0] in cmds else argv
    if cmd in ("help", "-h", "--help"):
        print(usage())
        if cmd != "help":
            print("\nOptions of the pipeline:\n")
            return run([HERE / "pipeline.py", "--help"])
        return 0
    if cmd in cmds and cmds[cmd][0] is not None:
        return run([cmds[cmd][0], *rest])
    if cmd == "test" and "test" in cmds:
        return run(["-m", "pytest", HERE / "tests", "-q", *rest])
    return run([HERE / "pipeline.py", *argv])          # not a command: options of the pipeline


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
