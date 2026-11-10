"""Format runtime diagnostics for command-line workflows."""

from pathlib import Path
import re
import subprocess


OMIT = re.compile(
    r"\bdevice\s*\d+\s*:|using device|\bsystem_info\b|\b(?:gpu|cpu)[ _-]?(?:name|info|model|properties)\b"
    r"|\b(?:tokens?|toks?)\s*/\s*(?:s|sec|second)\b|tokens? per second|\btps\b",
    re.IGNORECASE,
)


def filtered_output(text):
    return "".join(line for line in text.splitlines(keepends=True) if not OMIT.search(line))


def run_logged(command, log, env=None):
    with Path(log).open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                if not OMIT.search(line):
                    stream.write(line)
                    stream.flush()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
