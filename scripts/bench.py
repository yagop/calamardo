#!/usr/bin/env python3
"""Reproducible prefill and decode benchmark (verification 6).

Measures OpenAI-compatible servers from the client side: every request
streams a chat completion and reports
  - prefill: prompt tokens / time to the first streamed token,
  - decode: tokens after the first chunk / time from the first chunk to the
    end of the stream. A chunk can carry several tokens (a speculative
    decoder sends an accepted block at once), so the first chunk's share is
    estimated as generated / chunks; the end of the stream (the usage chunk)
    covers tokens that print nothing, such as the end-of-turn token.
  Runs that generate fewer than 16 tokens are shown but not summarized.

Each prompt starts with a random nonce, so no request reuses another's
prefix cache. Rounds alternate the order of the targets, so warm-up, thermal
state and background load do not favour whichever runs first.

Targets are URLs (a server already running) or commands that start one;
started servers run one at a time, each only during its own turn:

  scripts/bench.py --target main=http://127.0.0.1:8080
  scripts/bench.py \\
    --target new=http://127.0.0.1:8091 \\
    --launch "new=target/release/calamardo serve --addr 127.0.0.1:8091" \\
    --target old=http://127.0.0.1:8092 \\
    --launch "old=/tmp/calamardo-old serve --addr 127.0.0.1:8092" \\
    --context 1000,8000,30000 --rounds 3

Only the standard library is used.
"""

import argparse
import json
import os
import random
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_text() -> str:
    """The repository's Rust sources, in a fixed order: long, and the same
    text on every machine with the same checkout."""
    parts = []
    for base, dirs, files in sorted(os.walk(os.path.join(ROOT, "crates"))):
        dirs[:] = sorted(d for d in dirs if d not in ("vendor", "target"))
        for name in sorted(files):
            if name.endswith(".rs"):
                with open(os.path.join(base, name), encoding="utf-8") as f:
                    parts.append(f.read())
    return "\n".join(parts)
QUESTION = "Summarize the text above in five sentences."
# Fewer generated tokens than this say little about decode speed.
MIN_GENERATED = 16


def build_prompt(text: str, tokens: int, nonce: str, chars_per_token: float) -> str:
    """About `tokens` tokens of `text`, repeated as needed, after a nonce."""
    chars = max(int(tokens * chars_per_token), 1)
    body = (text * (chars // max(len(text), 1) + 1))[:chars]
    return f"[{nonce}]\n{body}\n\n{QUESTION}"


def get(url: str, timeout: float = 5.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def metrics(base: str) -> dict:
    """Calamardo's counters, if the server has /metrics."""
    try:
        text = get(base + "/metrics").decode()
    except (urllib.error.URLError, OSError, ValueError):
        return {}
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            parts = line.split()
            if len(parts) == 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return out


def wait_ready(base: str, proc, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise SystemExit(f"the server at {base} exited with code {proc.returncode}")
        for path in ("/health", "/v1/models"):
            try:
                get(base + path, timeout=2)
                return
            except (urllib.error.URLError, OSError):
                pass
        time.sleep(1)
    raise SystemExit(f"{base} did not answer within {timeout:.0f} s")


def stream_chat(base: str, model: str, prompt: str, args) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.effort:
        body["reasoning_effort"] = args.effort
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    req = urllib.request.Request(
        base + "/v1/chat/completions", json.dumps(body).encode(), headers
    )
    sent = time.perf_counter()
    first = end = None
    usage = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
                end = time.perf_counter()
            for choice in ev.get("choices") or []:
                d = choice.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
                    first = first or time.perf_counter()
                    chunks += 1
    if usage is None or first is None:
        raise RuntimeError("the stream had no tokens or no usage")
    prompt_tokens = usage["prompt_tokens"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    generated = usage["completion_tokens"]
    ttft = first - sent
    span = end - first
    after_first = generated * (1 - 1 / chunks)
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "generated": generated,
        "ttft": ttft,
        "prefill_tps": (prompt_tokens - cached) / ttft if ttft > 0 else 0.0,
        "decode_tps": after_first / span if span > 0 and chunks > 1 else 0.0,
        "short": generated < MIN_GENERATED,
    }


def parse_pairs(items, what):
    out = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise SystemExit(f"{what} must be NAME=VALUE: {item!r}")
        out[name] = value
    return out


def stop(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target", action="append", required=True, metavar="NAME=URL",
                   help="a server to measure (repeat to compare)")
    p.add_argument("--launch", action="append", metavar="NAME=COMMAND",
                   help="command that starts NAME's server for its turn")
    p.add_argument("--context", default="1000,8000",
                   help="prompt sizes in tokens, approximate (default 1000,8000)")
    p.add_argument("--rounds", type=int, default=3, help="rounds over all targets (default 3)")
    p.add_argument("--requests", type=int, default=1,
                   help="requests per target, size and round (default 1)")
    p.add_argument("--max-tokens", type=int, default=256, help="tokens to generate (default 256)")
    p.add_argument("--effort", default="low",
                   help="reasoning_effort sent with each request ('' to omit; default low)")
    p.add_argument("--model", default="calamardo", help="model name sent with each request")
    p.add_argument("--text", help="file the prompts are built from (default: the Rust sources)")
    p.add_argument("--chars-per-token", type=float, default=2.7,
                   help="characters per token of the text, to size the prompts "
                        "(default 2.7, measured on the Rust sources with Inkling's tokenizer; "
                        "about 4 for English prose)")
    p.add_argument("--api-key", default=os.environ.get("CALAMARDO_API_KEY"))
    p.add_argument("--timeout", type=float, default=3600, help="seconds per request")
    p.add_argument("--startup-timeout", type=float, default=600)
    p.add_argument("--log-dir", help="where launched servers write their output (default: discarded)")
    p.add_argument("--json", help="write every measurement to this file")
    p.add_argument("--seed", type=int, help="seed for the nonces (default: random)")
    args = p.parse_args()

    targets = parse_pairs(args.target, "--target")
    launches = parse_pairs(args.launch, "--launch")
    for name in launches:
        if name not in targets:
            raise SystemExit(f"--launch names {name!r}, which is not a --target")
    sizes = [int(s) for s in args.context.split(",") if s.strip()]
    text = open(args.text, encoding="utf-8").read() if args.text else default_text()
    rng = random.Random(args.seed)
    names = list(targets)
    results = []

    print(f"targets: {', '.join(f'{n} ({targets[n]})' for n in names)}")
    print(f"sizes: {sizes} tokens · {args.rounds} rounds · {args.requests} request(s) each · "
          f"max_tokens {args.max_tokens} · effort {args.effort or '-'}", flush=True)
    for rnd in range(args.rounds):
        # Alternate the order every round.
        order = names if rnd % 2 == 0 else names[::-1]
        for name in order:
            base = targets[name].rstrip("/")
            proc = None
            try:
                if name in launches:
                    out = subprocess.DEVNULL
                    if args.log_dir:
                        os.makedirs(args.log_dir, exist_ok=True)
                        out = open(os.path.join(args.log_dir, f"{name}-round{rnd + 1}.log"), "w")
                    proc = subprocess.Popen(shlex.split(launches[name]), stdout=out, stderr=subprocess.STDOUT)
                wait_ready(base, proc, args.startup_timeout)
                # A short request first: kernels compile and the first
                # allocations happen outside the measurements.
                stream_chat(base, args.model, build_prompt(text, 64, f"warmup-{rng.getrandbits(64):016x}", args.chars_per_token),
                            argparse.Namespace(**{**vars(args), "max_tokens": 8}))
                for size in sizes:
                    for _ in range(args.requests):
                        before = metrics(base)
                        r = stream_chat(base, args.model, build_prompt(text, size, f"{rng.getrandbits(64):016x}", args.chars_per_token), args)
                        after = metrics(base)
                        drafted = after.get("calamardo_drafted_tokens_total", 0) - before.get("calamardo_drafted_tokens_total", 0)
                        accepted = after.get("calamardo_accepted_tokens_total", 0) - before.get("calamardo_accepted_tokens_total", 0)
                        r.update(target=name, size=size, round=rnd + 1)
                        if drafted > 0:
                            r["acceptance"] = accepted / drafted
                        results.append(r)
                        acc = f" · drafts accepted {r['acceptance']:.0%}" if "acceptance" in r else ""
                        if r["short"]:
                            acc += " · too short, not summarized"
                        print(f"round {rnd + 1} {name:>10} ~{size:>6}: prompt {r['prompt_tokens']} "
                              f"({r['cached_tokens']} cached) · prefill {r['prefill_tps']:.0f} tok/s "
                              f"(first token {r['ttft']:.2f} s) · {r['generated']} generated at "
                              f"{r['decode_tps']:.1f} tok/s{acc}", flush=True)
            finally:
                stop(proc)

    print("\nmedians (min-max):")
    print(f"{'target':>10} {'size':>7} {'prefill tok/s':>22} {'decode tok/s':>22}")
    for size in sizes:
        for name in names:
            rs = [r for r in results if r["target"] == name and r["size"] == size and not r["short"]]
            if not rs:
                continue
            cell = lambda k, f: (f"{statistics.median(r[k] for r in rs):{f}} "
                                 f"({min(r[k] for r in rs):{f}}-{max(r[k] for r in rs):{f}})")
            print(f"{name:>10} {size:>7} {cell('prefill_tps', '.0f'):>22} {cell('decode_tps', '.1f'):>22}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {len(results)} measurements to {args.json}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
