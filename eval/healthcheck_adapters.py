from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

try:
    from .common import STYLE_MODELS
except ImportError:
    from eval.common import STYLE_MODELS


@dataclass
class HealthcheckResult:
    style: str
    model_id: str
    ok: bool
    latency_sec: float
    attempt: int
    response_preview: str
    applied_style: str | None = None
    style_source: str | None = None
    adapter_switched: bool | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Healthcheck for NeuLoRA adapters via local NeuLoRA backend (/api/chat)."
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://127.0.0.1:8800",
        help="NeuLoRA backend base URL.",
    )
    parser.add_argument(
        "--styles",
        type=str,
        default=",".join(STYLE_MODELS.keys()),
        help="Comma-separated adapter styles to check (default: all).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Reply with exactly: HEALTHCHECK_OK",
        help="User prompt used for healthcheck.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="HTTP timeout seconds per request.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count per adapter when request fails.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save detailed JSON result.",
    )
    return parser.parse_args()


def _chat_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/chat"


def _check_one(
    *,
    base_url: str,
    style: str,
    model_id: str,
    prompt: str,
    timeout: float,
    retries: int,
) -> HealthcheckResult:
    endpoint = _chat_endpoint(base_url)
    last_error: str | None = None
    latency = 0.0

    for attempt in range(1, retries + 1):
        body = {
            "message": prompt,
            "thread_id": f"healthcheck-{style}-{uuid4().hex[:8]}",
            "forced_style": style,
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            latency = time.perf_counter() - started
            payload = json.loads(raw)
            answer = (payload.get("answer") or "").strip()
            applied_style = payload.get("applied_style")
            style_source = payload.get("style_source")
            adapter_switched = payload.get("adapter_switched")

            if not answer:
                last_error = "empty answer"
            elif applied_style != style:
                last_error = (
                    f"style mismatch: requested={style}, applied={applied_style}, "
                    f"style_source={style_source}"
                )
            elif adapter_switched is not True:
                last_error = (
                    f"adapter_switched is not true (value={adapter_switched}); "
                    "backend may not be in PEFT multi-adapter mode."
                )
            else:
                return HealthcheckResult(
                    style=style,
                    model_id=model_id,
                    ok=True,
                    latency_sec=round(latency, 3),
                    attempt=attempt,
                    response_preview=answer[:120],
                    applied_style=applied_style,
                    style_source=style_source,
                    adapter_switched=adapter_switched,
                )
        except urllib.error.HTTPError as exc:
            latency = time.perf_counter() - started
            err_body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {err_body[:300]}"
        except urllib.error.URLError as exc:
            latency = time.perf_counter() - started
            last_error = f"URLError: {exc}"
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - started
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < retries:
            time.sleep(min(1.5, 0.5 * attempt))

    return HealthcheckResult(
        style=style,
        model_id=model_id,
        ok=False,
        latency_sec=round(latency, 3),
        attempt=retries,
        response_preview="",
        error=last_error,
    )


def main() -> int:
    args = parse_args()

    requested_styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    invalid = [s for s in requested_styles if s not in STYLE_MODELS]
    if invalid:
        print(
            f"[ERROR] Unsupported style(s): {', '.join(invalid)}. "
            f"Available: {', '.join(STYLE_MODELS.keys())}"
        )
        return 2

    print("== Adapter Healthcheck (/api/chat) ==")
    print(f"backend={args.backend_url}")
    print(f"styles={requested_styles}")

    results: list[HealthcheckResult] = []
    for style in requested_styles:
        model_id = STYLE_MODELS[style]
        result = _check_one(
            base_url=args.backend_url,
            style=style,
            model_id=model_id,
            prompt=args.prompt,
            timeout=args.timeout,
            retries=max(1, args.retries),
        )
        results.append(result)

        status = "OK" if result.ok else "FAIL"
        if result.ok:
            print(
                f"[{status}] style={result.style:<11} model={result.model_id} "
                f"latency={result.latency_sec:.3f}s attempt={result.attempt} "
                f"applied={result.applied_style} source={result.style_source} "
                f"preview={result.response_preview!r}"
            )
        else:
            print(
                f"[{status}] style={result.style:<11} model={result.model_id} "
                f"attempt={result.attempt} error={result.error}"
            )

    ok_count = sum(1 for r in results if r.ok)
    fail_count = len(results) - ok_count
    print("== Summary ==")
    print(f"total={len(results)} ok={ok_count} fail={fail_count}")

    payload = {
        "backend_url": args.backend_url,
        "total": len(results),
        "ok": ok_count,
        "fail": fail_count,
        "results": [asdict(r) for r in results],
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved JSON report to: {args.output_json}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
