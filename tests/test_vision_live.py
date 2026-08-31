#!/usr/bin/env python3
"""Live image-request acceptance gate. Run through ``./dev vision-test``."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _request(url: str, payload: dict | None, timeout: int) -> tuple[int, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


def main() -> int:
    if os.environ.get("KRASIS_DEV_SCRIPT") != "1":
        raise SystemExit("Run this gate through ./dev vision-test; direct execution is unsupported.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--expect", action="append", default=[])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        print(f"FAIL image does not exist: {image_path}")
        return 1
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    data_url = f"data:{mime};base64,{encoded}"
    base_url = f"http://127.0.0.1:{args.port}"

    model_status, model_body = _request(f"{base_url}/v1/models", None, args.timeout)
    if model_status != 200 or not isinstance(model_body, dict):
        print(f"FAIL /v1/models status={model_status} body={model_body}")
        return 1
    models = model_body.get("data") or []
    first_model = models[0] if models and isinstance(models[0], dict) else {}
    capabilities = first_model.get("capabilities") or {}
    input_modalities = first_model.get("input_modalities") or []
    advertises_vision = bool(capabilities.get("vision")) and "image" in input_modalities
    if not advertises_vision:
        print(f"FAIL server does not advertise vision: {model_body}")
        return 1
    model_id = str(first_model.get("id") or "")

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": False,
    }
    status, body = _request(f"{base_url}/v1/chat/completions", payload, args.timeout)
    if status != 200 or not isinstance(body, dict):
        print(f"FAIL image request status={status} body={body}")
        return 1
    try:
        text = str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        print(f"FAIL malformed image response: {body}")
        return 1
    folded = text.casefold()
    missing = [term for term in args.expect if term.casefold() not in folded]
    if missing:
        print(f"FAIL response omitted expected terms {missing}: {text}")
        return 1
    print(f"PASS model={model_id} image={image_path.name} response={text!r}")
    timing = body.get("krasis_timing")
    if timing is not None:
        print("TIMING " + json.dumps(timing, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
