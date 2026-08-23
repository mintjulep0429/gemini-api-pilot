import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pilot

MODEL = "gemini-3.1-flash-lite"
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
RESULT_PATH = Path("results/latest.json")


def write_result(payload):
    RESULT_PATH.parent.mkdir(exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload.get("summary", {}), ensure_ascii=False, indent=2))


def build_prompt():
    ids = [task["id"] for task in pilot.TASKS]
    schema = {"answers": {task_id: "<answer>" for task_id in ids}}
    blocks = []
    for task in pilot.TASKS:
        blocks.append(
            f"TASK_ID: {task['id']}\n{task['prompt']}\n"
            "Return only the answer value for this task inside the final answers object."
        )
    return (
        "Solve all tasks independently. Do not use tools, browsing, search, grounding, or external data.\n"
        "Return EXACTLY one JSON object matching this shape:\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\nEach key must appear exactly once. No markdown and no explanation.\n\n"
        + "\n\n".join(blocks)
    )


def call_once(prompt):
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "thinkingConfig": {"thinkingLevel": "minimal"},
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json"
        }
    }).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": API_KEY},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {exc.code}: {body_text}") from exc
    text = "".join(
        part.get("text", "")
        for part in data["candidates"][0]["content"]["parts"]
    )
    parsed = json.loads(text)
    return parsed, text, data.get("usageMetadata", {})


def main():
    base = {
        "model": MODEL,
        "mode": "single_api_call_16_held_out_tasks",
        "task_count": len(pilot.TASKS),
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "truth_boundary": "One real Gemini API call; no tools/grounding supplied; no Claude/OpenAI call.",
        "cost_guard": "Gemini 3.1 Flash-Lite Free Tier + public standard GitHub-hosted runner; no billing setup invoked.",
    }
    if not API_KEY:
        write_result({"summary": {**base, "status": "SECRET_MISSING", "passes": 0, "pass_rate": 0.0}, "results": [], "diagnostic": "GEMINI_API_KEY repository secret unavailable."})
        return
    try:
        parsed, raw, usage = call_once(build_prompt())
        answers = parsed.get("answers", {}) if isinstance(parsed, dict) else {}
        results = []
        family = {}
        for task in pilot.TASKS:
            answer = answers.get(task["id"])
            passed = pilot.validate(task, answer)
            results.append({"task_id": task["id"], "family": task["family"], "pass": passed, "answer": answer})
            row = family.setdefault(task["family"], {"passes": 0, "total": 0})
            row["total"] += 1
            row["passes"] += int(passed)
        for row in family.values():
            row["rate"] = row["passes"] / row["total"]
        passes = sum(int(item["pass"]) for item in results)
        write_result({
            "summary": {**base, "status": "API_OK", "passes": passes, "pass_rate": passes / len(results), "family": family, "usage": usage},
            "results": results,
            "raw_response": raw,
        })
    except Exception as exc:
        write_result({"summary": {**base, "status": "API_ERROR", "passes": 0, "pass_rate": 0.0}, "results": [], "diagnostic": repr(exc)})


if __name__ == "__main__":
    main()
