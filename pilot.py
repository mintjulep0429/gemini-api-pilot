import json, os, re, time, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime, timezone

MODEL = "gemini-2.5-flash"
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

TASKS = [
 {"id":"code_repair_01","family":"code_repair","prompt":"Return ONLY JSON: {\"answer\":\"...\"}. Fix the Python expression so it returns the sum of squares of even numbers in xs. Broken: sum(x*x for x in xs if x % 2 == 1). Return only the corrected expression.","type":"one_of","answers":["sum(x*x for x in xs if x % 2 == 0)"]},
 {"id":"code_repair_02","family":"code_repair","prompt":"Return ONLY JSON: {\"answer\":\"...\"}. Fix this SQL WHERE clause so NULL statuses are also included: status <> 'closed'. Return only the corrected SQL boolean expression.","type":"one_of","answers":["status <> 'closed' OR status IS NULL","status IS NULL OR status <> 'closed'","status != 'closed' OR status IS NULL","status IS NULL OR status != 'closed'"]},
 {"id":"code_repair_03","family":"code_repair","prompt":"Return ONLY JSON: {\"answer\":\"...\"}. A retry loop sleeps before the first attempt: for i in range(3): time.sleep(2**i); call(). Give the minimal Python statement that should guard the sleep so the first attempt is immediate.","type":"one_of","answers":["if i > 0:","if i:"]},
 {"id":"code_repair_04","family":"code_repair","prompt":"Return ONLY JSON: {\"answer\":\"...\"}. Fix this JavaScript equality check to reject type coercion: if (user.id == requestedId). Return only the corrected condition expression without if.","type":"one_of","answers":["user.id === requestedId","requestedId === user.id"]},
 {"id":"data_analysis_01","family":"data_analysis","prompt":"Return ONLY JSON: {\"answer\": number}. A test has 240 cases. 18 fail. What is the pass rate as a percentage, rounded to one decimal place?","type":"numeric","answer":92.5},
 {"id":"data_analysis_02","family":"data_analysis","prompt":"Return ONLY JSON: {\"answer\": number}. Three runs take 42, 58, and 50 seconds. What is the arithmetic mean in seconds?","type":"numeric","answer":50.0},
 {"id":"data_analysis_03","family":"data_analysis","prompt":"Return ONLY JSON: {\"answer\": number}. A system reduces duplicate effects from 8.0% to 2.0%. What is the relative reduction percentage?","type":"numeric","answer":75.0},
 {"id":"data_analysis_04","family":"data_analysis","prompt":"Return ONLY JSON: {\"answer\": number}. Utility = 0.4*completion + 0.3*verification - 0.2*duplicate_effect - 0.1*rework. completion=.9, verification=.8, duplicate_effect=.05, rework=.1. Return the utility.","type":"numeric","answer":0.58},
 {"id":"evidence_01","family":"research_evidence","prompt":"Return ONLY JSON: {\"answer\":\"E#\"}. Evidence: E1: The pilot used 16 held-out tasks. E2: The model was Gemini 2.5 Flash. E3: Search grounding was disabled. Which evidence directly supports the claim that no search tool was used?","type":"exact","answer":"E3"},
 {"id":"evidence_02","family":"research_evidence","prompt":"Return ONLY JSON: {\"answer\":[\"E#\",\"E#\"]}. Evidence: E1: Free tier input is free of charge. E2: Free tier output is free of charge. E3: Paid tier has token prices. Which TWO evidence items support the claim that standard text generation on the stated free tier has no token charge? Return IDs ascending.","type":"list","answer":["E1","E2"]},
 {"id":"evidence_03","family":"research_evidence","prompt":"Return ONLY JSON: {\"answer\":\"SUPPORTED\" or \"NOT_SUPPORTED\"}. Evidence: E1: The experiment recorded model outputs and token usage. E2: No statement describes Claude being executed. Claim: Claude was empirically benchmarked in this experiment. Is it supported?","type":"exact","answer":"NOT_SUPPORTED"},
 {"id":"evidence_04","family":"research_evidence","prompt":"Return ONLY JSON: {\"answer\":\"...\"}. Evidence: E1: A simulation produced a higher score for Method A. E2: No real-provider run was executed. Choose exactly one: SIMULATION_ONLY, EMPIRICALLY_SUPERIOR, WORLD_FIRST.","type":"exact","answer":"SIMULATION_ONLY"},
 {"id":"mixed_01","family":"mixed_long_horizon","prompt":"Return ONLY JSON: {\"answer\":[\"...\"]}. Goal: independently checked answer. A=one model answers (1 call), B=a second independent model answers (1 call), C=first model self-checks (1 call), D=deterministic verifier checks both answers (0 model calls). Independent checking requires two independently generated answers; self-check does not count; final output must be deterministically checked. Return cheapest valid action sequence.","type":"list","answer":["A","B","D"]},
 {"id":"mixed_02","family":"mixed_long_horizon","prompt":"Return ONLY JSON: {\"answer\":\"...\"}. A material effect has status UNKNOWN after a network timeout. Choose exactly one: RETRY, RECONCILE, MARK_PASS, IGNORE.","type":"exact","answer":"RECONCILE"},
 {"id":"mixed_03","family":"mixed_long_horizon","prompt":"Return ONLY JSON: {\"answer\":\"...\"}. Two evaluators both derive their judgment from the same source transcript. Choose exactly one: INDEPENDENT, CORRELATED, UNKNOWN.","type":"exact","answer":"CORRELATED"},
 {"id":"mixed_04","family":"mixed_long_horizon","prompt":"Return ONLY JSON: {\"answer\":\"...\"}. A challenger improves mean utility by 0.021. The preregistered meaningful margin is 0.038. Its confidence interval is entirely above zero. Choose exactly one: CHALLENGER_WIN, NO_WINNER, INCUMBENT_WIN.","type":"exact","answer":"NO_WINNER"}
]

def norm(v):
    return re.sub(r"\\s+", " ", str(v).strip())

def validate(t, answer):
    if t["type"] == "numeric":
        try: return abs(float(answer) - float(t["answer"])) < 1e-9
        except Exception: return False
    if t["type"] == "exact": return str(answer).strip() == str(t["answer"]).strip()
    if t["type"] == "one_of": return norm(answer) in {norm(x) for x in t["answers"]}
    if t["type"] == "list": return answer == t["answer"]
    return False

def call(prompt):
    body = json.dumps({
      "contents":[{"parts":[{"text":prompt}]}],
      "generationConfig":{"maxOutputTokens":512,"responseMimeType":"application/json"}
    }).encode()
    req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
      "Content-Type":"application/json", "x-goog-api-key":API_KEY
    })
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())
    text = "".join(p.get("text","") for p in data["candidates"][0]["content"]["parts"])
    parsed = json.loads(text)
    return parsed.get("answer"), text, data.get("usageMetadata", {})

def main():
    if not API_KEY:
        raise SystemExit("GEMINI_API_KEY secret missing")
    results=[]
    for i,t in enumerate(TASKS,1):
        started=time.time()
        rec={"task_id":t["id"],"family":t["family"],"pass":False}
        try:
            ans, raw, usage = call(t["prompt"])
            rec.update(answer=ans, raw=raw, usage=usage, pass=validate(t,ans), elapsed_s=round(time.time()-started,3), http_ok=True)
        except Exception as e:
            rec.update(answer=None, raw="", usage={}, elapsed_s=round(time.time()-started,3), http_ok=False, error=repr(e))
        results.append(rec)
        print(f"[{i:02d}/16] {t['id']} {'PASS' if rec['pass'] else 'FAIL'}")
        if i < len(TASKS): time.sleep(12)
    fam={}
    for r in results:
        x=fam.setdefault(r["family"],{"passes":0,"total":0})
        x["total"]+=1; x["passes"]+=int(r["pass"])
    for x in fam.values(): x["rate"]=x["passes"]/x["total"]
    passes=sum(int(r["pass"]) for r in results)
    usage_totals={}
    for r in results:
        for k,v in r.get("usage",{}).items():
            if isinstance(v,(int,float)): usage_totals[k]=usage_totals.get(k,0)+v
    payload={
      "summary":{"model":MODEL,"task_count":len(results),"passes":passes,"pass_rate":passes/len(results),"family":fam,"usage_totals":usage_totals,"executed_utc":datetime.now(timezone.utc).isoformat(),"truth_boundary":"Real Gemini API calls only; no tools/grounding supplied; no Claude/OpenAI call.","cost_guard":"Gemini Free Tier + public standard GitHub-hosted runner; no billing setup invoked."},
      "results":results
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/latest.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(payload["summary"],ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
