#!/usr/bin/env python3
"""100 more — sixth round, 10 brand new categories, nothing repeated from v1-v5."""
import json, random
from pathlib import Path
from collections import Counter
random.seed(333333)
OUT = Path("cycle-100-v6"); OUT.mkdir(exist_ok=True)
PHASES = ["DATA","TRAIN","EVAL","ERROR","PUBLISH","DEPLOY","DISCOVER","MONITOR","PROFILE","ITERATE"]
C = ["Cancun","Merida","Puebla","Guadalajara","Monterrey","Tijuana","Acapulco","Puerto_Vallarta","Oaxaca","San_Luis_Potosi","Bogota","Medellin","Cali","Cartagena","Barranquilla","Brasilia","Salvador","Fortaleza","Recife","Porto_Alegre","Lima","Cusco","Arequipa","Trujillo","Valparaiso"]
T = ["AFRM","BILL","UPST","SOFI","PYPL","SQ","COIN","MELI","SE","SHOP","TWLO","SEND","DOCN","GTLB","FRSH","OPEN","Z","YELP","TRIP","ETSY"]
TO = ["mycology","lichenology","bryology","pteridology","palynology","phytochemistry","ethnobotany","dendrochronology","phenology","synecology","aromatherapy","homeopathy","osteopathy","hydropathy","naturopathy"]
L = ["tpi","tvo","giz","kek","kqf","kpf","kaj","kag","kby","kbm","apu","ata","agf","agq","abo"]
MINERALS = ["azurite","malachite","rhodochrosite","fluorite","apatite","topaz","garnet","olivine","spinel","tourmaline","zircon","beryl","jadeite","lapis_lazuli","turquoise"]

def tc(name, args):
    return [{"function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]
def m(role, content=None, tc_val=None):
    x = {"role": role}
    if content is not None: x["content"] = content
    if tc_val: x["tool_calls"] = tc_val
    return x
def wt(name, props, req):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {"type": "object", "properties": props, "required": req}}}

res = []
def sv(i, gap, ex):
    p = PHASES[(i-1)%10]
    with open(OUT/f"iter-{i:03d}-{p.lower()}.jsonl","w") as f:
        f.write(json.dumps(ex, ensure_ascii=False)+"\n")
    res.append({"iter":i,"phase":p,"gap":gap})
    print(f"  #{i:3d} [{p:15s}] {gap}")

# ── 1-10: Progress/status tracking tools (percentage, stage, ETA)
for idx in range(10):
    i=1+idx; c=C[idx%25]; t=T[idx%15]; mneral=MINERALS[idx%15]
    if idx%3==0:
        call = tc("start_analysis",{"data":f"weather_{c}","depth":"full"})
        ex={"messages":[m("user",f"Analyze weather data for {c}"),m("assistant",tc_val=call),m("tool","job_id:j123, status:queued"),m("user","Status?"),m("assistant",tc_val=tc("check_progress",{"job_id":"j123"}))],"tools":[wt("start_analysis",{"data":{"type":"string"},"depth":{"type":"string"}},["data"]),wt("check_progress",{"job_id":{"type":"string"}},["job_id"])]}
        sv(i,f"Progress tracking weather {c}",ex)
    elif idx%3==1:
        call = tc("start_processing",{"task":f"stock_{t}","priority":"high"})
        ex={"messages":[m("user",f"Process stock data for {t}"),m("assistant",tc_val=call),m("tool","job_id:j456, progress:45%"),m("user","How long left?"),m("assistant",tc_val=tc("estimate_completion",{"job_id":"j456"}))],"tools":[wt("start_processing",{"task":{"type":"string"},"priority":{"type":"string"}},["task"]),wt("estimate_completion",{"job_id":{"type":"string"}},["job_id"])]}
        sv(i,f"Progress tracking stock {t}",ex)
    else:
        call = tc("batch_translate",{"texts":["hello","world","friend"],"target_lang":"es"})
        ex={"messages":[m("user","Batch translate 3 words to Spanish"),m("assistant",tc_val=call),m("tool","stage:2/3, eta:10s"),m("user","Status update")],"tools":[wt("batch_translate",{"texts":{"type":"array","items":{"type":"string"}},"target_lang":{"type":"string"}},["texts","target_lang"]),wt("batch_status",{"batch_id":{"type":"string"}},["batch_id"])]}
        sv(i,"Progress tracking batch translate",ex)

# ── 11-20: Idempotent tools (same input = same output)
for idx in range(10):
    i=11+idx; c=C[idx%25]; t=T[idx%15]; mneral=MINERALS[idx%15]
    if idx%3==0:
        call = tc("get_weather_cached",{"location":c,"cache_key":f"wx_{c}_today"})
        ex={"messages":[m("user",f"Weather {c} cached"),m("assistant",tc_val=call)],"tools":[wt("get_weather_cached",{"location":{"type":"string"},"cache_key":{"type":"string"}},["location","cache_key"])]}
        sv(i,f"Idempotent weather {c}",ex)
    elif idx%3==1:
        call = tc("get_stock_price_cached",{"ticker":t,"session":"market_close","cache_ttl":300})
        ex={"messages":[m("user",f"Stock {t} close price cached"),m("assistant",tc_val=call)],"tools":[wt("get_stock_price_cached",{"ticker":{"type":"string"},"session":{"type":"string"},"cache_ttl":{"type":"integer"}},["ticker","session","cache_ttl"])]}
        sv(i,f"Idempotent stock {t}",ex)
    else:
        call = tc("search_cached",{"query":mneral,"cache":True,"expire_after":3600})
        ex={"messages":[m("user",f"Search {mneral} cached"),m("assistant",tc_val=call)],"tools":[wt("search_cached",{"query":{"type":"string"},"cache":{"type":"boolean"},"expire_after":{"type":"integer"}},["query","cache"])]}
        sv(i,f"Idempotent search {mneral}",ex)

# ── 21-30: Validation rules (min/max, pattern, length constraints)
for idx in range(10):
    i=21+idx; c=C[idx%25]; t=T[idx%15]; mneral=MINERALS[idx%15]
    if idx%3==0:
        call = tc("book_flight",{"origin":"BKK","destination":c[:3].upper(),"date":"2026-10-01","passengers":2})
        ex={"messages":[m("user",f"Book flight to {c[:3]} for 2"),m("assistant",tc_val=call)],"tools":[wt("book_flight",{"origin":{"type":"string","minLength":3,"maxLength":4},"destination":{"type":"string","minLength":3,"maxLength":4},"date":{"type":"string","pattern":"\\d{4}-\\d{2}-\\d{2}"},"passengers":{"type":"integer","minimum":1,"maximum":9}},["origin","destination","date"])]}
        sv(i,f"Validation book to {c[:3]}",ex)
    elif idx%3==1:
        call = tc("send_email",{"to":f"user{idx}@example.com","subject":"Validated","body":"OK"})
        ex={"messages":[m("user","Send email with validation"),m("assistant",tc_val=call)],"tools":[wt("send_email",{"to":{"type":"string","pattern":"^[\\w.-]+@[\\w.-]+\\.\\w{2,}$"},"subject":{"type":"string","minLength":1,"maxLength":200},"body":{"type":"string","maxLength":10000}},["to","subject"])]}
        sv(i,"Validation email format",ex)
    else:
        call = tc("search_web",{"query":mneral,"max_results":5,"min_relevance":0.7})
        ex={"messages":[m("user",f"Search {mneral} validated"),m("assistant",tc_val=call)],"tools":[wt("search_web",{"query":{"type":"string","minLength":2},"max_results":{"type":"integer","minimum":1,"maximum":100},"min_relevance":{"type":"number","minimum":0,"maximum":1}},["query","max_results"])]}
        sv(i,f"Validation search {mneral}",ex)

# ── 31-40: Draft/save/publish workflow (multi-stage tool interaction)
for idx in range(10):
    i=31+idx; c=C[idx%25]; t=T[idx%15]; mneral=MINERALS[idx%15]
    if idx%4==0:
        ex={"messages":[m("user",f"Draft email about {c} weather"),m("assistant",tc_val=tc("draft_email",{"to":"team@c.com","subject":f"Weather {c} report","body":f"Weather analysis for {c}"})),m("tool","draft_id:d789, saved"),m("user","Publish it"),m("assistant",tc_val=tc("publish_draft",{"draft_id":"d789","schedule":"now"}))],"tools":[wt("draft_email",{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},["to","subject","body"]),wt("publish_draft",{"draft_id":{"type":"string"},"schedule":{"type":"string","enum":["now","end_of_day","tomorrow"]}},["draft_id","schedule"])]}
        sv(i,f"Draft->publish weather {c}",ex)
    elif idx%4==1:
        ex={"messages":[m("user",f"Create report for {t}"),m("assistant",tc_val=tc("create_report",{"title":f"{t} analysis","type":"quarterly","data":[t]})),m("tool","report_id:r456, status:draft"),m("user","Save and share"),m("assistant",tc_val=tc("save_report",{"report_id":"r456","share_with":["analyst@c.com"]}))],"tools":[wt("create_report",{"title":{"type":"string"},"type":{"type":"string"},"data":{"type":"array","items":{"type":"string"}}},["title","type","data"]),wt("save_report",{"report_id":{"type":"string"},"share_with":{"type":"array","items":{"type":"string"}}},["report_id"])]}
        sv(i,f"Draft->save report {t}",ex)
    elif idx%4==2:
        ex={"messages":[m("user",f"Compose news brief on {mneral}"),m("assistant",tc_val=tc("compose_brief",{"topic":mneral,"format":"summary","tone":"professional"})),m("tool","brief_id:b789, status:editing"),m("user","Finalize"),m("assistant",tc_val=tc("finalize_brief",{"brief_id":"b789","approve":True}))],"tools":[wt("compose_brief",{"topic":{"type":"string"},"format":{"type":"string"},"tone":{"type":"string"}},["topic","format"]),wt("finalize_brief",{"brief_id":{"type":"string"},"approve":{"type":"boolean"}},["brief_id","approve"])]}
        sv(i,f"Draft->finalize brief {mneral}",ex)
    else:
        ex={"messages":[m("user",f"Start flight booking to {c}"),m("assistant",tc_val=tc("initiate_booking",{"destination":c[:3],"passengers":1})),m("tool","booking_id:bk101, awaiting_payment"),m("user","Confirm"),m("assistant",tc_val=tc("confirm_booking",{"booking_id":"bk101","payment_method":"default"}))],"tools":[wt("initiate_booking",{"destination":{"type":"string"},"passengers":{"type":"integer"}},["destination"]),wt("confirm_booking",{"booking_id":{"type":"string"},"payment_method":{"type":"string"}},["booking_id","payment_method"])]}
        sv(i,f"Draft->confirm booking {c[:3]}",ex)

# ── 41-50: Budget/cost-aware tools
for idx in range(10):
    i=41+idx; c=C[idx%25]; t=T[idx%15]; top=TO[idx%15]
    if idx%3==0:
        call = tc("estimate_cost",{"tool":"get_weather","calls":100,"period":"monthly"})
        ex={"messages":[m("user","Cost estimate for weather API?"),m("assistant",tc_val=call)],"tools":[wt("estimate_cost",{"tool":{"type":"string"},"calls":{"type":"integer"},"period":{"type":"string"}},["tool","calls","period"])]}
        sv(i,f"Cost estimate weather API",ex)
    elif idx%3==1:
        c1 = tc("get_stock_price",{"ticker":t})
        c2 = tc("estimate_cost",{"tool":"get_stock_price","calls":10})
        ex={"messages":[m("user",f"Check {t} price and API cost"),m("assistant",tc_val=c1),m("tool","$450"),m("user","What did that cost?"),m("assistant",tc_val=c2)],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"]),wt("estimate_cost",{"tool":{"type":"string"},"calls":{"type":"integer"}},["tool","calls"])]}
        sv(i,f"Cost aware stock {t}",ex)
    else:
        call = tc("get_budget",{"project":top,"year":2026})
        ex={"messages":[m("user",f"Budget for {top} project"),m("assistant",tc_val=call)],"tools":[wt("get_budget",{"project":{"type":"string"},"year":{"type":"integer"}},["project","year"])]}
        sv(i,f"Budget query {top}",ex)

# ── 51-60: Async/polling tools (start job, check status later)
for idx in range(10):
    i=51+idx; t=T[idx%15]; top=TO[idx%15]; mneral=MINERALS[idx%15]
    if idx%3==0:
        call = tc("submit_analysis",{"type":"weather_pattern","location":C[idx%25]})
        ex={"messages":[m("user","Submit weather analysis job"),m("assistant",tc_val=call),m("tool","job_id:async_001, submitted"),m("user","Check results"),m("assistant",tc_val=tc("poll_results",{"job_id":"async_001","timeout":30}))],"tools":[wt("submit_analysis",{"type":{"type":"string"},"location":{"type":"string"}},["type","location"]),wt("poll_results",{"job_id":{"type":"string"},"timeout":{"type":"integer"}},["job_id"])]}
        sv(i,f"Async poll weather analysis",ex)
    elif idx%3==1:
        call = tc("schedule_backtest",{"strategy":t,"period":"6m","start_capital":100000})
        ex={"messages":[m("user",f"Backtest {t} strategy"),m("assistant",tc_val=call),m("tool","backtest_id:bt_001, running"),m("user","Is it done?"),m("assistant",tc_val=tc("check_backtest",{"backtest_id":"bt_001"}))],"tools":[wt("schedule_backtest",{"strategy":{"type":"string"},"period":{"type":"string"},"start_capital":{"type":"integer"}},["strategy","period","start_capital"]),wt("check_backtest",{"backtest_id":{"type":"string"}},["backtest_id"])]}
        sv(i,f"Async poll backtest {t}",ex)
    else:
        call = tc("enqueue_translation",{"texts":["doc1","doc2","doc3"],"target":"th","priority":"background"})
        ex={"messages":[m("user","Enqueue document translations"),m("assistant",tc_val=call),m("tool","queue_id:q789, position:3"),m("user","Check queue"),m("assistant",tc_val=tc("queue_status",{"queue_id":"q789"}))],"tools":[wt("enqueue_translation",{"texts":{"type":"array","items":{"type":"string"}},"target":{"type":"string"},"priority":{"type":"string","enum":["background","normal","urgent"]}},["texts","target"]),wt("queue_status",{"queue_id":{"type":"string"}},["queue_id"])]}
        sv(i,"Async poll translation queue",ex)

# ── 61-70: Tools with resource limits/quotas
for idx in range(10):
    i=61+idx; c=C[idx%25]; t=T[idx%15]; top=TO[idx%15]
    if idx%3==0:
        call = tc("get_api_quota",{"api":"weather","user_id":"current"})
        ex={"messages":[m("user","Check weather API quota"),m("assistant",tc_val=call)],"tools":[wt("get_api_quota",{"api":{"type":"string"},"user_id":{"type":"string"}},["api","user_id"])]}
        sv(i,f"Quota check weather API",ex)
    elif idx%3==1:
        call = tc("check_rate_limit",{"endpoint":"stock_prices","remaining":True})
        ex={"messages":[m("user","Remaining stock API calls?"),m("assistant",tc_val=call)],"tools":[wt("check_rate_limit",{"endpoint":{"type":"string"},"remaining":{"type":"boolean"}},["endpoint","remaining"])]}
        sv(i,f"Rate limit stock API",ex)
    else:
        call = tc("get_usage_stats",{"period":"this_month","group_by":"tool"})
        ex={"messages":[m("user","Monthly API usage stats"),m("assistant",tc_val=call)],"tools":[wt("get_usage_stats",{"period":{"type":"string","enum":["today","this_week","this_month"]},"group_by":{"type":"string"}},["period","group_by"])]}
        sv(i,"Usage stats monthly",ex)

# ── 71-80: Multi-tenant/user context tools
for idx in range(10):
    i=71+idx; c=C[idx%25]; t=T[idx%15]; mneral=MINERALS[idx%15]
    if idx%3==0:
        call = tc("switch_workspace",{"workspace_id":f"team_{idx}","user_context":"analyst"})
        ex={"messages":[m("user",f"Switch to team {idx} workspace"),m("assistant",tc_val=call)],"tools":[wt("switch_workspace",{"workspace_id":{"type":"string"},"user_context":{"type":"string","enum":["analyst","admin","viewer"]}},["workspace_id","user_context"])]}
        sv(i,f"Multi-tenant switch workspace {idx}",ex)
    elif idx%3==1:
        cc = tc("get_weather",{"location":c})
        ex={"messages":[m("user",f"Weather in {c} for client project"),m("assistant",tc_val=tc("set_context",{"project":"client_acme","role":"consultant"})),m("tool","context set"),m("user","Now get weather"),m("assistant",tc_val=cc)],"tools":[wt("set_context",{"project":{"type":"string"},"role":{"type":"string"}},["project","role"]),wt("get_weather",{"location":{"type":"string"}},["location"])]}
        sv(i,f"Multi-tenant context weather {c}",ex)
    else:
        call = tc("get_user_preferences",{"user_id":f"user_{idx}","scope":"tool_settings"})
        ex={"messages":[m("user","Get my tool preferences"),m("assistant",tc_val=call)],"tools":[wt("get_user_preferences",{"user_id":{"type":"string"},"scope":{"type":"string"}},["user_id","scope"])]}
        sv(i,"Multi-tenant user preferences",ex)

# ── 81-90: Export/import data tools
for idx in range(10):
    i=81+idx; c=C[idx%25]; t=T[idx%15]; top=TO[idx%15]; mneral=MINERALS[idx%15]
    if idx%4==0:
        call = tc("export_data",{"format":"csv","source":f"weather_{c}","fields":["date","temp","humidity"]})
        ex={"messages":[m("user",f"Export {c} weather as CSV"),m("assistant",tc_val=call)],"tools":[wt("export_data",{"format":{"type":"string","enum":["csv","json","parquet"]},"source":{"type":"string"},"fields":{"type":"array","items":{"type":"string"}}},["format","source"])]}
        sv(i,f"Export weather {c} CSV",ex)
    elif idx%4==1:
        call = tc("import_data",{"file_url":"s3://bucket/stock_prices.csv","destination":"analysis_db","mapping":{"ticker":"symbol","price":"close"}})
        ex={"messages":[m("user","Import stock CSV to analysis"),m("assistant",tc_val=call)],"tools":[wt("import_data",{"file_url":{"type":"string"},"destination":{"type":"string"},"mapping":{"type":"object"}},["file_url","destination"])]}
        sv(i,"Import stock CSV",ex)
    elif idx%4==2:
        call = tc("convert_format",{"input_format":"xml","output_format":"json","data":"<report>data</report>"})
        ex={"messages":[m("user","Convert XML report to JSON"),m("assistant",tc_val=call)],"tools":[wt("convert_format",{"input_format":{"type":"string"},"output_format":{"type":"string","enum":["csv","json","xml","yaml"]},"data":{"type":"string"}},["input_format","output_format","data"])]}
        sv(i,"Convert XML to JSON",ex)
    else:
        call = tc("backup_data",{"source":"all_tool_history","destination":"archive","retention_days":90})
        ex={"messages":[m("user","Backup tool history to archive"),m("assistant",tc_val=call)],"tools":[wt("backup_data",{"source":{"type":"string"},"destination":{"type":"string"},"retention_days":{"type":"integer"}},["source","destination","retention_days"])]}
        sv(i,"Backup tool history",ex)

# ── 91-100: Audit/logging tools
for idx in range(10):
    i=91+idx; c=C[idx%25]; t=T[idx%15]; top=TO[idx%15]; mneral=MINERALS[idx%15]
    if idx%5==0:
        call = tc("get_audit_log",{"tool":"get_weather","date_from":"2026-07-01","date_to":"2026-07-31","action":"view"})
        ex={"messages":[m("user","Audit weather tool usage July"),m("assistant",tc_val=call)],"tools":[wt("get_audit_log",{"tool":{"type":"string"},"date_from":{"type":"string"},"date_to":{"type":"string"},"action":{"type":"string","enum":["view","create","update","delete"]}},["tool","date_from","date_to"])]}
        sv(i,"Audit weather tool usage",ex)
    elif idx%5==1:
        call = tc("get_user_activity",{"user_id":f"user_{idx}","actions":["tool_call","search","email"]})
        ex={"messages":[m("user",f"Activity log for user {idx}"),m("assistant",tc_val=call)],"tools":[wt("get_user_activity",{"user_id":{"type":"string"},"actions":{"type":"array","items":{"type":"string"}}},["user_id","actions"])]}
        sv(i,f"Audit user activity {idx}",ex)
    elif idx%5==2:
        call = tc("track_change",{"entity":"tool_config","field":"version","old_value":"v1","new_value":"v2"})
        ex={"messages":[m("user","Log tool config version change"),m("assistant",tc_val=call)],"tools":[wt("track_change",{"entity":{"type":"string"},"field":{"type":"string"},"old_value":{"type":"string"},"new_value":{"type":"string"}},["entity","field","old_value","new_value"])]}
        sv(i,"Audit config change v1->v2",ex)
    elif idx%5==3:
        call = tc("get_system_health",{"components":["api","database","cache"],"since":"24h"})
        ex={"messages":[m("user","System health last 24h"),m("assistant",tc_val=call)],"tools":[wt("get_system_health",{"components":{"type":"array","items":{"type":"string"}},"since":{"type":"string"}},["components","since"])]}
        sv(i,"Audit system health 24h",ex)
    else:
        call = tc("get_error_report",{"severity":"error","timeframe":"last_7_days","group_by":"tool_name"})
        ex={"messages":[m("user","Error report for last week"),m("assistant",tc_val=call)],"tools":[wt("get_error_report",{"severity":{"type":"string","enum":["info","warning","error","critical"]},"timeframe":{"type":"string"},"group_by":{"type":"string"}},["severity","timeframe","group_by"])]}
        sv(i,"Audit error report 7d",ex)

print(f"\n{'='*60}")
print(f"CYCLE 100 V6 COMPLETE: {len(res)} new improvements")
print(f"{'='*60}")
with open(OUT/"all-100-v6.jsonl","w") as f:
    for p in sorted(OUT.glob("iter-*.jsonl")):
        f.write(p.read_text(encoding="utf-8"))
pc=Counter(r["phase"] for r in res)
for p in PHASES: print(f"  {p:15s}: {pc.get(p,0)}")
print(f"\n6-round total: 600 improvements")
print(f"  60 unique categories across v1-v6")
print(f"  Every phase: 60 examples each")
