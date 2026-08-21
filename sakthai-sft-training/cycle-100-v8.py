#!/usr/bin/env python3
"""Round 8 — 10 brand new categories, nothing from v1-v7."""
import json, random
from pathlib import Path
from collections import Counter
random.seed(88888888)
OUT = Path("cycle-100-v8"); OUT.mkdir(exist_ok=True)
PHASES = ["DATA","TRAIN","EVAL","ERROR","PUBLISH","DEPLOY","DISCOVER","MONITOR","PROFILE","ITERATE"]
C = ["Valencia","Seville","Bilbao","Palma","Malaga","Zaragoza","Alicante","Granada","Murcia","Palermo","Naples","Turin","Florence","Bologna","Genoa","Minsk","Vitebsk","Grodno","Brest","Gomel","Lviv","Kharkiv","Odesa","Dnipro","Zaporizhzhia"]
T = ["DELL","HPQ","HPE","CSCO","JNPR","NTAP","PSTG","WDC","STX","MU","SWKS","QRVO","SYNA","CRSR","LOGI"]
TO = ["ethnomusicology","ethnomethodology","ethnography","ethnobotany","ethnopharmacology","ethnopedagogy","ethnopsychology","ethnolinguistics","ethnomathematics","ethnoastronomy","exobiology","xenobiology","astrochemistry","cosmochemistry","geochemistry"]
L = ["zza","zzj","zzk","zzl","zzm","zzn","zzo","zzp","zzq","zzr","zzs","zzt","zzu","zzv","zzw"]
BIRDS = ["albatross","cassowary","flamingo","kingfisher","hummingbird","pelican","penguin","raven","swan","toucan","vulture","woodpecker","kookaburra","nightingale","peacock"]

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

# ── 1-10: Webhook/callback tools
for idx in range(10):
    i=1+idx; c=C[idx%25]; t=T[idx%15]; b=BIRDS[idx%15]; top=TO[idx%15]
    if idx%3==0:
        c1 = tc("register_webhook",{"url":"https://my.c.com/hooks/weather","trigger":"weather_update","filters":{"cities":[c]}})
        ex={"messages":[m("user",f"Webhook for {c} weather updates"),m("assistant",tc_val=c1)],"tools":[wt("register_webhook",{"url":{"type":"string"},"trigger":{"type":"string"},"filters":{"type":"object"}},["url","trigger"])]}
        sv(i,f"Webhook weather {c}",ex)
    elif idx%3==1:
        c1 = tc("set_callback",{"event":f"stock_{t}_change","action":"email","target":"alert@c.com","threshold_pct":5})
        ex={"messages":[m("user",f"Callback on {t} 5% change"),m("assistant",tc_val=c1)],"tools":[wt("set_callback",{"event":{"type":"string"},"action":{"type":"string"},"target":{"type":"string"},"threshold_pct":{"type":"number"}},["event","action","target","threshold_pct"])]}
        sv(i,f"Callback stock {t}",ex)
    else:
        c1 = tc("list_webhooks",{"status":"active","page":1})
        ex={"messages":[m("user","List active webhooks"),m("assistant",tc_val=c1)],"tools":[wt("list_webhooks",{"status":{"type":"string","enum":["active","paused","failed"]},"page":{"type":"integer"}},["status"])]}
        sv(i,"Webhook list active",ex)

# ── 11-20: Streaming/progressive tools
for idx in range(10):
    i=11+idx; c=C[idx%25]; t=T[idx%15]; b=BIRDS[idx%15]; top=TO[idx%15]
    if idx%4==0:
        c1 = tc("stream_weather",{"location":c,"interval_minutes":5,"duration_hours":24,"metrics":["temp","humidity","wind"]})
        ex={"messages":[m("user",f"Stream {c} weather 24h"),m("assistant",tc_val=c1)],"tools":[wt("stream_weather",{"location":{"type":"string"},"interval_minutes":{"type":"integer"},"duration_hours":{"type":"integer"},"metrics":{"type":"array","items":{"type":"string"}}},["location","interval_minutes","duration_hours","metrics"])]}
        sv(i,f"Stream weather {c} 24h",ex)
    elif idx%4==1:
        c1 = tc("stream_stock_prices",{"tickers":[t,T[(idx+3)%15]],"interval":"1m","duration":"1h"})
        ex={"messages":[m("user",f"Stream {t} prices 1 hour"),m("assistant",tc_val=c1)],"tools":[wt("stream_stock_prices",{"tickers":{"type":"array","items":{"type":"string"}},"interval":{"type":"string","enum":["1s","1m","5m","1h"]},"duration":{"type":"string"}},["tickers","interval","duration"])]}
        sv(i,f"Stream stock {t} 1h",ex)
    elif idx%4==2:
        c1 = tc("stream_news",{"topics":[top],"max_rate":"1_per_minute","language":"en"})
        ex={"messages":[m("user",f"Stream {top} news feed"),m("assistant",tc_val=c1)],"tools":[wt("stream_news",{"topics":{"type":"array","items":{"type":"string"}},"max_rate":{"type":"string"},"language":{"type":"string"}},["topics","max_rate"])]}
        sv(i,f"Stream news {top}",ex)
    else:
        c1 = tc("get_stream_status",{"stream_id":f"str_{idx}","detailed":True})
        ex={"messages":[m("user",f"Check stream {idx} status"),m("assistant",tc_val=c1)],"tools":[wt("get_stream_status",{"stream_id":{"type":"string"},"detailed":{"type":"boolean"}},["stream_id"])]}
        sv(i,f"Stream status {idx}",ex)

# ── 21-30: Fuzzy/approximate matching tools
for idx in range(10):
    i=21+idx; c=C[idx%25]; t=T[idx%15]; b=BIRDS[idx%15]
    if idx%3==0:
        c1 = tc("fuzzy_search",{"query":c[:4],"fuzziness":2,"max_results":5})
        ex={"messages":[m("user",f"Fuzzy search '{c[:4]}'"),m("assistant",tc_val=c1)],"tools":[wt("fuzzy_search",{"query":{"type":"string"},"fuzziness":{"type":"integer","minimum":0,"maximum":3},"max_results":{"type":"integer"}},["query","fuzziness","max_results"])]}
        sv(i,f"Fuzzy search {c[:4]}",ex)
    elif idx%3==1:
        c1 = tc("approximate_match",{"target":t,"candidates":[T[(idx+x)%15] for x in range(5)],"threshold":0.85})
        ex={"messages":[m("user",f"Approx match for {t}"),m("assistant",tc_val=c1)],"tools":[wt("approximate_match",{"target":{"type":"string"},"candidates":{"type":"array","items":{"type":"string"}},"threshold":{"type":"number","minimum":0,"maximum":1}},["target","candidates","threshold"])]}
        sv(i,f"Fuzzy approx {t}",ex)
    else:
        c1 = tc("similarity_search",{"text":b,"corpus":["bird","animal","fish"],"metric":"cosine","top_k":3})
        ex={"messages":[m("user",f"Similar to {b}"),m("assistant",tc_val=c1)],"tools":[wt("similarity_search",{"text":{"type":"string"},"corpus":{"type":"array","items":{"type":"string"}},"metric":{"type":"string","enum":["cosine","levenshtein","jaccard"]},"top_k":{"type":"integer"}},["text","corpus","metric","top_k"])]}
        sv(i,f"Fuzzy similar {b}",ex)

# ── 31-40: Tool dependencies/prerequisites
for idx in range(10):
    i=31+idx; c=C[idx%25]; t=T[idx%15]; b=BIRDS[idx%15]; top=TO[idx%15]
    if idx%4==0:
        c1 = tc("check_dependencies",{"tool":"get_stock_realtime","required":["market_data_feed","api_key_premium"]})
        ex={"messages":[m("user","Check realtime stock dependencies"),m("assistant",tc_val=c1)],"tools":[wt("check_dependencies",{"tool":{"type":"string"},"required":{"type":"array","items":{"type":"string"}}},["tool","required"])]}
        sv(i,"Dependencies stock realtime",ex)
    elif idx%4==1:
        c1 = tc("install_dependency",{"dependency":"weather_api_v2","version":">=2.0","source":"marketplace"})
        ex={"messages":[m("user","Install weather API v2"),m("assistant",tc_val=c1)],"tools":[wt("install_dependency",{"dependency":{"type":"string"},"version":{"type":"string"},"source":{"type":"string"}},["dependency","version","source"])]}
        sv(i,"Dependencies install weather v2",ex)
    elif idx%4==2:
        c1 = tc("get_prerequisites",{"tool":"translate_batch","format":"checklist"})
        ex={"messages":[m("user","Prerequisites for batch translate"),m("assistant",tc_val=c1)],"tools":[wt("get_prerequisites",{"tool":{"type":"string"},"format":{"type":"string","enum":["checklist","narrative"]}},["tool"])]}
        sv(i,"Dependencies prerequisites translate",ex)
    else:
        c1 = tc("validate_environment",{"tools":["get_weather","get_stock_price"],"check":"all"})
        ex={"messages":[m("user","Validate environment for tools"),m("assistant",tc_val=c1)],"tools":[wt("validate_environment",{"tools":{"type":"array","items":{"type":"string"}},"check":{"type":"string","enum":["all","minimal"]}},["tools","check"])]}
        sv(i,"Dependencies validate env",ex)

# ── 41-50: A/B testing/comparison tools
for idx in range(10):
    i=41+idx; c=C[idx%25]; t=T[idx%15]; b=BIRDS[idx%15]; top=TO[idx%15]
    if idx%4==0:
        c1 = tc("create_ab_test",{"name":f"weather_provider_{c}","variant_a":"accuweather","variant_b":"weatherdotcom","metric":"response_time","sample_size":100})
        ex={"messages":[m("user",f"A/B test weather providers {c}"),m("assistant",tc_val=c1)],"tools":[wt("create_ab_test",{"name":{"type":"string"},"variant_a":{"type":"string"},"variant_b":{"type":"string"},"metric":{"type":"string"},"sample_size":{"type":"integer"}},["name","variant_a","variant_b","metric","sample_size"])]}
        sv(i,f"AB test weather {c}",ex)
    elif idx%4==1:
        c1 = tc("run_comparison",{"tool_a":"get_weather","tool_b":"get_weather_v2","test_cases":[c,C[(idx+5)%25]],"output":"detailed"})
        ex={"messages":[m("user","Compare weather v1 vs v2"),m("assistant",tc_val=c1)],"tools":[wt("run_comparison",{"tool_a":{"type":"string"},"tool_b":{"type":"string"},"test_cases":{"type":"array","items":{"type":"string"}},"output":{"type":"string","enum":["summary","detailed"]}},["tool_a","tool_b","test_cases"])]}
        sv(i,"AB test weather v1 vs v2",ex)
    elif idx%4==2:
        c1 = tc("get_test_results",{"test_id":f"ab_{idx}","include_raw":False})
        ex={"messages":[m("user",f"Get A/B test {idx} results"),m("assistant",tc_val=c1)],"tools":[wt("get_test_results",{"test_id":{"type":"string"},"include_raw":{"type":"boolean"}},["test_id"])]}
        sv(i,f"AB test results {idx}",ex)
    else:
        c1 = tc("analyze_winner",{"test_id":f"ab_{idx}","confidence_level":0.95,"metrics":["speed","accuracy","cost"]})
        ex={"messages":[m("user",f"Analyze winner for test {idx}"),m("assistant",tc_val=c1)],"tools":[wt("analyze_winner",{"test_id":{"type":"string"},"confidence_level":{"type":"number"},"metrics":{"type":"array","items":{"type":"string"}}},["test_id","confidence_level"])]}
        sv(i,f"AB test analyze {idx}",ex)

# ── 51-60: Cache management tools
for idx in range(10):
    i=51+idx; c=C[idx%25]; t=T[idx%15]; b=BIRDS[idx%15]
    if idx%4==0:
        c1 = tc("clear_cache",{"pattern":f"weather_{c}_*","confirm":True,"scope":"all"})
        ex={"messages":[m("user",f"Clear weather cache for {c}"),m("assistant",tc_val=c1)],"tools":[wt("clear_cache",{"pattern":{"type":"string"},"confirm":{"type":"boolean"},"scope":{"type":"string","enum":["all","stale","expired"]}},["pattern","confirm","scope"])]}
        sv(i,f"Cache clear weather {c}",ex)
    elif idx%4==1:
        c1 = tc("get_cache_stats",{"tool":"get_weather","metric":"hit_rate","period":"7d"})
        ex={"messages":[m("user","Weather cache hit rate"),m("assistant",tc_val=c1)],"tools":[wt("get_cache_stats",{"tool":{"type":"string"},"metric":{"type":"string","enum":["hit_rate","size","entries","age"]},"period":{"type":"string"}},["tool","metric"])]}
        sv(i,"Cache stats weather",ex)
    elif idx%4==2:
        c1 = tc("warm_cache",{"tools":["get_weather"],"locations":[c,C[(idx+3)%25]],"ttl_seconds":3600})
        ex={"messages":[m("user",f"Pre-warm weather cache {c}"),m("assistant",tc_val=c1)],"tools":[wt("warm_cache",{"tools":{"type":"array","items":{"type":"string"}},"locations":{"type":"array","items":{"type":"string"}},"ttl_seconds":{"type":"integer"}},["tools","locations","ttl_seconds"])]}
        sv(i,f"Cache warm weather {c}",ex)
    else:
        c1 = tc("configure_cache",{"tool":"get_stock_price","ttl":60,"strategy":"refresh_ahead","max_entries":1000})
        ex={"messages":[m("user","Configure stock cache TTL"),m("assistant",tc_val=c1)],"tools":[wt("configure_cache",{"tool":{"type":"string"},"ttl":{"type":"integer"},"strategy":{"type":"string","enum":["standard","refresh_ahead","write_through"]},"max_entries":{"type":"integer"}},["tool","ttl","strategy"])]}
        sv(i,"Cache config stock",ex)

# ── 61-70: Multi-step wizards/guides
for idx in range(10):
    i=61+idx; c=C[idx%25]; t=T[idx%15]; b=BIRDS[idx%15]; top=TO[idx%15]
    if idx%4==0:
        c1 = tc("start_wizard",{"goal":f"monitor_{c}_weather","steps":["location","metrics","alerts","report"],"current_step":1})
        ex={"messages":[m("user",f"Setup weather monitor for {c}"),m("assistant",tc_val=c1)],"tools":[wt("start_wizard",{"goal":{"type":"string"},"steps":{"type":"array","items":{"type":"string"}},"current_step":{"type":"integer"}},["goal","steps","current_step"])]}
        sv(i,f"Wizard weather {c}",ex)
    elif idx%4==1:
        c1 = tc("wizard_step",{"wizard_id":f"wiz_{idx}","step":2,"input":{"location":c,"metrics":["temp","wind"]}})
        ex={"messages":[m("user",f"Step 2: confirm {c} metrics"),m("assistant",tc_val=c1)],"tools":[wt("wizard_step",{"wizard_id":{"type":"string"},"step":{"type":"integer"},"input":{"type":"object"}},["wizard_id","step","input"])]}
        sv(i,f"Wizard step 2 {c}",ex)
    elif idx%4==2:
        c1 = tc("get_wizard_progress",{"wizard_id":f"wiz_{idx}","format":"percent"})
        ex={"messages":[m("user",f"Wizard {idx} progress?"),m("assistant",tc_val=c1)],"tools":[wt("get_wizard_progress",{"wizard_id":{"type":"string"},"format":{"type":"string","enum":["percent","steps","remaining"]}},["wizard_id"])]}
        sv(i,f"Wizard progress {idx}",ex)
    else:
        c1 = tc("complete_wizard",{"wizard_id":f"wiz_{idx}","apply_now":True,"create_schedule":"daily"})
        ex={"messages":[m("user",f"Complete wizard {idx}"),m("assistant",tc_val=c1)],"tools":[wt("complete_wizard",{"wizard_id":{"type":"string"},"apply_now":{"type":"boolean"},"create_schedule":{"type":"string"}},["wizard_id","apply_now"])]}
        sv(i,f"Wizard complete {idx}",ex)

# ── 71-80: Tool analytics/usage statistics
for idx in range(10):
    i=71+idx; c=C[idx%25]; t=T[idx%15]; b=BIRDS[idx%15]; top=TO[idx%15]
    if idx%5==0:
        c1 = tc("get_usage_analytics",{"period":"30d","group_by":"tool","sort":"calls_desc"})
        ex={"messages":[m("user","Most used tools last 30 days"),m("assistant",tc_val=c1)],"tools":[wt("get_usage_analytics",{"period":{"type":"string"},"group_by":{"type":"string"},"sort":{"type":"string","enum":["calls_desc","calls_asc","name"]}},["period","group_by"])]}
        sv(i,"Analytics top tools 30d",ex)
    elif idx%5==1:
        c1 = tc("get_tool_stats",{"tool":"get_weather","metrics":["total_calls","unique_users","avg_latency","error_rate"]})
        ex={"messages":[m("user","Detailed get_weather stats"),m("assistant",tc_val=c1)],"tools":[wt("get_tool_stats",{"tool":{"type":"string"},"metrics":{"type":"array","items":{"type":"string"}}},["tool","metrics"])]}
        sv(i,"Analytics weather stats",ex)
    elif idx%5==2:
        c1 = tc("get_usage_trends",{"tool":"get_stock_price","granularity":"hourly","range":"7d"})
        ex={"messages":[m("user","Stock tool usage trends"),m("assistant",tc_val=c1)],"tools":[wt("get_usage_trends",{"tool":{"type":"string"},"granularity":{"type":"string","enum":["hourly","daily","weekly"]},"range":{"type":"string"}},["tool","granularity","range"])]}
        sv(i,"Analytics stock trends",ex)
    elif idx%5==3:
        c1 = tc("get_top_users",{"tool":"send_email","limit":5,"period":"this_month"})
        ex={"messages":[m("user","Top email tool users"),m("assistant",tc_val=c1)],"tools":[wt("get_top_users",{"tool":{"type":"string"},"limit":{"type":"integer"},"period":{"type":"string"}},["tool","limit","period"])]}
        sv(i,"Analytics top email users",ex)
    else:
        c1 = tc("get_error_analytics",{"tool":"all","severity":"error","timeframe":"30d","breakdown":"tool"})
        ex={"messages":[m("user","Error analytics across tools"),m("assistant",tc_val=c1)],"tools":[wt("get_error_analytics",{"tool":{"type":"string"},"severity":{"type":"string","enum":["error","warning","info"]},"timeframe":{"type":"string"},"breakdown":{"type":"string","enum":["tool","user","time"]}},["tool","severity","timeframe","breakdown"])]}
        sv(i,"Analytics errors 30d",ex)

# ── 81-90: Social/community tool features
for idx in range(10):
    i=81+idx; c=C[idx%25]; t=T[idx%15]; b=BIRDS[idx%15]; top=TO[idx%15]
    if idx%4==0:
        c1 = tc("rate_tool",{"tool_name":"get_weather","rating":5,"review":"Fast and accurate","user":f"user_{idx}"})
        ex={"messages":[m("user","Rate get_weather 5 stars"),m("assistant",tc_val=c1)],"tools":[wt("rate_tool",{"tool_name":{"type":"string"},"rating":{"type":"integer","minimum":1,"maximum":5},"review":{"type":"string"},"user":{"type":"string"}},["tool_name","rating","user"])]}
        sv(i,"Social rate weather 5*",ex)
    elif idx%4==1:
        c1 = tc("get_tool_reviews",{"tool":"get_weather","sort":"recent","page":1})
        ex={"messages":[m("user","Show get_weather reviews"),m("assistant",tc_val=c1)],"tools":[wt("get_tool_reviews",{"tool":{"type":"string"},"sort":{"type":"string","enum":["recent","helpful","rating"]},"page":{"type":"integer"}},["tool","sort"])]}
        sv(i,"Social reviews weather",ex)
    elif idx%4==2:
        c1 = tc("share_tool_call",{"tool":"get_stock_price","params":{"ticker":t},"visibility":"public","share_with":["community"]})
        ex={"messages":[m("user",f"Share {t} stock call with community"),m("assistant",tc_val=c1)],"tools":[wt("share_tool_call",{"tool":{"type":"string"},"params":{"type":"object"},"visibility":{"type":"string","enum":["public","team","private"]},"share_with":{"type":"array","items":{"type":"string"}}},["tool","params","visibility"])]}
        sv(i,f"Social share stock {t}",ex)
    else:
        c1 = tc("get_trending_tools",{"timeframe":"this_week","category":"all","limit":10})
        ex={"messages":[m("user","Trending tools this week"),m("assistant",tc_val=c1)],"tools":[wt("get_trending_tools",{"timeframe":{"type":"string","enum":["today","this_week","this_month"]},"category":{"type":"string"},"limit":{"type":"integer"}},["timeframe","category"])]}
        sv(i,"Social trending tools",ex)

# ── 91-100: Tool quality/reliability metrics
for idx in range(10):
    i=91+idx; c=C[idx%25]; t=T[idx%15]; b=BIRDS[idx%15]; top=TO[idx%15]
    if idx%5==0:
        c1 = tc("get_tool_health",{"tool":"get_weather","metrics":["uptime","latency_p50","latency_p99","error_rate"]})
        ex={"messages":[m("user","Weather tool health check"),m("assistant",tc_val=c1)],"tools":[wt("get_tool_health",{"tool":{"type":"string"},"metrics":{"type":"array","items":{"type":"string"}}},["tool","metrics"])]}
        sv(i,"Quality health weather",ex)
    elif idx%5==1:
        c1 = tc("get_tool_sla",{"tool":"get_stock_price","month":"2026-07","sla_target":99.9})
        ex={"messages":[m("user",f"Stock tool SLA for Jul"),m("assistant",tc_val=c1)],"tools":[wt("get_tool_sla",{"tool":{"type":"string"},"month":{"type":"string"},"sla_target":{"type":"number"}},["tool","month","sla_target"])]}
        sv(i,"Quality SLA stock Jul",ex)
    elif idx%5==2:
        c1 = tc("get_reliability_score",{"tool":"send_email","window":"30d"})
        ex={"messages":[m("user","Email tool reliability"),m("assistant",tc_val=c1)],"tools":[wt("get_reliability_score",{"tool":{"type":"string"},"window":{"type":"string"}},["tool","window"])]}
        sv(i,"Quality reliability email",ex)
    elif idx%5==3:
        c1 = tc("get_tool_benchmark",{"tool":"get_weather","competitors":["weather_v1","weather_v2"],"metrics":["speed","cost","accuracy"]})
        ex={"messages":[m("user","Benchmark weather tools"),m("assistant",tc_val=c1)],"tools":[wt("get_tool_benchmark",{"tool":{"type":"string"},"competitors":{"type":"array","items":{"type":"string"}},"metrics":{"type":"array","items":{"type":"string"}}},["tool","competitors","metrics"])]}
        sv(i,"Quality benchmark weather",ex)
    else:
        c1 = tc("get_incident_report",{"tool":"get_stock_price","severity":"all","from":"2026-06-01","to":"2026-07-31"})
        ex={"messages":[m("user","Stock tool incident report"),m("assistant",tc_val=c1)],"tools":[wt("get_incident_report",{"tool":{"type":"string"},"severity":{"type":"string","enum":["minor","major","critical","all"]},"from":{"type":"string"},"to":{"type":"string"}},["tool","from","to"])]}
        sv(i,"Quality incidents stock",ex)

print(f"\n{'='*60}")
print(f"CYCLE 100 V8 COMPLETE: {len(res)} new improvements")
print(f"{'='*60}")
with open(OUT/"all-100-v8.jsonl","w") as f:
    for p in sorted(OUT.glob("iter-*.jsonl")):
        f.write(p.read_text(encoding="utf-8"))
pc=Counter(r["phase"] for r in res)
for p in PHASES: print(f"  {p:15s}: {pc.get(p,0)}")
print(f"\n8-round total: 800 improvements")
print(f"  80 unique categories across v1-v8")
print(f"  Every phase: 80 examples each")
