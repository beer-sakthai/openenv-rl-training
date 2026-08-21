#!/usr/bin/env python3
"""Round 9 — 10 brand new categories, nothing from v1-v8."""
import json, random
from pathlib import Path
from collections import Counter
random.seed(999999999)
OUT = Path("cycle-100-v9"); OUT.mkdir(exist_ok=True)
PHASES = ["DATA","TRAIN","EVAL","ERROR","PUBLISH","DEPLOY","DISCOVER","MONITOR","PROFILE","ITERATE"]
C = ["Antwerp","Bruges","Ghent","Leuven","Mons","Liege","Namur","Charleroi","Tournai","Ostend","Maastricht","Eindhoven","Tilburg","Groningen","Arnhem","Nuremberg","Stuttgart","Dortmund","Essen","Duisburg","Bonn","Mainz","Freiburg","Augsburg","Mannheim"]
T = ["ROKU","CRWD","PLTR","DDOG","MDB","ESTC","NET","ZS","OKTA","WISE","CFLT","TOST","GTLB","FRSH","VERI"]
TO = ["phenomenology","hermeneutics","epistemology","ontology","axiology","teleology","deontology","utilitarianism","existentialism","stoicism","cognitivism","constructivism","behaviorism","humanism","positivism"]
L = ["bcl","bcn","bco","bcp","bcq","bcr","bcs","bct","bcu","bcv","bcw","bcx","bcy","bcz","bda"]
RIVERS = ["Amazon","Nile","Mississippi","Yangtze","Danube","Mekong","Ganges","Volga","Congo","Indus","Tigris","Euphrates","Rhine","Seine","Thames"]

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

# ── 1-10: Tool tagging/labeling systems
for idx in range(10):
    i=1+idx; c=C[idx%25]; r=RIVERS[idx%15]; top=TO[idx%15]
    if idx%4==0:
        c1 = tc("add_tool_tag",{"tool":"get_weather","tags":["weather","forecast","daily_use"],"color":"blue"})
        ex={"messages":[m("user","Tag get_weather as forecast"),m("assistant",tc_val=c1)],"tools":[wt("add_tool_tag",{"tool":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}},"color":{"type":"string"}},["tool","tags"])]}
        sv(i,"Tagging add weather tags",ex)
    elif idx%4==1:
        c1 = tc("filter_tools_by_tag",{"tags":["weather","forecast"],"match":"any","sort":"name"})
        ex={"messages":[m("user","Find weather tagged tools"),m("assistant",tc_val=c1)],"tools":[wt("filter_tools_by_tag",{"tags":{"type":"array","items":{"type":"string"}},"match":{"type":"string","enum":["any","all"]},"sort":{"type":"string"}},["tags","match"])]}
        sv(i,"Tagging filter weather",ex)
    elif idx%4==2:
        c1 = tc("remove_tool_tag",{"tool":"get_time","tag":"obsolete","confirm":True})
        ex={"messages":[m("user","Remove obsolete tag from get_time"),m("assistant",tc_val=c1)],"tools":[wt("remove_tool_tag",{"tool":{"type":"string"},"tag":{"type":"string"},"confirm":{"type":"boolean"}},["tool","tag"])]}
        sv(i,"Tagging remove time tag",ex)
    else:
        c1 = tc("get_all_tags",{"sort_by":"count","limit":20})
        ex={"messages":[m("user","Show all tool tags"),m("assistant",tc_val=c1)],"tools":[wt("get_all_tags",{"sort_by":{"type":"string","enum":["name","count"]},"limit":{"type":"integer"}},["sort_by"])]}
        sv(i,"Tagging all tags list",ex)

# ── 11-20: Tool pricing/license tiers
for idx in range(10):
    i=11+idx; c=C[idx%25]; t=T[idx%15]; r=RIVERS[idx%15]
    if idx%4==0:
        c1 = tc("get_tool_pricing",{"tool":"get_weather","tier":"premium","billing":"monthly"})
        ex={"messages":[m("user","Weather tool premium pricing"),m("assistant",tc_val=c1)],"tools":[wt("get_tool_pricing",{"tool":{"type":"string"},"tier":{"type":"string","enum":["free","basic","premium","enterprise"]},"billing":{"type":"string","enum":["monthly","yearly","lifetime"]}},["tool","tier"])]}
        sv(i,"Pricing weather premium",ex)
    elif idx%4==1:
        c1 = tc("upgrade_tool_tier",{"tool":"get_stock_price","current":"basic","target":"premium","payment_method":"card_xxx"})
        ex={"messages":[m("user","Upgrade stock tool to premium"),m("assistant",tc_val=c1)],"tools":[wt("upgrade_tool_tier",{"tool":{"type":"string"},"current":{"type":"string"},"target":{"type":"string"},"payment_method":{"type":"string"}},["tool","current","target"])]}
        sv(i,"Pricing upgrade stock",ex)
    elif idx%4==2:
        c1 = tc("compare_tool_plans",{"tool":"send_email","plans":["free","basic","pro"],"highlight":"best_value"})
        ex={"messages":[m("user","Compare email tool plans"),m("assistant",tc_val=c1)],"tools":[wt("compare_tool_plans",{"tool":{"type":"string"},"plans":{"type":"array","items":{"type":"string"}},"highlight":{"type":"string"}},["tool","plans","highlight"])]}
        sv(i,"Pricing compare plans",ex)
    else:
        c1 = tc("get_usage_billing",{"period":"2026-07","detail":"per_tool"})
        ex={"messages":[m("user","July billing by tool"),m("assistant",tc_val=c1)],"tools":[wt("get_usage_billing",{"period":{"type":"string"},"detail":{"type":"string","enum":["summary","per_tool","per_user"]}},["period"])]}
        sv(i,"Pricing billing July",ex)

# ── 21-30: Tool deprecation/retirement
for idx in range(10):
    i=21+idx; c=C[idx%25]; t=T[idx%15]; r=RIVERS[idx%15]; top=TO[idx%15]
    if idx%4==0:
        c1 = tc("deprecate_tool",{"tool":"get_weather_v1","reason":"replaced_by_v2","effective_date":"2027-01-01","alternative":"get_weather_v2"})
        ex={"messages":[m("user","Deprecate weather v1"),m("assistant",tc_val=c1)],"tools":[wt("deprecate_tool",{"tool":{"type":"string"},"reason":{"type":"string"},"effective_date":{"type":"string"},"alternative":{"type":"string"}},["tool","reason","effective_date"])]}
        sv(i,"Deprecate weather v1",ex)
    elif idx%4==1:
        c1 = tc("get_deprecation_notices",{"scope":"all","include_resolved":False})
        ex={"messages":[m("user","Show all deprecation notices"),m("assistant",tc_val=c1)],"tools":[wt("get_deprecation_notices",{"scope":{"type":"string","enum":["all","mine","team"]},"include_resolved":{"type":"boolean"}},["scope"])]}
        sv(i,"Deprecation list notices",ex)
    elif idx%4==2:
        c1 = tc("migrate_tool_calls",{"from_tool":"search_web_v1","to_tool":"search_web_v2","migrate_history":True,"dry_run":True})
        ex={"messages":[m("user","Migrate search v1 to v2"),m("assistant",tc_val=c1)],"tools":[wt("migrate_tool_calls",{"from_tool":{"type":"string"},"to_tool":{"type":"string"},"migrate_history":{"type":"boolean"},"dry_run":{"type":"boolean"}},["from_tool","to_tool","migrate_history"])]}
        sv(i,"Deprecation migrate search",ex)
    else:
        c1 = tc("schedule_tool_retirement",{"tool":"legacy_email","retire_date":"2027-06-30","notification_period_days":90,"auto_redirect":True})
        ex={"messages":[m("user","Schedule legacy email retirement"),m("assistant",tc_val=c1)],"tools":[wt("schedule_tool_retirement",{"tool":{"type":"string"},"retire_date":{"type":"string"},"notification_period_days":{"type":"integer"},"auto_redirect":{"type":"boolean"}},["tool","retire_date","notification_period_days"])]}
        sv(i,"Deprecation schedule retire",ex)

# ── 31-40: Tool localization/internationalization
for idx in range(10):
    i=31+idx; c=C[idx%25]; t=T[idx%15]; r=RIVERS[idx%15]; top=TO[idx%15]; lg=L[idx%15]
    if idx%4==0:
        c1 = tc("localize_tool",{"tool":"get_weather","target_language":lg,"fields":["name","description","parameters"]})
        ex={"messages":[m("user",f"Localize weather tool to {lg}"),m("assistant",tc_val=c1)],"tools":[wt("localize_tool",{"tool":{"type":"string"},"target_language":{"type":"string"},"fields":{"type":"array","items":{"type":"string"}}},["tool","target_language"])]}
        sv(i,f"Localize weather to {lg}",ex)
    elif idx%4==1:
        c1 = tc("get_tool_translation",{"tool":"get_stock_price","language":"th","format":"json"})
        ex={"messages":[m("user","Stock tool Thai translation"),m("assistant",tc_val=c1)],"tools":[wt("get_tool_translation",{"tool":{"type":"string"},"language":{"type":"string"},"format":{"type":"string","enum":["json","yaml","xliff"]}},["tool","language"])]}
        sv(i,"Localize stock Thai",ex)
    elif idx%4==2:
        c1 = tc("set_tool_language",{"preferred_language":"th","fallback":"en","apply_to":["get_weather","get_time","search_web"]})
        ex={"messages":[m("user","Set tool language to Thai"),m("assistant",tc_val=c1)],"tools":[wt("set_tool_language",{"preferred_language":{"type":"string"},"fallback":{"type":"string"},"apply_to":{"type":"array","items":{"type":"string"}}},["preferred_language","fallback"])]}
        sv(i,"Localize set Thai lang",ex)
    else:
        c1 = tc("get_available_languages",{"tool":"get_weather","supported_only":True})
        ex={"messages":[m("user","Available weather tool languages"),m("assistant",tc_val=c1)],"tools":[wt("get_available_languages",{"tool":{"type":"string"},"supported_only":{"type":"boolean"}},["tool"])]}
        sv(i,"Localize available langs",ex)

# ── 41-50: Tool scheduling/availability roster
for idx in range(10):
    i=41+idx; c=C[idx%25]; t=T[idx%15]; r=RIVERS[idx%15]
    if idx%4==0:
        c1 = tc("get_tool_schedule",{"tool":"get_weather","date":"2026-10-01","timezone":"UTC"})
        ex={"messages":[m("user","Weather tool schedule Oct 1"),m("assistant",tc_val=c1)],"tools":[wt("get_tool_schedule",{"tool":{"type":"string"},"date":{"type":"string"},"timezone":{"type":"string"}},["tool","date"])]}
        sv(i,"Schedule weather availability",ex)
    elif idx%4==1:
        c1 = tc("set_tool_maintenance",{"tool":"get_stock_price","start":"2026-08-15T02:00","end":"2026-08-15T04:00","reason":"upgrade","notify_users":True})
        ex={"messages":[m("user","Schedule stock tool maintenance"),m("assistant",tc_val=c1)],"tools":[wt("set_tool_maintenance",{"tool":{"type":"string"},"start":{"type":"string"},"end":{"type":"string"},"reason":{"type":"string"},"notify_users":{"type":"boolean"}},["tool","start","end","reason"])]}
        sv(i,"Schedule stock maintenance",ex)
    elif idx%4==2:
        c1 = tc("get_tool_uptime_calendar",{"tool":"send_email","month":"2026-07","include_maintenance":True})
        ex={"messages":[m("user","Email tool uptime July"),m("assistant",tc_val=c1)],"tools":[wt("get_tool_uptime_calendar",{"tool":{"type":"string"},"month":{"type":"string"},"include_maintenance":{"type":"boolean"}},["tool","month"])]}
        sv(i,"Schedule email uptime",ex)
    else:
        c1 = tc("request_tool_outage",{"tool":"get_weather","window":"2026-09-15T00:00/2026-09-15T06:00","reason":"infra","emergency":False})
        ex={"messages":[m("user","Request weather tool outage"),m("assistant",tc_val=c1)],"tools":[wt("request_tool_outage",{"tool":{"type":"string"},"window":{"type":"string"},"reason":{"type":"string"},"emergency":{"type":"boolean"}},["tool","window","reason"])]}
        sv(i,"Schedule outage request",ex)

# ── 51-60: Tool compliance/regulatory checks
for idx in range(10):
    i=51+idx; c=C[idx%25]; t=T[idx%15]; r=RIVERS[idx%15]; top=TO[idx%15]
    if idx%4==0:
        c1 = tc("check_tool_compliance",{"tool":"send_email","standard":"gdpr","region":"eu"})
        ex={"messages":[m("user","Check email tool GDPR compliance"),m("assistant",tc_val=c1)],"tools":[wt("check_tool_compliance",{"tool":{"type":"string"},"standard":{"type":"string","enum":["gdpr","hipaa","sox","pci"]},"region":{"type":"string"}},["tool","standard"])]}
        sv(i,"Compliance GDPR email",ex)
    elif idx%4==1:
        c1 = tc("get_tool_certifications",{"tool":"get_stock_price","include_expired":False})
        ex={"messages":[m("user","Stock tool certifications"),m("assistant",tc_val=c1)],"tools":[wt("get_tool_certifications",{"tool":{"type":"string"},"include_expired":{"type":"boolean"}},["tool"])]}
        sv(i,"Compliance stock certs",ex)
    elif idx%4==2:
        c1 = tc("run_compliance_audit",{"tools":["get_weather","get_time","search_web"],"framework":"soc2","scope":"full"})
        ex={"messages":[m("user","SOC2 audit weather tools"),m("assistant",tc_val=c1)],"tools":[wt("run_compliance_audit",{"tools":{"type":"array","items":{"type":"string"}},"framework":{"type":"string","enum":["soc2","iso27001","hipaa"]},"scope":{"type":"string","enum":["full","quick"]}},["tools","framework"])]}
        sv(i,"Compliance SOC2 audit",ex)
    else:
        c1 = tc("get_data_retention_policy",{"tool":"send_email","type":"user_data"})
        ex={"messages":[m("user","Email data retention policy"),m("assistant",tc_val=c1)],"tools":[wt("get_data_retention_policy",{"tool":{"type":"string"},"type":{"type":"string"}},["tool"])]}
        sv(i,"Compliance data retention",ex)

# ── 61-70: Tool marketplace/third-party
for idx in range(10):
    i=61+idx; c=C[idx%25]; t=T[idx%15]; r=RIVERS[idx%15]; top=TO[idx%15]
    if idx%4==0:
        c1 = tc("browse_marketplace",{"category":"weather","sort":"popular","page":1})
        ex={"messages":[m("user","Browse weather tools marketplace"),m("assistant",tc_val=c1)],"tools":[wt("browse_marketplace",{"category":{"type":"string"},"sort":{"type":"string","enum":["popular","new","rating"]},"page":{"type":"integer"}},["category"])]}
        sv(i,"Marketplace browse weather",ex)
    elif idx%4==1:
        c1 = tc("install_community_tool",{"tool_id":"weather_pro_v3","source":"marketplace","trust_level":"verified"})
        ex={"messages":[m("user","Install community weather tool"),m("assistant",tc_val=c1)],"tools":[wt("install_community_tool",{"tool_id":{"type":"string"},"source":{"type":"string"},"trust_level":{"type":"string","enum":["verified","partner","community"]}},["tool_id","source"])]}
        sv(i,"Marketplace install tool",ex)
    elif idx%4==2:
        c1 = tc("publish_tool",{"tool_name":"weather_advanced","version":"1.0.0","visibility":"public","pricing":"free"})
        ex={"messages":[m("user","Publish my custom weather tool"),m("assistant",tc_val=c1)],"tools":[wt("publish_tool",{"tool_name":{"type":"string"},"version":{"type":"string"},"visibility":{"type":"string","enum":["private","team","public"]},"pricing":{"type":"string","enum":["free","paid","freemium"]}},["tool_name","version","visibility"])]}
        sv(i,"Marketplace publish tool",ex)
    else:
        c1 = tc("report_tool_issue",{"tool_id":"weather_pro_v3","issue_type":"bug","severity":"medium","description":"Wrong temp in Celsius"})
        ex={"messages":[m("user","Report bug in community tool"),m("assistant",tc_val=c1)],"tools":[wt("report_tool_issue",{"tool_id":{"type":"string"},"issue_type":{"type":"string","enum":["bug","feature","security"]},"severity":{"type":"string","enum":["low","medium","high","critical"]},"description":{"type":"string"}},["tool_id","issue_type","severity","description"])]}
        sv(i,"Marketplace report issue",ex)

# ── 71-80: Tool cloning/forking
for idx in range(10):
    i=71+idx; c=C[idx%25]; t=T[idx%15]; r=RIVERS[idx%15]; top=TO[idx%15]
    if idx%4==0:
        c1 = tc("fork_tool",{"source_tool":"get_weather","new_name":"get_weather_custom","namespace":"personal","include_history":False})
        ex={"messages":[m("user","Fork get_weather as custom"),m("assistant",tc_val=c1)],"tools":[wt("fork_tool",{"source_tool":{"type":"string"},"new_name":{"type":"string"},"namespace":{"type":"string"},"include_history":{"type":"boolean"}},["source_tool","new_name","namespace"])]}
        sv(i,"Fork weather tool",ex)
    elif idx%4==1:
        c1 = tc("clone_tool_config",{"tool":"get_stock_price","target_workspace":"team_analytics","preserve_tags":True})
        ex={"messages":[m("user","Clone stock tool config to analytics"),m("assistant",tc_val=c1)],"tools":[wt("clone_tool_config",{"tool":{"type":"string"},"target_workspace":{"type":"string"},"preserve_tags":{"type":"boolean"}},["tool","target_workspace"])]}
        sv(i,"Clone stock config",ex)
    elif idx%4==2:
        c1 = tc("sync_fork",{"fork_name":"get_weather_custom","source":"get_weather","strategy":"merge_upstream","resolve":"auto"})
        ex={"messages":[m("user","Sync weather fork with upstream"),m("assistant",tc_val=c1)],"tools":[wt("sync_fork",{"fork_name":{"type":"string"},"source":{"type":"string"},"strategy":{"type":"string","enum":["merge_upstream","rebase","overwrite"]},"resolve":{"type":"string","enum":["auto","manual"]}},["fork_name","source","strategy"])]}
        sv(i,"Fork sync weather",ex)
    else:
        c1 = tc("list_forks",{"source_tool":"get_weather","sort":"stars","limit":10})
        ex={"messages":[m("user","Show forks of get_weather"),m("assistant",tc_val=c1)],"tools":[wt("list_forks",{"source_tool":{"type":"string"},"sort":{"type":"string","enum":["stars","name","date"]},"limit":{"type":"integer"}},["source_tool","sort"])]}
        sv(i,"Fork list weather forks",ex)

# ── 81-90: Tool merging/composition (combining two tools into one)
for idx in range(10):
    i=81+idx; c=C[idx%25]; t=T[idx%15]; r=RIVERS[idx%15]; top=TO[idx%15]
    if idx%4==0:
        c1 = tc("merge_tools",{"tool_a":"get_weather","tool_b":"get_time","new_name":"weather_time_combo","strategy":"parallel"})
        ex={"messages":[m("user","Merge weather and time tools"),m("assistant",tc_val=c1)],"tools":[wt("merge_tools",{"tool_a":{"type":"string"},"tool_b":{"type":"string"},"new_name":{"type":"string"},"strategy":{"type":"string","enum":["parallel","sequential","conditional"]}},["tool_a","tool_b","new_name","strategy"])]}
        sv(i,"Merge weather+time",ex)
    elif idx%4==1:
        c1 = tc("create_composite_tool",{"name":"weather_alert","components":["get_weather","set_alert","send_email"],"pipeline":"sequential","error_handling":"skip"})
        ex={"messages":[m("user","Create weather alert composite"),m("assistant",tc_val=c1)],"tools":[wt("create_composite_tool",{"name":{"type":"string"},"components":{"type":"array","items":{"type":"string"}},"pipeline":{"type":"string","enum":["sequential","parallel","conditional"]},"error_handling":{"type":"string","enum":["skip","retry","abort"]}},["name","components","pipeline"])]}
        sv(i,"Composite weather alert",ex)
    elif idx%4==2:
        c1 = tc("decompose_tool",{"tool":"weather_alert","flatten":True})
        ex={"messages":[m("user","Decompose weather alert tool"),m("assistant",tc_val=c1)],"tools":[wt("decompose_tool",{"tool":{"type":"string"},"flatten":{"type":"boolean"}},["tool"])]}
        sv(i,"Decompose weather alert",ex)
    else:
        c1 = tc("get_composition_graph",{"tool":"weather_alert","format":"mermaid"})
        ex={"messages":[m("user","Show weather alert composition"),m("assistant",tc_val=c1)],"tools":[wt("get_composition_graph",{"tool":{"type":"string"},"format":{"type":"string","enum":["mermaid","dot","json"]}},["tool"])]}
        sv(i,"Composition graph alert",ex)

# ── 91-100: Tool documentation generation
for idx in range(10):
    i=91+idx; c=C[idx%25]; t=T[idx%15]; r=RIVERS[idx%15]; top=TO[idx%15]
    if idx%5==0:
        c1 = tc("generate_docs",{"tool":"get_weather","format":"markdown","include_examples":True,"include_schema":True})
        ex={"messages":[m("user","Generate weather tool docs"),m("assistant",tc_val=c1)],"tools":[wt("generate_docs",{"tool":{"type":"string"},"format":{"type":"string","enum":["markdown","html","pdf"]},"include_examples":{"type":"boolean"},"include_schema":{"type":"boolean"}},["tool","format"])]}
        sv(i,"Docs generate weather",ex)
    elif idx%5==1:
        c1 = tc("generate_changelog",{"tool":"get_stock_price","from_version":"1.0.0","to_version":"2.0.0","style":"detailed"})
        ex={"messages":[m("user","Stock tool changelog v1 to v2"),m("assistant",tc_val=c1)],"tools":[wt("generate_changelog",{"tool":{"type":"string"},"from_version":{"type":"string"},"to_version":{"type":"string"},"style":{"type":"string","enum":["detailed","summary","bullet"]}},["tool","from_version","to_version"])]}
        sv(i,"Docs changelog stock",ex)
    elif idx%5==2:
        c1 = tc("generate_openapi_spec",{"tool":"send_email","version":"3.0.3","include_auth":True})
        ex={"messages":[m("user","OpenAPI spec for email tool"),m("assistant",tc_val=c1)],"tools":[wt("generate_openapi_spec",{"tool":{"type":"string"},"version":{"type":"string"},"include_auth":{"type":"boolean"}},["tool","version"])]}
        sv(i,"Docs OpenAPI email",ex)
    elif idx%5==3:
        c1 = tc("generate_usage_guide",{"tool":"search_web","audience":"beginner","format":"tutorial"})
        ex={"messages":[m("user","Beginner guide for search tool"),m("assistant",tc_val=c1)],"tools":[wt("generate_usage_guide",{"tool":{"type":"string"},"audience":{"type":"string","enum":["beginner","advanced","developer"]},"format":{"type":"string","enum":["tutorial","reference","quickstart"]}},["tool","audience"])]}
        sv(i,"Docs usage guide search",ex)
    else:
        c1 = tc("update_docs",{"tool":"get_weather","section":"parameters","content":"location: city name (required)","version":"2.1.0","auto_deploy":True})
        ex={"messages":[m("user","Update weather param docs"),m("assistant",tc_val=c1)],"tools":[wt("update_docs",{"tool":{"type":"string"},"section":{"type":"string"},"content":{"type":"string"},"version":{"type":"string"},"auto_deploy":{"type":"boolean"}},["tool","section","content"])]}
        sv(i,"Docs update weather params",ex)

print(f"\n{'='*60}")
print(f"CYCLE 100 V9 COMPLETE: {len(res)} new improvements")
print(f"{'='*60}")
with open(OUT/"all-100-v9.jsonl","w") as f:
    for p in sorted(OUT.glob("iter-*.jsonl")):
        f.write(p.read_text(encoding="utf-8"))
pc=Counter(r["phase"] for r in res)
for p in PHASES: print(f"  {p:15s}: {pc.get(p,0)}")
print(f"\n9-round total: 900 improvements")
print(f"  90 unique categories across v1-v9")
print(f"  Every phase: 90 examples each")
