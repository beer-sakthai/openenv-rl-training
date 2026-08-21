#!/usr/bin/env python3
"""100 more — seventh round. 10 new categories, nothing from v1-v6."""
import json, random
from pathlib import Path
from collections import Counter
random.seed(7777777)
OUT = Path("cycle-100-v7"); OUT.mkdir(exist_ok=True)
PHASES = ["DATA","TRAIN","EVAL","ERROR","PUBLISH","DEPLOY","DISCOVER","MONITOR","PROFILE","ITERATE"]
C = ["Bristol","Cardiff","Edinburgh","Belfast","Leeds","Sheffield","Nottingham","Leicester","Southampton","Portsmouth","Aberdeen","Dundee","Swansea","Newport","Derry","Krakow","Wroclaw","Poznan","Gdansk","Lodz","Linz","Salzburg","Innsbruck","Graz","Klagenfurt"]
T = ["WDAY","NOW","CRM","ADBE","ADSK","ANSS","CDNS","SNPS","TXN","ADI","NXPI","STM","MRVL","QCOM","AVGO"]
TO = ["speleology","vulcanology","seismology","hydrology","glaciology","oceanography","limnology","potamology","sedimentology","tephrochronology","aerodynamics","thermodynamics","fluid_dynamics","kinematics","mechanics"]
L = ["am","ge","ti","om","or","so","lg","ln","lu","lua","lub","lue","lui","luk","lul"]
GEMS = ["amethyst","citrine","emerald","ruby","sapphire","opal","topaz","turquoise","garnet","peridot","moonstone","aquamarine","heliodor","tanzanite","iolite"]

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

# ── 1-10: Tool discovery/search (finding tools by keywords, categories)
for idx in range(10):
    i=1+idx; c=C[idx%25]; t=T[idx%15]; g=GEMS[idx%15]; top=TO[idx%15]
    if idx%3==0:
        call = tc("find_tool",{"keywords":["weather","temperature","forecast"],"category":"data"})
        ex={"messages":[m("user","Find tools for weather data"),m("assistant",tc_val=call)],"tools":[wt("find_tool",{"keywords":{"type":"array","items":{"type":"string"}},"category":{"type":"string"}},["keywords"])]}
        sv(i,"Tool discovery weather tools",ex)
    elif idx%3==1:
        call = tc("search_tools",{"query":"stock market price","sort_by":"popularity","limit":5})
        ex={"messages":[m("user","Search stock market tools"),m("assistant",tc_val=call)],"tools":[wt("search_tools",{"query":{"type":"string"},"sort_by":{"type":"string","enum":["popularity","name","recent"]},"limit":{"type":"integer"}},["query","sort_by"])]}
        sv(i,"Tool discovery stock tools",ex)
    else:
        call = tc("get_tool_categories",{"parent":"communication","depth":2})
        ex={"messages":[m("user","What communication tools exist?"),m("assistant",tc_val=call)],"tools":[wt("get_tool_categories",{"parent":{"type":"string"},"depth":{"type":"integer"}},["parent"])]}
        sv(i,"Tool discovery categories",ex)

# ── 11-20: Tool favorites/bookmarks
for idx in range(10):
    i=11+idx; c=C[idx%25]; t=T[idx%15]; g=GEMS[idx%15]
    if idx%4==0:
        call = tc("add_favorite",{"tool_name":"get_weather","user_nickname":"wx_check","tags":["daily","weather"]})
        ex={"messages":[m("user","Add get_weather to favorites"),m("assistant",tc_val=call)],"tools":[wt("add_favorite",{"tool_name":{"type":"string"},"user_nickname":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}}},["tool_name"])]}
        sv(i,"Favorites add weather tool",ex)
    elif idx%4==1:
        call = tc("list_favorites",{"sort_by":"last_used","max_items":10})
        ex={"messages":[m("user","Show my favorite tools"),m("assistant",tc_val=call)],"tools":[wt("list_favorites",{"sort_by":{"type":"string","enum":["name","last_used","most_used"]},"max_items":{"type":"integer"}},["sort_by"])]}
        sv(i,"Favorites list tools",ex)
    elif idx%4==2:
        call = tc("get_recent_tools",{"hours":24,"include_usage":True})
        ex={"messages":[m("user","Recently used tools"),m("assistant",tc_val=call)],"tools":[wt("get_recent_tools",{"hours":{"type":"integer"},"include_usage":{"type":"boolean"}},["hours"])]}
        sv(i,"Favorites recent tools",ex)
    else:
        call = tc("remove_favorite",{"tool_name":"get_time","confirm":True})
        ex={"messages":[m("user","Remove get_time from favorites"),m("assistant",tc_val=call)],"tools":[wt("remove_favorite",{"tool_name":{"type":"string"},"confirm":{"type":"boolean"}},["tool_name","confirm"])]}
        sv(i,"Favorites remove tool",ex)

# ── 21-30: Tool suggestions/recommendations
for idx in range(10):
    i=21+idx; c=C[idx%25]; t=T[idx%15]; top=TO[idx%15]; g=GEMS[idx%15]
    if idx%4==0:
        call = tc("suggest_tool",{"query":f"weather in {c}","context":["travel","planning"],"user_id":"current"})
        ex={"messages":[m("user",f"Suggest tool for {c} trip"),m("assistant",tc_val=call)],"tools":[wt("suggest_tool",{"query":{"type":"string"},"context":{"type":"array","items":{"type":"string"}},"user_id":{"type":"string"}},["query","user_id"])]}
        sv(i,f"Suggest tool weather {c}",ex)
    elif idx%4==1:
        call = tc("recommend_tools",{"task_description":"monitor stock portfolio daily","frequency":"daily","preferred_tools":["get_stock_price"]})
        ex={"messages":[m("user","Recommend tools for stock monitoring"),m("assistant",tc_val=call)],"tools":[wt("recommend_tools",{"task_description":{"type":"string"},"frequency":{"type":"string","enum":["once","daily","weekly"]},"preferred_tools":{"type":"array","items":{"type":"string"}}},["task_description"])]}
        sv(i,"Suggest tools stock monitoring",ex)
    elif idx%4==2:
        call = tc("autocomplete_tool",{"partial":"get_","max_suggestions":5})
        ex={"messages":[m("user","Autocomplete get_ tools"),m("assistant",tc_val=call)],"tools":[wt("autocomplete_tool",{"partial":{"type":"string"},"max_suggestions":{"type":"integer"}},["partial"])]}
        sv(i,"Suggest autocomplete get_",ex)
    else:
        call = tc("get_similar_tools",{"tool_name":"get_weather","criteria":"functionality","limit":3})
        ex={"messages":[m("user","Similar tools to get_weather"),m("assistant",tc_val=call)],"tools":[wt("get_similar_tools",{"tool_name":{"type":"string"},"criteria":{"type":"string","enum":["functionality","popularity","category"]},"limit":{"type":"integer"}},["tool_name","criteria"])]}
        sv(i,"Suggest similar to weather",ex)

# ── 31-40: Undo/redo and version history for tool calls
for idx in range(10):
    i=31+idx; c=C[idx%25]; t=T[idx%15]; g=GEMS[idx%15]
    if idx%4==0:
        c1 = tc("get_weather",{"location":c})
        c2 = tc("get_time",{"location":c})
        ex={"messages":[m("user",f"Weather in {c}?"),m("assistant",tc_val=c1),m("user","Actually get time"),m("assistant",tc_val=c2),m("user","Undo that, weather please"),m("assistant",tc_val=tc("undo_last_call",{"steps":1}))],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"]),wt("get_time",{"location":{"type":"string"}},["location"]),wt("undo_last_call",{"steps":{"type":"integer"}},["steps"])]}
        sv(i,f"Undo weather/time {c}",ex)
    elif idx%4==1:
        c1 = tc("send_email",{"to":"a@b.com","subject":"First","body":"Version1"})
        c2 = tc("send_email",{"to":"a@b.com","subject":"Second","body":"Version2"})
        ex={"messages":[m("user","Send email v1"),m("assistant",tc_val=c1),m("user","Change to v2"),m("assistant",tc_val=c2),m("user","Redo v1"),m("assistant",tc_val=tc("redo_version",{"action_id":"email_1"}))],"tools":[wt("send_email",{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},["to","subject"]),wt("redo_version",{"action_id":{"type":"string"}},["action_id"])]}
        sv(i,"Undo redo email versions",ex)
    elif idx%4==2:
        call = tc("get_action_history",{"tool":"get_weather","user":"self","limit":5})
        ex={"messages":[m("user","Show my last 5 weather calls"),m("assistant",tc_val=call)],"tools":[wt("get_action_history",{"tool":{"type":"string"},"user":{"type":"string"},"limit":{"type":"integer"}},["tool","user","limit"])]}
        sv(i,"Undo action history weather",ex)
    else:
        call = tc("rollback_changes",{"tool":"send_email","since":"2026-07-01","reason":"test"})
        ex={"messages":[m("user","Rollback email changes since July"),m("assistant",tc_val=call)],"tools":[wt("rollback_changes",{"tool":{"type":"string"},"since":{"type":"string"},"reason":{"type":"string"}},["tool","since"])]}
        sv(i,"Undo rollback email changes",ex)

# ── 41-50: Template/snippet tools (reusable tool call templates)
for idx in range(10):
    i=41+idx; c=C[idx%25]; t=T[idx%15]; g=GEMS[idx%15]
    if idx%4==0:
        call = tc("create_template",{"name":"morning_brief","steps":["get_weather","get_news","get_stock_price"],"schedule":"daily_9am"})
        ex={"messages":[m("user","Create morning brief template"),m("assistant",tc_val=call)],"tools":[wt("create_template",{"name":{"type":"string"},"steps":{"type":"array","items":{"type":"string"}},"schedule":{"type":"string"}},["name","steps"])]}
        sv(i,"Template morning brief",ex)
    elif idx%4==1:
        call = tc("run_template",{"template_name":"morning_brief","params":{"cities":[c],"topics":["AI"],"tickers":[t]}})
        ex={"messages":[m("user","Run morning brief template"),m("assistant",tc_val=call)],"tools":[wt("run_template",{"template_name":{"type":"string"},"params":{"type":"object"}},["template_name","params"])]}
        sv(i,f"Template run brief {c}",ex)
    elif idx%4==2:
        call = tc("save_snippet",{"name":"check_weather_and_stock","tools":["get_weather","get_stock_price"],"defaults":{"location":"Bangkok","ticker":"AAPL"}})
        ex={"messages":[m("user","Save weather+stock snippet"),m("assistant",tc_val=call)],"tools":[wt("save_snippet",{"name":{"type":"string"},"tools":{"type":"array","items":{"type":"string"}},"defaults":{"type":"object"}},["name","tools"])]}
        sv(i,"Template save snippet",ex)
    else:
        call = tc("list_templates",{"category":"daily","shared":False})
        ex={"messages":[m("user","My daily templates"),m("assistant",tc_val=call)],"tools":[wt("list_templates",{"category":{"type":"string"},"shared":{"type":"boolean"}},["category","shared"])]}
        sv(i,"Template list daily",ex)

# ── 51-60: Tool composition (combining tools into composite)
for idx in range(10):
    i=51+idx; c=C[idx%25]; t=T[idx%15]; top=TO[idx%15]; g=GEMS[idx%15]
    if idx%4==0:
        c1 = tc("get_weather",{"location":c})
        c2 = tc("get_restaurant_info",{"name":"any","location":c})
        c3 = tc("get_directions",{"origin":"current","destination":c})
        ex={"messages":[m("user",f"Plan a day in {c}"),m("assistant",tc_val=c1),m("tool","Sunny 25C"),m("assistant",tc_val=c2),m("tool","Italian place nearby"),m("assistant",tc_val=c3)],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"]),wt("get_restaurant_info",{"name":{"type":"string"},"location":{"type":"string"}},["name"]),wt("get_directions",{"origin":{"type":"string"},"destination":{"type":"string"}},["origin","destination"])]}
        sv(i,f"Composite plan day {c}",ex)
    elif idx%4==1:
        c1 = tc("get_stock_price",{"ticker":t})
        c2 = tc("get_news",{"topic":t,"count":3})
        ex={"messages":[m("user",f"Full {t} analysis"),m("assistant",tc_val=c1),m("tool","$450"),m("assistant",tc_val=c2)],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"]),wt("get_news",{"topic":{"type":"string"},"count":{"type":"integer"}},["topic"])]}
        sv(i,f"Composite stock analysis {t}",ex)
    elif idx%4==2:
        c1 = tc("translate_text",{"text":"guide","target_lang":"th"})
        c2 = tc("search_web",{"query":"thai travel guide"})
        ex={"messages":[m("user","Create Thai travel guide"),m("assistant",tc_val=c1),m("tool","translated"),m("assistant",tc_val=c2)],"tools":[wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        sv(i,"Composite travel guide TH",ex)
    else:
        c1 = tc("get_news",{"topic":top,"count":5})
        c2 = tc("send_email",{"to":"digest@c.com","subject":f"{top} digest","body":"compiled"})
        ex={"messages":[m("user",f"Compile and email {top} digest"),m("assistant",tc_val=c1),m("tool","5 articles"),m("assistant",tc_val=c2)],"tools":[wt("get_news",{"topic":{"type":"string"},"count":{"type":"integer"}},["topic"]),wt("send_email",{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},["to","subject","body"])]}
        sv(i,f"Composite news digest {top}",ex)

# ── 61-70: Visualization/chart tools
for idx in range(10):
    i=61+idx; c=C[idx%25]; t=T[idx%15]; top=TO[idx%15]; g=GEMS[idx%15]
    if idx%3==0:
        call = tc("create_chart",{"type":"line","data_source":f"weather_{c}","x_axis":"date","y_axis":"temp","title":f"Weather {c} Jul"})
        ex={"messages":[m("user",f"Chart weather data for {c}"),m("assistant",tc_val=call)],"tools":[wt("create_chart",{"type":{"type":"string","enum":["line","bar","pie","scatter"]},"data_source":{"type":"string"},"x_axis":{"type":"string"},"y_axis":{"type":"string"},"title":{"type":"string"}},["type","data_source","x_axis","y_axis"])]}
        sv(i,f"Visual chart weather {c}",ex)
    elif idx%3==1:
        call = tc("create_dashboard",{"panels":[{"tool":"get_stock_price","ticker":t,"chart":"line"},{"tool":"get_news","topic":t,"chart":"bar"}],"name":f"{t}_dashboard"})
        ex={"messages":[m("user",f"Dashboard for {t}"),m("assistant",tc_val=call)],"tools":[wt("create_dashboard",{"panels":{"type":"array","items":{"type":"object"}},"name":{"type":"string"}},["panels","name"])]}
        sv(i,f"Visual dashboard {t}",ex)
    else:
        call = tc("generate_report",{"format":"pdf","sections":[f"weather_{c}","news_{top}"],"include_charts":True})
        ex={"messages":[m("user",f"PDF report {c} weather + {top} news"),m("assistant",tc_val=call)],"tools":[wt("generate_report",{"format":{"type":"string","enum":["pdf","html","markdown"]},"sections":{"type":"array","items":{"type":"string"}},"include_charts":{"type":"boolean"}},["format","sections"])]}
        sv(i,f"Visual report {c}+{top}",ex)

# ── 71-80: Notification/alert tools
for idx in range(10):
    i=71+idx; c=C[idx%25]; t=T[idx%15]; top=TO[idx%15]; g=GEMS[idx%15]
    if idx%4==0:
        call = tc("set_alert",{"condition":f"temperature in {c} > 35","action":"notify","channel":"email","target":"me@c.com"})
        ex={"messages":[m("user",f"Alert me when {c} exceeds 35C"),m("assistant",tc_val=call)],"tools":[wt("set_alert",{"condition":{"type":"string"},"action":{"type":"string"},"channel":{"type":"string","enum":["email","sms","slack"]},"target":{"type":"string"}},["condition","action","channel","target"])]}
        sv(i,f"Alert weather {c} >35",ex)
    elif idx%4==1:
        call = tc("set_price_alert",{"ticker":t,"threshold":500,"direction":"above","alert_type":"once"})
        ex={"messages":[m("user",f"Alert when {t} exceeds $500"),m("assistant",tc_val=call)],"tools":[wt("set_price_alert",{"ticker":{"type":"string"},"threshold":{"type":"number"},"direction":{"type":"string","enum":["above","below"]},"alert_type":{"type":"string","enum":["once","every"]}},["ticker","threshold","direction"])]}
        sv(i,f"Alert stock {t}>500",ex)
    elif idx%4==2:
        call = tc("get_active_alerts",{"status":"active","page":1})
        ex={"messages":[m("user","Show my active alerts"),m("assistant",tc_val=call)],"tools":[wt("get_active_alerts",{"status":{"type":"string","enum":["active","paused","triggered"]},"page":{"type":"integer"}},["status"])]}
        sv(i,"Alert list active",ex)
    else:
        call = tc("create_news_alert",{"keywords":[top,"breakthrough"],"source":"preprint","notify_immediately":True})
        ex={"messages":[m("user",f"Alert on {top} breakthroughs"),m("assistant",tc_val=call)],"tools":[wt("create_news_alert",{"keywords":{"type":"array","items":{"type":"string"}},"source":{"type":"string"},"notify_immediately":{"type":"boolean"}},["keywords","source"])]}
        sv(i,f"Alert news {top}",ex)

# ── 81-90: Tool permissions/roles
for idx in range(10):
    i=81+idx; c=C[idx%25]; t=T[idx%15]; g=GEMS[idx%15]
    if idx%4==0:
        call = tc("check_tool_access",{"tool":"get_stock_price","user":f"user_{idx}","required_role":"analyst"})
        ex={"messages":[m("user",f"Can user {idx} access stock tools?"),m("assistant",tc_val=call)],"tools":[wt("check_tool_access",{"tool":{"type":"string"},"user":{"type":"string"},"required_role":{"type":"string"}},["tool","user","required_role"])]}
        sv(i,f"Permissions check stock {idx}",ex)
    elif idx%4==1:
        call = tc("grant_tool_access",{"tool":"get_weather","user":f"user_{idx}","role":"viewer","expires":"2027-01-01"})
        ex={"messages":[m("user",f"Grant weather access to user {idx}"),m("assistant",tc_val=call)],"tools":[wt("grant_tool_access",{"tool":{"type":"string"},"user":{"type":"string"},"role":{"type":"string","enum":["viewer","user","admin"]},"expires":{"type":"string"}},["tool","user","role"])]}
        sv(i,f"Permissions grant weather {idx}",ex)
    elif idx%4==2:
        call = tc("get_tool_permissions",{"tool":"send_email","format":"detailed"})
        ex={"messages":[m("user","Who can send emails?"),m("assistant",tc_val=call)],"tools":[wt("get_tool_permissions",{"tool":{"type":"string"},"format":{"type":"string","enum":["simple","detailed"]}},["tool"])]}
        sv(i,"Permissions email list",ex)
    else:
        call = tc("revoke_tool_access",{"tool":"get_time","user":f"user_{idx}","reason":"expired_trial"})
        ex={"messages":[m("user",f"Revoke time tool for user {idx}"),m("assistant",tc_val=call)],"tools":[wt("revoke_tool_access",{"tool":{"type":"string"},"user":{"type":"string"},"reason":{"type":"string"}},["tool","user","reason"])]}
        sv(i,f"Permissions revoke time {idx}",ex)

# ── 91-100: Tool testing/sandbox mode
for idx in range(10):
    i=91+idx; c=C[idx%25]; t=T[idx%15]; g=GEMS[idx%15]; top=TO[idx%15]
    if idx%4==0:
        call = tc("dry_run",{"tool":"send_email","params":{"to":"test@t.com","subject":"Test","body":"Dry"},"simulate":True})
        ex={"messages":[m("user","Dry run send_email test"),m("assistant",tc_val=call)],"tools":[wt("dry_run",{"tool":{"type":"string"},"params":{"type":"object"},"simulate":{"type":"boolean"}},["tool","params","simulate"])]}
        sv(i,"Sandbox dry run email",ex)
    elif idx%4==1:
        call = tc("enter_sandbox",{"mode":"isolated","allowed_tools":["get_weather","get_time"],"reset_on_exit":True})
        ex={"messages":[m("user","Enter sandbox mode"),m("assistant",tc_val=call)],"tools":[wt("enter_sandbox",{"mode":{"type":"string","enum":["isolated","monitored"]},"allowed_tools":{"type":"array","items":{"type":"string"}},"reset_on_exit":{"type":"boolean"}},["mode","allowed_tools"])]}
        sv(i,"Sandbox enter isolated",ex)
    elif idx%4==2:
        call = tc("run_test_suite",{"tool":"get_weather","test_cases":[{"location":"Paris"},{"location":"","expect_error":True}],"report":"detailed"})
        ex={"messages":[m("user","Test get_weather with cases"),m("assistant",tc_val=call)],"tools":[wt("run_test_suite",{"tool":{"type":"string"},"test_cases":{"type":"array","items":{"type":"object"}},"report":{"type":"string","enum":["simple","detailed"]}},["tool","test_cases"])]}
        sv(i,"Sandbox test weather",ex)
    else:
        call = tc("exit_sandbox",{"discard_changes":True})
        ex={"messages":[m("user","Exit sandbox discard changes"),m("assistant",tc_val=call)],"tools":[wt("exit_sandbox",{"discard_changes":{"type":"boolean"}},["discard_changes"])]}
        sv(i,"Sandbox exit discard",ex)

print(f"\n{'='*60}")
print(f"CYCLE 100 V7 COMPLETE: {len(res)} new improvements")
print(f"{'='*60}")
with open(OUT/"all-100-v7.jsonl","w") as f:
    for p in sorted(OUT.glob("iter-*.jsonl")):
        f.write(p.read_text(encoding="utf-8"))
pc=Counter(r["phase"] for r in res)
for p in PHASES: print(f"  {p:15s}: {pc.get(p,0)}")
print(f"\n7-round total: 700 improvements")
print(f"  70 unique categories across v1-v7")
print(f"  Every phase: 70 examples each")
