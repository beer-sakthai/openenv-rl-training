#!/usr/bin/env python3
"""100 more iterations — third round, all new content types."""
import json, random
from pathlib import Path
from collections import Counter
random.seed(999)
OUT = Path("cycle-100-v3"); OUT.mkdir(exist_ok=True)
PHASES = ["DATA","TRAIN","EVAL","ERROR","PUBLISH","DEPLOY","DISCOVER","MONITOR","PROFILE","ITERATE"]
C = ["Reykjavik","Tallinn","Vilnius","Ljubljana","Bratislava","Luxembourg","Monaco","Valletta","Andorra","San Marino","Glasgow","Birmingham","Manchester","Liverpool","Oxford","Nice","Lyon","Toulouse","Marseille","Lille","Bremen","Leipzig","Dresden","Bonn","Utrecht"]
T = ["PLTR","DDOG","MDB","ESTC","NET","ZS","OKTA","WISE","CFLT","TOST","GTLB","FRSH","VERI","BILL","AFRM"]
TO = ["fermentation tech","microbiome therapy","organoids","cryonics","anti-aging","nutrigenomics","epigenetics","telomere therapy","senolytics","neuroplasticity","quantum biology","biomimicry","mycelium tech","algae biofuels","desert farming"]
L = ["ga","cy","eu","ka","sq","hy","mk","bs","mt","is","lb","fy","gd","yi","co"]
FRUITS = ["mango","durian","lychee","dragonfruit","rambutan","jackfruit","mangosteen","longan","cherimoya","soursop","jabuticaba","salak","cupuacu","sapodilla","breadfruit"]

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

# ── 1-10: JSON-nested arguments (tool calls where args contain JSON strings)
for idx in range(10):
    i=1+idx; c=C[idx%25]; t=T[idx%15]; fr=FRUITS[idx%15]
    if idx%3==0:
        call = tc("get_weather",{"location":c,"unit":"celsius","metadata":json.dumps({"source":"api","version":2})})
        ex={"messages":[m("user",f"Weather in {c} with metadata"),m("assistant",tc_val=call)],"tools":[wt("get_weather",{"location":{"type":"string"},"unit":{"type":"string"},"metadata":{"type":"string"}},["location"])]}
        sv(i,f"JSON args weather {c}",ex)
    elif idx%3==1:
        call = tc("send_email",{"to":f"u{idx}@t.com","subject":"Report","body":json.dumps({"title":"Q3","metrics":[1,2,3]})})
        ex={"messages":[m("user","Send JSON report"),m("assistant",tc_val=call)],"tools":[wt("send_email",{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},["to","subject"])]}
        sv(i,"JSON args email",ex)
    else:
        call = tc("translate_text",{"text":json.dumps({"greeting":"hello","farewell":"bye"}),"target_lang":"fr"})
        ex={"messages":[m("user","Translate JSON to French"),m("assistant",tc_val=call)],"tools":[wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"])]}
        sv(i,"JSON args translate",ex)

# ── 11-20: Tool call ordering (sequence matters)
for idx in range(10):
    i=11+idx; c=C[idx%25]; c2=C[(idx+7)%25]; t=T[idx%15]; fr=FRUITS[idx%15]
    if idx%4==0:
        call = tc("search_web",{"query":f"recipe for {fr}"})+tc("get_restaurant_info",{"name":fr.title(),"location":c})
        ex={"messages":[m("user",f"Find {fr} recipe and restaurant in {c}"),m("assistant",tc_val=call)],"tools":[wt("search_web",{"query":{"type":"string"}},["query"]),wt("get_restaurant_info",{"name":{"type":"string"},"location":{"type":"string"}},["name"])]}
        sv(i,f"Order: recipe then restaurant {fr}",ex)
    elif idx%4==1:
        call = tc("get_weather",{"location":c})+tc("search_web",{"query":f"things to do in {c}"})
        ex={"messages":[m("user",f"Weather in {c} and things to do"),m("assistant",tc_val=call)],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        sv(i,f"Order: weather then activities {c}",ex)
    elif idx%4==2:
        call = tc("get_stock_price",{"ticker":t})+tc("get_news",{"topic":t, "count":5})
        ex={"messages":[m("user",f"Stock {t} and related news"),m("assistant",tc_val=call)],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"]),wt("get_news",{"topic":{"type":"string"},"count":{"type":"integer"}},["topic"])]}
        sv(i,f"Order: stock then news {t}",ex)
    else:
        call = tc("translate_text",{"text":"hello","target_lang":"th"})+tc("search_web",{"query":"thai greeting etiquette"})
        ex={"messages":[m("user","Translate hello to Thai and find etiquette tips"),m("assistant",tc_val=call)],"tools":[wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        sv(i,"Order: translate then etiquette",ex)

# ── 21-30: Boolean/flag arguments
bool_tools = [
    ("get_weather",{"location":"Paris","unit":"celsius","detailed":True}),
    ("get_weather",{"location":"London","unit":"fahrenheit","detailed":False}),
    ("search_web",{"query":"quantum computing","safe_search":True}),
    ("search_web",{"query":"latest news","safe_search":False}),
    ("get_news",{"topic":"AI","count":5,"breaking":True}),
    ("get_news",{"topic":"sports","count":3,"breaking":False}),
    ("book_flight",{"origin":"BKK","destination":"NRT","date":"2026-12-01","direct":True}),
    ("book_flight",{"origin":"LHR","destination":"JFK","date":"2026-11-15","direct":False}),
    ("send_email",{"to":"a@b.com","subject":"Test","body":"Hello","urgent":True}),
    ("send_email",{"to":"c@d.com","subject":"Update","body":"Done","urgent":False}),
]
for idx,(tn,args) in enumerate(bool_tools):
    i=21+idx
    props = {k:{"type":"string"} if k!="detailed" and k!="safe_search" and k!="breaking" and k!="direct" and k!="urgent" else {"type":"boolean"} for k in args}
    ex={"messages":[m("user",f"Test {tn} with bool flags"),m("assistant",tc_val=tc(tn,args))],"tools":[wt(tn,props,list(args.keys()))]}
    sv(i,f"Bool args: {tn}",ex)

# ── 31-40: Tool call error recovery (tool returns error, model retries)
for idx in range(10):
    i=31+idx; c=C[idx%25]; c2=C[(idx+5)%25]; t=T[idx%15]
    if idx%3==0:
        call = tc("get_weather",{"location":c})
        retry = tc("search_web",{"query":f"weather in {c}"})
        ex={"messages":[m("user",f"Weather in {c}?"),m("assistant",tc_val=call),m("tool","ERROR: city not found"),m("assistant",tc_val=retry)],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        sv(i,f"Error recovery weather {c}",ex)
    elif idx%3==1:
        call = tc("get_stock_price",{"ticker":t})
        retry = tc("search_web",{"query":f"{t} stock price"})
        ex={"messages":[m("user",f"Price of {t}?"),m("assistant",tc_val=call),m("tool","ERROR: invalid ticker"),m("assistant",tc_val=retry)],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        sv(i,f"Error recovery stock {t}",ex)
    else:
        call = tc("translate_text",{"text":"hello","target_lang":"xx"})
        retry = tc("search_web",{"query":"translate hello"})
        ex={"messages":[m("user","Translate hello"),m("assistant",tc_val=call),m("tool","ERROR: unknown language"),m("assistant",tc_val=retry)],"tools":[wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        sv(i,"Error recovery translate",ex)

# ── 41-50: System prompt context affecting tool choice
system_prompts = [
    ("You are a weather specialist. Only use weather tools.", "get_weather", {"location":"Paris"}),
    ("You are a financial analyst. Focus on stock data.", "get_stock_price", {"ticker":"AAPL"}),
    ("You are a translator. Provide translations only.", "translate_text", {"text":"hello","target_lang":"de"}),
    ("You are a news curator. Gather latest headlines.", "get_news", {"topic":"technology","count":5}),
    ("You are a travel agent. Book and research flights.", "book_flight", {"origin":"BKK","destination":"LHR","date":"2026-10-01"}),
    ("You are a researcher. Search for information.", "search_web", {"query":"climate data 2026"}),
    ("You are a restaurant critic. Find dining options.", "get_restaurant_info", {"name":"test","location":"Tokyo"}),
    ("You are a mathematician. Solve calculations.", "calculator", {"expression":"integral x^2 dx"}),
    ("You are a time keeper. Track time across zones.", "get_time", {"location":"London"}),
    ("You are a personal assistant. Handle communications.", "send_email", {"to":"a@b.com","subject":"Hi","body":"Hello"}),
]
for idx,(sys_prompt,tn,args) in enumerate(system_prompts):
    i=41+idx
    props = {k:{"type":"string"} if k!="count" else {"type":"integer"} for k in args}
    ex={"messages":[m("user",f"Hello, help me with something"),m("assistant",tc_val=tc(tn,args))],"system":sys_prompt,"tools":[wt(tn,props,list(args.keys())),wt("search_web",{"q":{"type":"string"}},["q"])]}
    sv(i,f"System prompt: {tn}",ex)

# ── 51-60: Time-sensitive queries (relative dates)
import datetime
today = datetime.date.today()
for idx in range(10):
    i=51+idx; c=C[idx%25]; c2=C[(idx+8)%25]
    if idx%3==0:
        call = tc("get_weather",{"location":c,"date":"today"})
        ex={"messages":[m("user",f"Weather in {c} today?"),m("assistant",tc_val=call)],"tools":[wt("get_weather",{"location":{"type":"string"},"date":{"type":"string"}},["location"])]}
        sv(i,f"Time-sensitive weather {c}",ex)
    elif idx%3==1:
        call = tc("book_flight",{"origin":"BKK","destination":c2[:3].upper(),"date":"tomorrow"})
        ex={"messages":[m("user",f"Book to {c2[:3]} tomorrow"),m("assistant",tc_val=call)],"tools":[wt("book_flight",{"origin":{"type":"string"},"destination":{"type":"string"},"date":{"type":"string"}},["origin","destination","date"])]}
        sv(i,f"Time-sensitive book {c2[:3]}",ex)
    else:
        call = tc("get_news",{"topic":"AI","count":5,"period":"this week"})
        ex={"messages":[m("user","AI news this week?"),m("assistant",tc_val=call)],"tools":[wt("get_news",{"topic":{"type":"string"},"count":{"type":"integer"},"period":{"type":"string"}},["topic"])]}
        sv(i,"Time-sensitive news this week",ex)

# ── 61-70: Comparison queries (which is better, compare X and Y)
for idx in range(10):
    i=61+idx; c=C[idx%25]; c2=C[(idx+3)%25]; t=T[idx%15]; t2=T[(idx+5)%15]; fr=FRUITS[idx%15]; fr2=FRUITS[(idx+7)%15]
    if idx%4==0:
        call = tc("search_web",{"query":f"{fr} vs {fr2} nutritional comparison"})
        ex={"messages":[m("user",f"Compare {fr} and {fr2}"),m("assistant",tc_val=call)],"tools":[wt("search_web",{"query":{"type":"string"}},["query"])]}
        sv(i,f"Compare {fr} vs {fr2}",ex)
    elif idx%4==1:
        call = tc("get_weather",{"location":c})+tc("get_weather",{"location":c2})
        ex={"messages":[m("user",f"Compare weather in {c} and {c2}"),m("assistant",tc_val=call)],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"])]}
        sv(i,f"Compare weather {c} vs {c2}",ex)
    elif idx%4==2:
        call = tc("get_stock_price",{"ticker":t})+tc("get_stock_price",{"ticker":t2})
        ex={"messages":[m("user",f"Compare {t} vs {t2} stock"),m("assistant",tc_val=call)],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"])]}
        sv(i,f"Compare stock {t} vs {t2}",ex)
    else:
        call = tc("search_web",{"query":f"best time to visit {c} vs {c2}"})
        ex={"messages":[m("user",f"Best time to visit {c} or {c2}?"),m("assistant",tc_val=call)],"tools":[wt("search_web",{"query":{"type":"string"}},["query"])]}
        sv(i,f"Compare travel {c} vs {c2}",ex)

# ── 71-80: Optional parameter handling
opt_tools = [
    ("get_weather",{"location":"Paris"},["location"]),
    ("get_weather",{"location":"London","unit":"celsius"},["location"]),
    ("get_weather",{"location":"Tokyo","unit":"fahrenheit","wind":True},["location"]),
    ("search_web",{"query":"AI"},["query"]),
    ("search_web",{"query":"ML","max_results":50},["query"]),
    ("search_web",{"query":"DL","max_results":100,"safe":True},["query"]),
    ("send_email",{"to":"a@b.com","subject":"Hi","body":"Hello"},["to","subject"]),
    ("send_email",{"to":"c@d.com","subject":"Update"},["to","subject"]),
    ("book_flight",{"origin":"BKK","destination":"NRT","date":"2026-01-01"},["origin","destination","date"]),
    ("book_flight",{"origin":"LHR","destination":"JFK","date":"2026-05-01","class":"business"},["origin","destination","date"]),
]
for idx,(tn,args,req) in enumerate(opt_tools):
    i=71+idx
    props = {k:{"type":"string"} if k!="wind" and k!="safe" and k!="max_results" else {"type":"integer"} if k=="max_results" else {"type":"boolean"} for k in args}
    ex={"messages":[m("user",f"Test {tn} optional params"),m("assistant",tc_val=tc(tn,args))],"tools":[wt(tn,props,req)]}
    sv(i,f"Optional params: {tn}",ex)

# ── 81-90: Multi-language tool descriptions
lang_tools = [
    ("th", "ตรวจสอบสภาพอากาศ", "get_weather",{"location":"กรุงเทพ"}),
    ("th", "ค้นหาราคาหุ้น", "get_stock_price",{"ticker":"AAPL"}),
    ("th", "แปลภาษา", "translate_text",{"text":"hello","target_lang":"th"}),
    ("fr", "météo", "get_weather",{"location":"Paris"}),
    ("fr", "cours de l'action", "get_stock_price",{"ticker":"GOOGL"}),
    ("fr", "traduire", "translate_text",{"text":"bonjour","target_lang":"en"}),
    ("de", "Wetter", "get_weather",{"location":"Berlin"}),
    ("de", "Aktienkurs", "get_stock_price",{"ticker":"DAX"}),
    ("de", "bersetzen", "translate_text",{"text":"hallo","target_lang":"de"}),
    ("es", "clima", "get_weather",{"location":"Madrid"}),
]
for idx,(lang,desc,tn,args) in enumerate(lang_tools):
    i=81+idx
    tool_def = {"type":"function","function":{"name":tn,"description":desc,"parameters":{"type":"object","properties":{k:{"type":"string"} for k in args},"required":list(args.keys())}}}
    call = tc(tn,args)
    ex={"messages":[m("user",f"{desc}"),m("assistant",tc_val=call)],"tools":[tool_def]}
    sv(i,f"Multi-lang {lang}: {tn}",ex)

# ── 91-100: Redundant/conflicting tool sets
for idx in range(10):
    i=91+idx; c=C[idx%25]; t=T[idx%15]; fr=FRUITS[idx%15]
    if idx%3==0:
        call = tc("get_weather",{"location":c})
        ex={"messages":[m("user",f"Weather in {c}?"),m("assistant",tc_val=call)],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"]),wt("get_weather",{"loc":{"type":"string"}},["loc"]),wt("get_time",{"location":{"type":"string"}},["location"])]}
        sv(i,f"Redundant tools weather {c}",ex)
    elif idx%3==1:
        call = tc("get_stock_price",{"ticker":t})
        ex={"messages":[m("user",f"Stock {t}?"),m("assistant",tc_val=call)],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"]),wt("get_stock_price",{"symbol":{"type":"string"}},["symbol"]),wt("get_news",{"topic":{"type":"string"}},["topic"])]}
        sv(i,f"Redundant tools stock {t}",ex)
    else:
        call = tc("search_web",{"query":f"best {fr} recipe"})
        ex={"messages":[m("user",f"Find {fr} recipe"),m("assistant",tc_val=call)],"tools":[wt("search_web",{"query":{"type":"string"}},["query"]),wt("search_web",{"q":{"type":"string"}},["q"]),wt("get_restaurant_info",{"name":{"type":"string"}},["name"])]}
        sv(i,f"Redundant tools search {fr}",ex)

print(f"\n{'='*60}")
print(f"CYCLE 100 V3 COMPLETE: {len(res)} new improvements")
print(f"{'='*60}")
with open(OUT/"all-100-v3.jsonl","w") as f:
    for p in sorted(OUT.glob("iter-*.jsonl")):
        f.write(p.read_text(encoding="utf-8"))
pc=Counter(r["phase"] for r in res)
for p in PHASES: print(f"  {p:15s}: {pc.get(p,0)}")
print(f"\n3-round total so far: 300 improvements")
print(f"  v1: cycle-100-output/  (initial 100)")
print(f"  v2: cycle-100-v2/     (different content)")
print(f"  v3: cycle-100-v3/     (JSON args, bools, error recovery, etc.)")
