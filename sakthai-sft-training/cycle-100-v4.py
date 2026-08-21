#!/usr/bin/env python3
"""100 more — fourth round, 10 new categories never used before."""
import json, random
from pathlib import Path
from collections import Counter
random.seed(7777)
OUT = Path("cycle-100-v4"); OUT.mkdir(exist_ok=True)
PHASES = ["DATA","TRAIN","EVAL","ERROR","PUBLISH","DEPLOY","DISCOVER","MONITOR","PROFILE","ITERATE"]
C = ["Hiroshima","Nagoya","Fukuoka","Sapporo","Kobe","Kyoto","Kawasaki","Yokohama","Sendai","Okayama","Helsinki","Turku","Tampere","Oulu","Rovaniemi","Zurich","Geneva","Basel","Bern","Lugano","Riga","Kaunas","Tartu","Belgrade","Zagreb"]
T = ["AFRM","BILL","UPST","SOFI","PYPL","SQ","COIN","MELI","SE","SHOP","TWLO","SEND","DOCN","GTLB","FRSH"]
TO = ["photonic computing","neuromorphic chips","spintronics","topological qubits","DNA storage","bioacoustics","echolocation","electroreception","bioluminescence","cryoturbation","thermokarst","palsa","solifluction","nivation","glacial rebound"]
L = ["jv","su","rw","ny","nso","st","mg","ts","ve","sn","sg","ht","bm","gn","kg"]
PLANETS = ["Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto","Ceres","Eris","Makemake","Haumea","Titan","Europa","Ganymede"]

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

# ── 1-10: Tools with array parameters
for idx in range(10):
    i=1+idx; c=C[idx%25]; t=T[idx%15]; pl=PLANETS[idx%15]
    if idx%3==0:
        call = tc("get_weather",{"locations":[c, C[(idx+5)%25]],"unit":"celsius"})
        ex={"messages":[m("user",f"Weather in multiple cities"),m("assistant",tc_val=call)],"tools":[wt("get_weather",{"locations":{"type":"array","items":{"type":"string"}},"unit":{"type":"string"}},["locations"])]}
        sv(i,f"Array params weather {c}",ex)
    elif idx%3==1:
        call = tc("get_stock_price",{"tickers":[t, T[(idx+5)%15]],"currency":"USD"})
        ex={"messages":[m("user",f"Multiple stock prices"),m("assistant",tc_val=call)],"tools":[wt("get_stock_price",{"tickers":{"type":"array","items":{"type":"string"}},"currency":{"type":"string"}},["tickers"])]}
        sv(i,f"Array params stocks",ex)
    else:
        call = tc("send_email",{"to":["a@b.com","c@d.com"],"subject":"Group","body":"Hello all"})
        ex={"messages":[m("user","Email multiple people"),m("assistant",tc_val=call)],"tools":[wt("send_email",{"to":{"type":"array","items":{"type":"string"}},"subject":{"type":"string"},"body":{"type":"string"}},["to","subject"])]}
        sv(i,"Array params email group",ex)

# ── 11-20: Tools with no arguments (zero-param calls)
for idx in range(10):
    i=11+idx
    if idx%4==0:
        call = tc("get_time",{})
        ex={"messages":[m("user","What time is it?"),m("assistant",tc_val=call)],"tools":[wt("get_time",{},[])]}
        sv(i,"Zero-param get_time",ex)
    elif idx%4==1:
        call = tc("get_date",{})
        ex={"messages":[m("user","What's today's date?"),m("assistant",tc_val=call)],"tools":[wt("get_date",{},[])]}
        sv(i,"Zero-param get_date",ex)
    elif idx%4==2:
        call = tc("get_random_number",{})
        ex={"messages":[m("user","Give me a random number"),m("assistant",tc_val=call)],"tools":[wt("get_random_number",{},[])]}
        sv(i,"Zero-param random",ex)
    else:
        call = tc("get_status",{})
        ex={"messages":[m("user","System status?"),m("assistant",tc_val=call)],"tools":[wt("get_status",{},[])]}
        sv(i,"Zero-param status",ex)

# ── 21-30: Tools with enum constraints
enum_tools = [
    ("get_weather",{"location":"Paris","unit":"celsius","wind_scale":"beaufort"},{"unit":{"type":"string","enum":["celsius","fahrenheit"]},"wind_scale":{"type":"string","enum":["beaufort","ms","knots"]}}),
    ("get_weather",{"location":"London","unit":"fahrenheit","wind_scale":"knots"},{"unit":{"type":"string","enum":["celsius","fahrenheit"]},"wind_scale":{"type":"string","enum":["beaufort","ms","knots"]}}),
    ("book_flight",{"origin":"BKK","destination":"NRT","date":"2026-10-01","class":"economy"},{"class":{"type":"string","enum":["economy","business","first"]}}),
    ("book_flight",{"origin":"LHR","destination":"JFK","date":"2026-11-01","class":"business"},{"class":{"type":"string","enum":["economy","business","first"]}}),
    ("get_directions",{"origin":"A","destination":"B","mode":"driving"},{"mode":{"type":"string","enum":["driving","walking","transit","bicycling"]}}),
    ("get_directions",{"origin":"C","destination":"D","mode":"bicycling"},{"mode":{"type":"string","enum":["driving","walking","transit","bicycling"]}}),
    ("send_email",{"to":"a@b.com","subject":"Hi","body":"Hello","priority":"high"},{"priority":{"type":"string","enum":["low","normal","high","urgent"]}}),
    ("send_email",{"to":"c@d.com","subject":"Test","body":"World","priority":"low"},{"priority":{"type":"string","enum":["low","normal","high","urgent"]}}),
    ("get_news",{"topic":"AI","count":5,"sort_by":"relevance"},{"sort_by":{"type":"string","enum":["relevance","date","popularity"]}}),
    ("get_news",{"topic":"sports","count":3,"sort_by":"date"},{"sort_by":{"type":"string","enum":["relevance","date","popularity"]}}),
]
for idx,(tn,args,extra) in enumerate(enum_tools):
    i=21+idx
    props = {k:{"type":"string"} if k not in extra else extra[k] for k in args}
    ex={"messages":[m("user",f"Enum test {tn}"),m("assistant",tc_val=tc(tn,args))],"tools":[wt(tn,props,list(args.keys()))]}
    sv(i,f"Enum constraints {tn}",ex)

# ── 31-40: Unicode/special chars in arguments
unicode_args = [
    ("get_weather",{"location":"München","unit":"celsius"}),
    ("get_weather",{"location":"São Paulo","unit":"celsius"}),
    ("get_restaurant_info",{"name":"Café Zürich","location":"Zürich"}),
    ("get_restaurant_info",{"name":"Döner Kebab","location":"Köln"}),
    ("translate_text",{"text":"café au lait","target_lang":"en"}),
    ("translate_text",{"text":"piñata","target_lang":"es"}),
    ("send_email",{"to":"usér@tést.com","subject":"Méil","body":"Cöol"}),
    ("search_web",{"query":"naïve bayes algorithm"}),
    ("get_news",{"topic":"façade engineering","count":5}),
    ("book_flight",{"origin":"Ål","destination":"Øl","date":"2026-10-01"}),
]
for idx,(tn,args) in enumerate(unicode_args):
    i=31+idx
    props = {k:{"type":"string"} if k!="count" else {"type":"integer"} for k in args}
    ex={"messages":[m("user",f"Unicode test {tn}"),m("assistant",tc_val=tc(tn,args))],"tools":[wt(tn,props,list(args.keys()))]}
    sv(i,f"Unicode args {tn}",ex)

# ── 41-50: Context from previous assistant responses
for idx in range(10):
    i=41+idx; c=C[idx%25]; c2=C[(idx+5)%25]; t=T[idx%15]; pl=PLANETS[idx%15]; pl2=PLANETS[(idx+7)%15]
    if idx%4==0:
        prev = tc("get_weather",{"location":c})
        ex={"messages":[m("user",f"Weather in {c}?"),m("assistant",tc_val=prev),m("tool","Sunny 25C"),m("user","What about humidity?"),m("assistant",content="Let me check the humidity levels.")],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"])]}
        sv(i,f"Assistant context weather {c}",ex)
    elif idx%4==1:
        prev = tc("get_stock_price",{"ticker":t})
        ex={"messages":[m("user",f"{t} price?"),m("assistant",tc_val=prev),m("tool","$450"),m("user","P/E ratio?"),m("assistant",content="The P/E ratio helps evaluate valuation.")],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"])]}
        sv(i,f"Assistant context stock {t}",ex)
    elif idx%4==2:
        prev = tc("search_web",{"query":f"{pl} exploration"})
        ex={"messages":[m("user",f"Tell me about {pl}"),m("assistant",tc_val=prev),m("tool","Result: NASA..."),m("user",f"What about {pl2}?"),m("assistant",content=f"Let me compare {pl} and {pl2}.")],"tools":[wt("search_web",{"query":{"type":"string"}},["query"])]}
        sv(i,f"Assistant context planets",ex)
    else:
        prev = tc("translate_text",{"text":"hello","target_lang":"fr"})
        ex={"messages":[m("user","Translate hello"),m("assistant",tc_val=prev),m("tool","bonjour"),m("user","Pronunciation?"),m("assistant",content="Bonjour is pronounced with a soft 'j'.")],"tools":[wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"])]}
        sv(i,"Assistant context translate",ex)

# ── 51-60: Domain-specific terminology (medical, legal, finance)
domain_args = [
    ("search_web",{"query":"myocardial infarction treatment guidelines 2026"}),
    ("get_news",{"topic":"FDA drug approvals","count":5}),
    ("search_web",{"query":"habeas corpus precedent cases"}),
    ("search_web",{"query":"quantitative easing inflation impact"}),
    ("get_news",{"topic":"GDP growth projections","count":3}),
    ("search_web",{"query":"CRISPR-Cas9 ethical considerations"}),
    ("get_news",{"topic":"mergers and acquisitions","count":5}),
    ("search_web",{"query":"supreme court docket 2026 term"}),
    ("search_web",{"query":"machine learning operations MLOps best practices"}),
    ("get_news",{"topic":"central bank interest rates","count":5}),
]
for idx,(tn,args) in enumerate(domain_args):
    i=51+idx
    props = {k:{"type":"string"} if k!="count" else {"type":"integer"} for k in args}
    ex={"messages":[m("user",f"Domain query #{idx+1}"),m("assistant",tc_val=tc(tn,args))],"tools":[wt(tn,props,list(args.keys()))]}
    sv(i,f"Domain: {list(args.values())[0][:30]}",ex)

# ── 61-70: Negative number/decimal handling
num_args = [
    ("get_weather",{"location":"Moscow","unit":"celsius","temperature":-15}),
    ("get_weather",{"location":"Yakutsk","unit":"celsius","temperature":-40}),
    ("get_weather",{"location":"Antarctica","unit":"fahrenheit","temperature":-20}),
    ("get_stock_price",{"ticker":"AAPL","change":-3.5}),
    ("get_stock_price",{"ticker":"TSLA","change":-8.2}),
    ("get_stock_price",{"ticker":"MSFT","change":1.25}),
    ("calculator",{"expression":"-15 + -20 + -30"}),
    ("calculator",{"expression":"0.1 + 0.2"}),
    ("calculator",{"expression":"-5 * -5"}),
    ("calculator",{"expression":"100 / -3"}),
]
for idx,(tn,args) in enumerate(num_args):
    i=61+idx
    props = {}
    for k in args:
        if isinstance(args[k], bool): props[k] = {"type":"boolean"}
        elif isinstance(args[k], int): props[k] = {"type":"integer"}
        elif isinstance(args[k], float): props[k] = {"type":"number"}
        else: props[k] = {"type":"string"}
    ex={"messages":[m("user",f"Number test #{idx+1}"),m("assistant",tc_val=tc(tn,args))],"tools":[wt(tn,props,list(args.keys()))]}
    sv(i,f"Neg/dec: {list(args.values())[0]}",ex)

# ── 71-80: Template pattern variation (same query, different params)
cities_pool = C[:5]
for idx in range(10):
    i=71+idx; c1=cities_pool[idx%5]; c2=cities_pool[(idx+2)%5]; c3=cities_pool[(idx+4)%5]
    if idx%3==0:
        call = tc("get_weather",{"location":c1})
        ex={"messages":[m("user",f"What's the weather in {c1} like?"),m("assistant",tc_val=call)],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"])]}
        sv(i,f"Template: weather {c1}",ex)
    elif idx%3==1:
        call = tc("get_weather",{"location":c2})
        ex={"messages":[m("user",f"How is the weather in {c2}?"),m("assistant",tc_val=call)],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"])]}
        sv(i,f"Template: weather {c2}",ex)
    else:
        call = tc("get_weather",{"location":c3})
        ex={"messages":[m("user",f"Can you check the weather in {c3}?"),m("assistant",tc_val=call)],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"])]}
        sv(i,f"Template: weather {c3}",ex)

# ── 81-90: Cross-tool dependencies (output of A is input for B)
for idx in range(10):
    i=81+idx; c=C[idx%25]; t=T[idx%15]
    if idx%3==0:
        s1 = tc("search_web",{"query":f"coordinates of {c}"})
        s2 = tc("get_weather",{"location":c})
        ex={"messages":[m("user",f"Weather at location {c} coordinates"),m("assistant",tc_val=s1+s2)],"tools":[wt("search_web",{"query":{"type":"string"}},["query"]),wt("get_weather",{"location":{"type":"string"}},["location"])]}
        sv(i,f"Cross-dep weather {c}",ex)
    elif idx%3==1:
        s1 = tc("get_stock_price",{"ticker":t})
        s2 = tc("search_web",{"query":f"{t} earnings report 2026"})
        ex={"messages":[m("user",f"Research {t} stock and earnings"),m("assistant",tc_val=s1+s2)],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        sv(i,f"Cross-dep stock {t}",ex)
    else:
        s1 = tc("translate_text",{"text":"thank you","target_lang":"ja"})
        s2 = tc("search_web",{"query":"japanese thank you etiquette"})
        ex={"messages":[m("user","Translate and find etiquette"),m("assistant",tc_val=s1+s2)],"tools":[wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        sv(i,"Cross-dep translate+etiquette",ex)

# ── 91-100: Tool description with embedded examples (few-shot in description)
few_shot_tools = [
    wt("get_weather",{"location":{"type":"string","description":"City name e.g. Paris, London, Tokyo"}},["location"]),
    wt("get_weather",{"location":{"type":"string","description":"City name like Bangkok, Berlin, Rome"}},["location"]),
    wt("get_stock_price",{"ticker":{"type":"string","description":"Stock symbol e.g. AAPL, GOOGL, MSFT"}},["ticker"]),
    wt("get_stock_price",{"ticker":{"type":"string","description":"Ticker symbol like TSLA, NVDA, AMD"}},["ticker"]),
    wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string","description":"Language code: th, fr, de, es, ja"}},["text","target_lang"]),
    wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string","description":"Target language: Thai=th, French=fr, German=de"}},["text","target_lang"]),
    wt("book_flight",{"origin":{"type":"string","description":"Airport code: BKK, NRT, LHR, JFK"}, "destination":{"type":"string"},"date":{"type":"string"}},["origin","destination","date"]),
    wt("book_flight",{"origin":{"type":"string"},"destination":{"type":"string","description":"Airport code e.g. LAX, CDG, DXB, SIN"},"date":{"type":"string"}},["origin","destination","date"]),
    wt("search_web",{"query":{"type":"string","description":"Search term, e.g. 'latest AI news'"}},["query"]),
    wt("get_news",{"topic":{"type":"string","description":"Topic like technology, sports, health"},"count":{"type":"integer"}},["topic"]),
]
for idx,tool_def in enumerate(few_shot_tools):
    i=91+idx; tn=tool_def["function"]["name"]
    if "location" in tool_def["function"]["parameters"].get("properties",{}):
        call = tc(tn,{"location":C[idx%25]})
    elif "ticker" in tool_def["function"]["parameters"].get("properties",{}):
        call = tc(tn,{"ticker":T[idx%15]})
    elif "query" in tool_def["function"]["parameters"].get("properties",{}):
        call = tc(tn,{"query":"test"})
    elif "text" in tool_def["function"]["parameters"].get("properties",{}):
        call = tc(tn,{"text":"hello","target_lang":"fr"})
    elif "origin" in tool_def["function"]["parameters"].get("properties",{}):
        call = tc(tn,{"origin":"BKK","destination":"NRT","date":"2026-10-01"})
    elif "topic" in tool_def["function"]["parameters"].get("properties",{}):
        call = tc(tn,{"topic":"AI","count":5})
    else:
        call = tc(tn,{})
    ex={"messages":[m("user",f"Few-shot tool test"),m("assistant",tc_val=call)],"tools":[tool_def]}
    sv(i,f"Few-shot desc: {tn}",ex)

print(f"\n{'='*60}")
print(f"CYCLE 100 V4 COMPLETE: {len(res)} new improvements")
print(f"{'='*60}")
with open(OUT/"all-100-v4.jsonl","w") as f:
    for p in sorted(OUT.glob("iter-*.jsonl")):
        f.write(p.read_text(encoding="utf-8"))
pc=Counter(r["phase"] for r in res)
for p in PHASES: print(f"  {p:15s}: {pc.get(p,0)}")
print(f"\n4-round total: 400 improvements")
print(f"  v1: single/parallel/irrelevance/multiturn/hardneg/multihop/coercion/heldout/safety")
print(f"  v2: negative_q/int_coercion/multistep/empty_args/tricky_irr/format/3way/long/deep/boundary")
print(f"  v3: json_args/ordering/bools/error_recovery/sysprompt/timesensitive/comparison/optional/multilang/redundant")
print(f"  v4: arrays/zeroparam/enums/unicode/context/domain/negdecimal/templates/crossdep/fewshot")
