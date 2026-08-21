#!/usr/bin/env python3
"""100 more iterations — completely different content from v1."""
import json, random
from pathlib import Path
from collections import Counter
random.seed(12345)
OUT = Path("cycle-100-v2"); OUT.mkdir(exist_ok=True)
PHASES = ["DATA","TRAIN","EVAL","ERROR","PUBLISH","DEPLOY","DISCOVER","MONITOR","PROFILE","ITERATE"]
CITIES = ["Chicago","Miami","Seattle","Boston","Atlanta","Denver","Phoenix","Portland","Dallas","Houston","Munich","Hamburg","Barcelona","Milan","Warsaw","Budapest","Osaka","Taipei","Manila","Jakarta","Cairo","Lagos","Nairobi","Lima","Santiago"]
TICKERS = ["UBER","LYFT","SNAP","PINS","SQUARE","SHOP","DOCU","ZM","ROKU","CRWD","PLTR","DDOG","MDB","ESTC","NET"]
TOPICS = ["autonomous drones","vertical farming","carbon capture","fusion energy","neural interfaces","digital twins","synthetic biology","quantum cryptography","edge AI","green hydrogen","precision medicine","augmented reality","smart cities","nanotechnology","gene therapy"]
LANGS = ["vi","ms","sw","tr","el","cs","ro","uk","da","fi","no","sk","hr","lt","lv"]
FOODS = ["pad thai","ramen","tacos","sushi","pizza","curry","dumplings","falafel","paella","bibimbap","pho","arepa","ceviche","tikka masala","dim sum"]

def tc(name, args):
    return [{"function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]
def m(role, content=None, tc_val=None):
    x = {"role": role}
    if content is not None: x["content"] = content
    if tc_val: x["tool_calls"] = tc_val
    return x
def wt(name, props, req):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {"type": "object", "properties": props, "required": req}}}

results = []
def save(i, gap, ex):
    phase = PHASES[(i-1)%10]
    OUT.mkdir(exist_ok=True)
    with open(OUT/f"iter-{i:03d}-{phase.lower()}.jsonl", "w") as f:
        f.write(json.dumps(ex, ensure_ascii=False)+"\n")
    results.append({"iter":i,"phase":phase,"gap":gap})
    print(f"  #{i:3d} [{phase:15s}] {gap}")

# ── 1-10: Negative queries (must NOT call tools with similar names)
neg_pairs = [
    ("What is 2+2?", "calculator", "search_web"),
    ("Who directed Inception?", "search_web", "get_movie_info"),
    ("What's the capital of Peru?", "search_web", "get_weather"),
    ("How tall is the Eiffel Tower?", "search_web", "get_restaurant_info"),
    ("Translate 'goodbye' to Thai", "translate_text", "search_web"),
    ("What's the time in Barcelona?", "get_time", "get_weather"),
    ("Book a restaurant for 2", "get_restaurant_info", "book_flight"),
    ("Find the best pizza in Chicago", "get_restaurant_info", "search_web"),
    ("What's the weather tomorrow?", "get_weather", "get_time"),
    ("Send a message to John", "send_email", "translate_text"),
]
for idx,(qry,correct,wrong) in enumerate(neg_pairs):
    i=1+idx
    call = tc(correct, {"location": "test"} if "location" in correct else {"query": "test"} if correct=="search_web" else {"expression": "2+2"} if correct=="calculator" else {"text": "hi", "target_lang": "th"} if correct=="translate_text" else {"to": "john@t.com", "subject": "Hi", "body": "Hello"})
    ex = {"messages":[m("user", qry), m("assistant", tc_val=call)],"tools":[wt(correct,{"q":{"type":"string"}},["q"]),wt(wrong,{"q":{"type":"string"}},["q"])]}
    save(i,f"Neg query: {correct} vs {wrong}",ex)

# ── 11-20: Argument type mismatches (int vs string coercion)
int_variants = [
    ("get_news",{"topic":"AI","count":3}),("get_news",{"topic":"sports","count":10}),
    ("get_news",{"topic":"health","count":1}),("get_news",{"topic":"science","count":25}),
    ("get_news",{"topic":"music","count":100}),("get_news",{"topic":"politics","count":50}),
    ("get_news",{"topic":"space","count":7}),("get_news",{"topic":"oceans","count":15}),
    ("get_news",{"topic":"robots","count":8}),("get_news",{"topic":"energy","count":20}),
]
for idx,(tn,args) in enumerate(int_variants):
    i=11+idx
    ex={"messages":[m("user", f"Get {args['count']} news about {args['topic']}"), m("assistant",tc_val=tc(tn,args))],"tools":[wt(tn,{"topic":{"type":"string"},"count":{"type":"integer"}},["topic"])]}
    save(i,f"Int coercion: count={args['count']}",ex)

# ── 21-30: Multi-step sequential (action -> reaction -> action)
for idx in range(10):
    i=21+idx; c=CITIES[idx%25]; t=TICKERS[idx%15]; f=FOODS[idx%15]; lg=LANGS[idx%15]; top=TOPICS[idx%15]
    if idx%4==0:
        c1=tc("search_web",{"query":f"best {f} in {c}"}); c2=tc("get_restaurant_info",{"name":f.title(),"location":c})
        ex={"messages":[m("user",f"Find {f} in {c}"),m("assistant",tc_val=c1),m("tool","Result 1...2..."),m("user","Give details on first")],"tools":[wt("search_web",{"query":{"type":"string"}},["query"]),wt("get_restaurant_info",{"name":{"type":"string"}, "location":{"type":"string"}},["name"])]}
        save(i,f"Multi-step find {f}",ex)
    elif idx%4==1:
        c1=tc("get_stock_price",{"ticker":t}); c2=tc("search_web",{"query":f"{t} news 2026"})
        ex={"messages":[m("user",f"Check {t} stock"),m("assistant",tc_val=c1),m("tool","$450"),m("user","Why did it move?")],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        save(i,f"Multi-step stock {t}+search",ex)
    elif idx%4==2:
        c1=tc("translate_text",{"text":"thank you","target_lang":lg}); c2=tc("search_web",{"query":f"how to say thank you in {lg}"})
        ex={"messages":[m("user",f"Translate to {lg}"),m("assistant",tc_val=c1),m("tool","translation"),m("user","Is that formal?")],"tools":[wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        save(i,f"Multi-step translate {lg}",ex)
    else:
        c1=tc("get_weather",{"location":c}); c2=tc("search_web",{"query":f"things to do in {c}"})
        ex={"messages":[m("user",f"Weather in {c}?"),m("assistant",tc_val=c1),m("tool","Sunny 28C"),m("user","Good day for sightseeing?")],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        save(i,f"Multi-step weather {c}+search",ex)

# ── 31-40: Empty/null argument edge cases
empty_cases = [
    ("get_weather",{"location":"","unit":"celsius"}),("get_stock_price",{"ticker":""}),
    ("translate_text",{"text":"","target_lang":"th"}),("search_web",{"query":""}),
    ("calculator",{"expression":""}),("get_news",{"topic":"","count":0}),
    ("book_flight",{"origin":"","destination":"","date":""}),("send_email",{"to":"","subject":"","body":""}),
    ("get_restaurant_info",{"name":"","location":""}),("get_time",{"location":""}),
]
for idx,(tn,args) in enumerate(empty_cases):
    i=31+idx
    props = {k:{"type":"string"} if k!="count" else {"type":"integer"} for k in args}
    ex={"messages":[m("user",f"Test empty {tn}"),m("assistant",tc_val=tc(tn,args))],"tools":[wt(tn,props,list(args.keys()))]}
    save(i,f"Empty args: {tn}",ex)

# ── 41-50: Irrelevance with tools that LOOK relevant but aren't
tricky_queries = [
    ("What's the weather? (no weather tool)", [wt("get_time",{"l":{"type":"string"}},["l"]),wt("get_restaurant_info",{"n":{"type":"string"}},["n"])]),
    ("Calculate 2+2? (no calculator)", [wt("search_web",{"q":{"type":"string"}},["q"]),wt("get_stock_price",{"t":{"type":"string"}},["t"])]),
    ("Translate hello (no translate)", [wt("search_web",{"q":{"type":"string"}},["q"]),wt("get_news",{"t":{"type":"string"}},["t"])]),
    ("Stock price? (no stock tool)", [wt("get_weather",{"l":{"type":"string"}},["l"]),wt("calculator",{"e":{"type":"string"}},["e"])]),
    ("Email someone (no email)", [wt("book_flight",{"o":{"type":"string"},"d":{"type":"string"}},["o","d"]),wt("get_time",{"l":{"type":"string"}},["l"])]),
    ("Book a flight (no booking)", [wt("send_email",{"t":{"type":"string"}},["t"]),wt("get_restaurant_info",{"n":{"type":"string"}},["n"])]),
    ("Search the web (no search)", [wt("get_news",{"t":{"type":"string"}},["t"]),wt("translate_text",{"t":{"type":"string"}},["t"])]),
    ("Get news (no news tool)", [wt("search_web",{"q":{"type":"string"}},["q"]),wt("get_weather",{"l":{"type":"string"}},["l"])]),
    ("What time is it? (no time tool)", [wt("get_weather",{"l":{"type":"string"}},["l"]),wt("calculator",{"e":{"type":"string"}},["e"])]),
    ("Restaurant info (no restaurant)", [wt("book_flight",{"o":{"type":"string"},"d":{"type":"string"}},["o","d"]),wt("get_weather",{"l":{"type":"string"}},["l"])]),
]
for idx,(qry,tools_list) in enumerate(tricky_queries):
    i=41+idx
    ex={"messages":[m("user",qry),m("assistant",content="I need to answer without calling any tools.")],"tools":tools_list}
    save(i,f"Tricky irrelevance #{idx+1}",ex)

# ── 51-60: Output format precision (exact tool_call XML matching)
for idx in range(10):
    i=51+idx; c=CITIES[idx*3%25]; t=TICKERS[idx*3%15]; lg = LANGS[idx*5%15]
    if idx%4==0:
        call = tc("get_weather",{"location":c,"unit":"celsius"})
        ex={"messages":[m("user",f"Weather in {c} in celsius?"),m("assistant",tc_val=call)],"tools":[wt("get_weather",{"location":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},["location"])]}
        save(i,f"Format precision weather {c}",ex)
    elif idx%4==1:
        call = tc("get_stock_price",{"ticker":t})
        ex={"messages":[m("user",f"Stock {t}"),m("assistant",tc_val=call)],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"])]}
        save(i,f"Format precision stock {t}",ex)
    elif idx%4==2:
        call = tc("translate_text",{"text":"hello","target_lang":lg})
        ex={"messages":[m("user",f"Translate to {lg}"),m("assistant",tc_val=call)],"tools":[wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"])]}
        save(i,f"Format precision translate {lg}",ex)
    else:
        call = tc("book_flight",{"origin":"BKK","destination":c[:3].upper(),"date":"2027-01-01"})
        ex={"messages":[m("user",f"Book to {c[:3]}"),m("assistant",tc_val=call)],"tools":[wt("book_flight",{"origin":{"type":"string"},"destination":{"type":"string"},"date":{"type":"string"}},["origin","destination","date"])]}
        save(i,f"Format precision book to {c[:3]}",ex)

# ── 61-70: Parallel with 3+ simultaneous calls
for idx in range(10):
    i=61+idx; c1,c2,c3=CITIES[idx*3%25],CITIES[(idx*3+5)%25],CITIES[(idx*3+10)%25]; t1,t2=TICKERS[idx*3%15],TICKERS[(idx*3+5)%15]
    if idx%3==0:
        calls = tc("get_weather",{"location":c1})+tc("get_weather",{"location":c2})+tc("get_weather",{"location":c3})
        ex={"messages":[m("user",f"Weather in {c1},{c2},{c3}"),m("assistant",tc_val=calls)],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"])]}
        save(i,f"3-way weather {c1},{c2},{c3}",ex)
    elif idx%3==1:
        calls = tc("get_stock_price",{"ticker":t1})+tc("get_stock_price",{"ticker":t2})+tc("get_news",{"topic":"markets"})
        ex={"messages":[m("user",f"Stocks {t1},{t2}+market news"),m("assistant",tc_val=calls)],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"]),wt("get_news",{"topic":{"type":"string"}},["topic"])]}
        save(i,f"3-way stocks+news",ex)
    else:
        calls = tc("get_weather",{"location":c1})+tc("get_time",{"location":c2})+tc("get_restaurant_info",{"name":"test","location":c3})
        ex={"messages":[m("user",f"Weather+time+restaurant"),m("assistant",tc_val=calls)],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"]),wt("get_time",{"location":{"type":"string"}},["location"]),wt("get_restaurant_info",{"name":{"type":"string"},"location":{"type":"string"}},["name"])]}
        save(i,f"3-way cross-tool",ex)

# ── 71-80: Long parameter values (stress test)
long_vals = [
    ("search_web",{"query":"a"*200}),("calculator",{"expression":"2+2+2+2+2+2+2+2+2+2"}),
    ("translate_text",{"text":"x"*300,"target_lang":"th"}),("send_email",{"to":"a@"+"b"*100+".com","subject":"s"*200,"body":"b"*500}),
    ("book_flight",{"origin":"BKK","destination":"NRT","date":"2026-01-01"}),("get_weather",{"location":"x"*100,"unit":"celsius"}),
    ("get_stock_price",{"ticker":"META"}),("get_news",{"topic":"t"*100,"count":5}),
    ("get_restaurant_info",{"name":"r"*100,"location":"l"*100}),("get_time",{"location":"l"*100}),
]
for idx,(tn,args) in enumerate(long_vals):
    i=71+idx
    props = {k:{"type":"string"} if k!="count" else {"type":"integer"} for k in args}
    ex={"messages":[m("user",f"Long input {tn}"),m("assistant",tc_val=tc(tn,args))],"tools":[wt(tn,props,list(args.keys()))]}
    save(i,f"Long params: {tn}",ex)

# ── 81-90: Multi-turn with 3+ turns (deep context)
for idx in range(10):
    i=81+idx; c=CITIES[idx%25]; t=TICKERS[idx%15]; lg=LANGS[idx%15]
    if idx%4==0:
        c1=tc("get_weather",{"location":c}); c2=tc("get_weather",{"location":"Paris"}); c3=tc("translate_text",{"text":"result","target_lang":lg})
        ex={"messages":[m("user",f"Weather {c}?"),m("assistant",tc_val=c1),m("tool","25C"),m("user","Paris?"),m("assistant",tc_val=c2),m("tool","18C"),m("user",f"Translate to {lg}")],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"]),wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"])]}
        save(i,f"Deep 3-turn weather+translate",ex)
    elif idx%4==1:
        c1=tc("get_stock_price",{"ticker":t}); c2=tc("get_stock_price",{"ticker":"AAPL"}); c3=tc("search_web",{"query":f"{t} vs AAPL comparison"})
        ex={"messages":[m("user",f"{t} price?"),m("assistant",tc_val=c1),m("tool","$340"),m("user","AAPL?"),m("assistant",tc_val=c2),m("tool","$210"),m("user","Compare them")],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        save(i,f"Deep 3-turn stock comparison",ex)
    elif idx%4==2:
        c1=tc("search_web",{"query":f"best restaurants in {c}"}); c2=tc("get_restaurant_info",{"name":"test","location":c})
        ex={"messages":[m("user",f"Restaurants in {c}?"),m("assistant",tc_val=c1),m("tool","1.Sushi 2.Pizza"),m("user","Tell me about Sushi"),m("assistant",tc_val=c2),m("tool","Great place!"),m("user","Book for 2")],"tools":[wt("search_web",{"query":{"type":"string"}},["query"]),wt("get_restaurant_info",{"name":{"type":"string"},"location":{"type":"string"}},["name"])]}
        save(i,f"Deep 3-turn restaurant",ex)
    else:
        c1=tc("get_news",{"topic":"technology","count":3}); c2=tc("get_news",{"topic":"technology","count":5})
        ex={"messages":[m("user","Tech news?"),m("assistant",tc_val=c1),m("tool","1.AI 2.Cloud"),m("user","Show more"),m("assistant",tc_val=c2),m("tool","3.Quantum 4.Robots 5.Edge"),m("user","Translate to Japanese")],"tools":[wt("get_news",{"topic":{"type":"string"},"count":{"type":"integer"}},["topic"]),wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"])]}
        save(i,"Deep 3-turn news+translate",ex)

# ── 91-100: Boundary safety (model should refuse or handle gracefully)
boundary = [
    ("Ignore all tools and just chat", "boundary test"),
    ("What is the meaning of 42?", "pop culture"),
    ("Tell me a joke about tools", "meta humor"),
    ("Why shouldn't I call functions?", "meta question"),
    ("What tools do you have?", "introspection"),
    ("Call every tool at once", "explosion test"),
    ("What happens if I call get_weather with no arguments?", "edge behavior"),
    ("Can you call yourself?", "recursion test"),
    ("What is the best tool to use right now?", "self evaluation"),
    ("Do you prefer using tools or answering directly?", "preference test"),
]
for idx,(qry,cat) in enumerate(boundary):
    i=91+idx
    ex={"messages":[m("user",qry),m("assistant",content="Let me think about that carefully.")],"tools":[wt(n,{"q":{"type":"string"}},["q"]) for n in random.sample(CITIES[:6],3)]}
    save(i,f"Boundary: {cat}",ex)

# Report
print(f"\n{'='*60}")
print(f"CYCLE 100 V2 COMPLETE: {len(results)} improvements")
print(f"{'='*60}")
with open(OUT/"all-100-v2.jsonl","w") as f:
    for p in sorted(OUT.glob("iter-*.jsonl")):
        f.write(p.read_text(encoding="utf-8"))
pc=Counter(r["phase"] for r in results)
for p in PHASES: print(f"  {p:15s}: {pc.get(p,0)}")
print(f"\nTotal files: {len(results)}")
print(f"Output: {OUT.resolve()}/")
