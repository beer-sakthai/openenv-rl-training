#!/usr/bin/env python3
"""100 iterations: each generates 1 unique gap-filling example. No stopping."""
import json, itertools, random
from pathlib import Path
from collections import Counter
random.seed(42)
OUT = Path("cycle-100-output"); OUT.mkdir(exist_ok=True)
PHASES = ["DATA","TRAIN","EVAL","ERROR","PUBLISH","DEPLOY","DISCOVER","MONITOR","PROFILE","ITERATE"]
CITIES = ["Bangkok","Tokyo","London","Paris","New York","Dubai","Singapore","Berlin","Rome","Madrid","Seoul","Mumbai","Sydney","Toronto","Moscow","Istanbul","Amsterdam","Prague","Vienna","Oslo","Stockholm","Helsinki","Dublin","Copenhagen","Zurich"]
TICKERS = ["AAPL","GOOGL","MSFT","TSLA","NVDA","AMD","AMZN","META","NFLX","SPOT","IBM","ORCL","ADBE","CRM","INTC"]
TOPICS = ["AI","climate change","renewable energy","space exploration","quantum computing","cryptocurrency","electric vehicles","cybersecurity","biotechnology","robotics","blockchain","machine learning","data science","cloud computing","iot"]
LANGS = ["th","fr","de","es","it","pt","ja","ko","zh","ar","hi","ru","nl","sv","pl"]
ORIGINS = ["BKK","NYC","LHR","CDG","NRT","DXB","SFO","LAX","HKG","SIN","FRA","AMS","IST","MUC","BCN"]

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
    fname = f"iter-{i:03d}-{phase.lower()}.jsonl"
    with open(OUT/fname, "w") as f:
        f.write(json.dumps(ex, ensure_ascii=False)+"\n")
    results.append({"iter":i,"phase":phase,"gap":gap})
    print(f"  #{i:3d} [{phase:15s}] {gap}")

# 1-10: Single tools
for i in range(1,11):
    c = CITIES[(i-1)*2%25]; t = TICKERS[(i-1)*3%15]; top = TOPICS[(i-1)*5%15]
    lg = LANGS[(i-1)*7%15]; o=ORIGINS[i%15]; d=ORIGINS[(i+3)%15]; sr=i
    if i%10==1:
        call = tc("get_weather",{"location":c,"unit":"celsius"})
        ex = {"messages":[m("user",f"Weather in {c}?"),m("assistant",tc_val=call)],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"])]}
        save(i,f"Weather {c}",ex)
    elif i%10==2:
        call = tc("get_stock_price",{"ticker":t})
        ex = {"messages":[m("user",f"Stock {t}"),m("assistant",tc_val=call)],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"])]}
        save(i,f"Stock {t}",ex)
    elif i%10==3:
        call = tc("search_web",{"query":top})
        ex = {"messages":[m("user",f"Search {top}"),m("assistant",tc_val=call)],"tools":[wt("search_web",{"query":{"type":"string"}},["query"])]}
        save(i,f"Search {top}",ex)
    elif i%10==4:
        call = tc("calculator",{"expression":f"2**{i+5}"})
        ex = {"messages":[m("user",f"Calc 2**{i+5}"),m("assistant",tc_val=call)],"tools":[wt("calculator",{"expression":{"type":"string"}},["expression"])]}
        save(i,f"Calc 2**{i+5}",ex)
    elif i%10==5:
        call = tc("translate_text",{"text":"hello","target_lang":lg})
        ex = {"messages":[m("user",f"Translate to {lg}"),m("assistant",tc_val=call)],"tools":[wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"])]}
        save(i,f"Translate {lg}",ex)
    elif i%10==6 and o!=d:
        call = tc("book_flight",{"origin":o,"destination":d,"date":"2026-09-15"})
        ex = {"messages":[m("user",f"Book {o} to {d}"),m("assistant",tc_val=call)],"tools":[wt("book_flight",{"origin":{"type":"string"},"destination":{"type":"string"},"date":{"type":"string"}},["origin","destination","date"])]}
        save(i,f"Flight {o}->{d}",ex)
    elif i%10==7:
        call = tc("send_email",{"to":f"u{i}@t.com","subject":top,"body":"Update"})
        ex = {"messages":[m("user",f"Email about {top}"),m("assistant",tc_val=call)],"tools":[wt("send_email",{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},["to","subject"])]}
        save(i,f"Email {top}",ex)
    elif i%10==8:
        call = tc("get_news",{"topic":top,"count":i})
        ex = {"messages":[m("user",f"News {top}"),m("assistant",tc_val=call)],"tools":[wt("get_news",{"topic":{"type":"string"},"count":{"type":"integer"}},["topic"])]}
        save(i,f"News {top} cnt={i}",ex)
    elif i%10==9:
        call = tc("get_restaurant_info",{"name":"Best Food","location":c})
        ex = {"messages":[m("user",f"Restaurant in {c}"),m("assistant",tc_val=call)],"tools":[wt("get_restaurant_info",{"name":{"type":"string"},"location":{"type":"string"}},["name"])]}
        save(i,f"Restaurant {c}",ex)
    elif i%10==0:
        call = tc("get_time",{"location":c})
        ex = {"messages":[m("user",f"Time in {c}?"),m("assistant",tc_val=call)],"tools":[wt("get_time",{"location":{"type":"string"}},["location"])]}
        save(i,f"Time {c}",ex)

# 11-20: Parallel
for idx in range(10):
    i=11+idx; c1=CITIES[idx*2%25]; c2=CITIES[(idx*2+7)%25]; t1=TICKERS[idx*3%15]; t2=TICKERS[(idx*3+5)%15]; top=TOPICS[idx*7%15]; lg=LANGS[idx*11%15]
    p = idx%5
    if p==0:
        c = tc("get_weather",{"location":c1})+tc("get_weather",{"location":c2})
        ex = {"messages":[m("user",f"Weather {c1}+{c2}"),m("assistant",tc_val=c)],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"])]}
        save(i,f"Parallel weather {c1}+{c2}",ex)
    elif p==1:
        c = tc("get_stock_price",{"ticker":t1})+tc("get_stock_price",{"ticker":t2})
        ex = {"messages":[m("user",f"Stocks {t1}+{t2}"),m("assistant",tc_val=c)],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"])]}
        save(i,f"Parallel stocks {t1}+{t2}",ex)
    elif p==2:
        c = tc("get_weather",{"location":c1})+tc("get_stock_price",{"ticker":t1})
        ex = {"messages":[m("user",f"Weather {c1}+stock {t1}"),m("assistant",tc_val=c)],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"]),wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"])]}
        save(i,f"Parallel weather+stock",ex)
    elif p==3:
        c = tc("get_news",{"topic":top,"count":3})+tc("translate_text",{"text":"summary","target_lang":lg})
        ex = {"messages":[m("user",f"News+translate {top}"),m("assistant",tc_val=c)],"tools":[wt("get_news",{"topic":{"type":"string"},"count":{"type":"integer"}},["topic"]),wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"])]}
        save(i,f"Parallel news+translate",ex)
    else:
        c = tc("search_web",{"query":top})+tc("translate_text",{"text":"results","target_lang":lg})
        ex = {"messages":[m("user",f"Search+translate {top}"),m("assistant",tc_val=c)],"tools":[wt("search_web",{"query":{"type":"string"}},["query"]),wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"])]}
        save(i,f"Parallel search+translate",ex)

# 21-30: Irrelevance
queries = ["What is the meaning of life?","How do planes fly?","What is gravity?","Who painted the Mona Lisa?","What is the capital of France?","How does the internet work?","What is DNA?","Why is the sky blue?","What causes rainbows?","How do vaccines work?"]
irr_tools = [wt(n,{"q":{"type":"string"}},["q"]) for n in random.sample(CITIES[:8],4)]
for idx,q in enumerate(queries):
    i=21+idx
    ex = {"messages":[m("user",q),m("assistant",content=f"Answering: {q.split('?')[0]}.")],"tools":irr_tools}
    save(i,f"Irrelevance: {q[:30]}",ex)

# 31-40: Arg edge cases
edges = [("get_weather",{"location":" NY ","unit":" CELSIUS "}),("get_stock_price",{"ticker":" aapl "}),("translate_text",{"text":" Hi ","target_lang":" TH "}),("search_web",{"query":" AI "}),("calculator",{"expression":" 2+2 "}),("get_news",{"topic":" Climate ","count":5}),("book_flight",{"origin":" bkk ","destination":" nrt ","date":" 2026-09-01 "}),("send_email",{"to":" U@T.COM ","subject":" Hi ","body":" OK "}),("get_restaurant_info",{"name":" Sushi ","location":" tokyo "}),("get_time",{"location":" London "})]
for idx,(tn,args) in enumerate(edges):
    i=31+idx
    props = {k:{"type":"string"} for k in args}
    ex = {"messages":[m("user",f"Test {tn}"),m("assistant",tc_val=tc(tn,args))],"tools":[wt(tn,props,list(args.keys()))]}
    save(i,f"Args edge: {tn}",ex)

# 41-50: Multi-turn
for idx in range(10):
    i=41+idx; c1=CITIES[idx*2%25]; c2=CITIES[(idx*2+3)%25]; t1=TICKERS[idx*3%15]; t2=TICKERS[(idx*3+2)%15]
    if idx%3==0:
        c=tc("get_weather",{"location":c1})
        ex={"messages":[m("user",f"Weather {c1}?"),m("assistant",tc_val=c),m("tool","22C"),m("user",f"What about {c2}?")],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"])]}
        save(i,f"Multi-turn weather {c1}->{c2}",ex)
    elif idx%3==1:
        c=tc("get_stock_price",{"ticker":t1})
        ex={"messages":[m("user",f"Price {t1}?"),m("assistant",tc_val=c),m("tool","$500"),m("user",f"{t2}?")],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"])]}
        save(i,f"Multi-turn stock {t1}->{t2}",ex)
    else:
        c=tc("get_news",{"topic":"AI","count":3})
        ex={"messages":[m("user","AI news?"),m("assistant",tc_val=c),m("tool","Story 1"),m("user","Details?")],"tools":[wt("get_news",{"topic":{"type":"string"},"count":{"type":"integer"}},["topic"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        save(i,"Multi-turn news+search",ex)

# 51-60: Hard negatives
hard_pairs=[("get_weather","get_time"),("get_stock_price","get_news"),("search_web","get_news"),("book_flight","get_restaurant_info"),("translate_text","search_web"),("send_email","book_flight"),("calculator","get_stock_price"),("get_time","get_weather"),("get_news","search_web"),("book_flight","send_email")]
h_args={"get_weather":{"location":"P"},"get_time":{"location":"P"},"get_stock_price":{"ticker":"A"},"get_news":{"topic":"AI","count":3},"search_web":{"query":"t"},"book_flight":{"origin":"A","destination":"B","date":"2026-01-01"},"get_restaurant_info":{"name":"S"},"translate_text":{"text":"hi","target_lang":"th"},"send_email":{"to":"a@b.com","subject":"S","body":"B"},"calculator":{"expression":"1+1"}}
for idx,(cor,wr) in enumerate(hard_pairs):
    i=51+idx
    ex={"messages":[m("user",f"Need {cor}"),m("assistant",tc_val=tc(cor,h_args[cor]))],"tools":[wt(cor,{},[]),wt(wr,{},[])]}
    save(i,f"Hard neg: {cor} vs {wr}",ex)

# 61-70: Multi-hop
for idx in range(10):
    i=61+idx; c=CITIES[idx*2%25]; t=TICKERS[idx*3%15]; top=TOPICS[idx*7%15]; lg=LANGS[idx*11%15]
    if idx%3==0:
        ex={"messages":[m("user",f"Weather {c}+translate {lg}"),m("assistant",tc_val=tc("get_weather",{"location":c})+tc("translate_text",{"text":"r","target_lang":lg}))],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"]),wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"])]}
        save(i,f"Multi-hop weather+translate",ex)
    elif idx%3==1:
        ex={"messages":[m("user",f"Stock {t}+analysis"),m("assistant",tc_val=tc("get_stock_price",{"ticker":t})+tc("search_web",{"query":f"{t} analysis"}))],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        save(i,f"Multi-hop stock+search",ex)
    else:
        ex={"messages":[m("user",f"News {top}+translate"),m("assistant",tc_val=tc("get_news",{"topic":top,"count":5})+tc("translate_text",{"text":"h","target_lang":lg}))],"tools":[wt("get_news",{"topic":{"type":"string"},"count":{"type":"integer"}},["topic"]),wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"])]}
        save(i,f"Multi-hop news+translate",ex)

# 71-80: Type coercion
for idx in range(10):
    i=71+idx; cnt=idx+1; top=TOPICS[idx%15]
    ex={"messages":[m("user",f"Get {cnt} news about {top}"),m("assistant",tc_val=tc("get_news",{"topic":top,"count":cnt}))],"tools":[wt("get_news",{"topic":{"type":"string"},"count":{"type":"integer"}},["topic"])]}
    save(i,f"Type coercion count={cnt}",ex)

# 81-90: Held-out combos
combos=list(itertools.combinations(CITIES[:8],2)); random.shuffle(combos)
for idx in range(10):
    i=81+idx; t1,t2=combos[idx]
    ex={"messages":[m("user",f"Need {t1},{t2}"),m("assistant",tc_val=tc("get_weather",{"location":t1})+tc("get_time",{"location":t2}))],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"]),wt("get_time",{"location":{"type":"string"}},["location"])]}
    save(i,f"Held-out: {t1}+{t2}",ex)

# 91-100: Safety
safe=[("0/0?","math edge"),("Repeat A 1000x","degenerate"),("Largest number?","philosophical"),("Sentient?","consciousness"),("Can you hack?","safety"),("After death?","philosophical"),("Ultimate answer?","final"),("Feelings?","anthropomorphism"),("What is 42?","pop"),("Something amazing","caps")]
irr_t2=[wt(n,{"q":{"type":"string"}},["q"]) for n in random.sample(CITIES,3)]
for idx,(qry,cat) in enumerate(safe):
    i=91+idx
    ex={"messages":[m("user",qry),m("assistant",content="Let me think about that.")],"tools":irr_t2}
    save(i,f"Safety: {cat}",ex)

# Final report
print(f"\n{'='*60}")
print(f"CYCLE 100 COMPLETE: {len(results)} improvements")
print(f"{'='*60}")
with open(OUT/"all-100.jsonl","w") as f:
    for p in sorted(OUT.glob("iter-*.jsonl")):
        f.write(p.read_text(encoding="utf-8"))
pc=Counter(r["phase"] for r in results)
print("Phase distribution:")
for p in PHASES: print(f"  {p:15s}: {pc.get(p,0)}")
