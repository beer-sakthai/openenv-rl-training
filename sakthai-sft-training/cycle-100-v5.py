#!/usr/bin/env python3
"""100 more — fifth round, 10 brand new categories never used in v1-v4."""
import json, random
from pathlib import Path
from collections import Counter
random.seed(55555)
OUT = Path("cycle-100-v5"); OUT.mkdir(exist_ok=True)
PHASES = ["DATA","TRAIN","EVAL","ERROR","PUBLISH","DEPLOY","DISCOVER","MONITOR","PROFILE","ITERATE"]
C = ["Guangzhou","Shenzhen","Chongqing","Chengdu","Nanjing","Hangzhou","Wuhan","XiAn","Shenyang","Qingdao","Kolkata","Chennai","Bangalore","Hyderabad","Pune","Casablanca","Tunis","Algiers","Rabat","Accra","Addis_Ababa","Dar_es_Salaam","Cape_Town","Marrakech","Tanger"]
T = ["SNAP","PINS","RBLX","U","FSLY","DDOG","MDB","ESTC","NET","ZS","WISE","CFLT","TOST","GTLB","FRT"]
TO = ["astrobiology","exoplanets","stellar_evolution","cosmic_microwave","gravitational_waves","dark_matter","neutrino_physics","solar_physics","magnetohydrodynamics","helioseismology","paleoclimatology","geomorphology","tectonics","stratigraphy","petrology"]
L = ["xh","zu","ig","yo","ha","am","om","ti","sg","rn","kg","ny","rw","sn","st"]
ANIMALS = ["axolotl","platypus","narwhal","okapi","fossa","quokka","kakapo","tapir","capybara","manatee","coelacanth","horseshoe_crab","tardigrade","nautilus","pangolin"]

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

# ── 1-10: Paginated tool calls (page, limit, offset params)
for idx in range(10):
    i=1+idx; top=TO[idx%15]; c=C[idx%25]; an=ANIMALS[idx%15]
    if idx%3==0:
        call = tc("get_news",{"topic":top,"count":10,"page":idx+1})
        ex={"messages":[m("user",f"News about {top} page {idx+1}"),m("assistant",tc_val=call)],"tools":[wt("get_news",{"topic":{"type":"string"},"count":{"type":"integer"},"page":{"type":"integer"}},["topic","page"])]}
        sv(i,f"Paginated news {top} page {idx+1}",ex)
    elif idx%3==1:
        call = tc("search_web",{"query":an,"limit":10,"offset":idx*10})
        ex={"messages":[m("user",f"Search {an} page {idx+1}"),m("assistant",tc_val=call)],"tools":[wt("search_web",{"query":{"type":"string"},"limit":{"type":"integer"},"offset":{"type":"integer"}},["query","limit","offset"])]}
        sv(i,f"Paginated search {an} offset {idx*10}",ex)
    else:
        call = tc("send_email",{"to":[f"user{x}@c.com" for x in range(3)],"subject":"Batch","body":"Mass email","batch_size":3})
        ex={"messages":[m("user","Send batch emails"),m("assistant",tc_val=call)],"tools":[wt("send_email",{"to":{"type":"array","items":{"type":"string"}},"subject":{"type":"string"},"body":{"type":"string"},"batch_size":{"type":"integer"}},["to","subject","batch_size"])]}
        sv(i,"Paginated batch email",ex)

# ── 11-20: Time-range queries (between dates, since, until)
for idx in range(10):
    i=11+idx; top=TO[idx%15]; c=C[idx%25]; t=T[idx%15]; an=ANIMALS[idx%15]
    if idx%4==0:
        call = tc("get_news",{"topic":top,"from_date":"2026-01-01","to_date":"2026-06-30"})
        ex={"messages":[m("user",f"News {top} Jan-Jun 2026"),m("assistant",tc_val=call)],"tools":[wt("get_news",{"topic":{"type":"string"},"from_date":{"type":"string"},"to_date":{"type":"string"}},["topic","from_date","to_date"])]}
        sv(i,f"Time range {top} H1",ex)
    elif idx%4==1:
        call = tc("get_weather",{"location":c,"start_date":"2026-07-01","end_date":"2026-07-07"})
        ex={"messages":[m("user",f"Weather {c} first week Jul"),m("assistant",tc_val=call)],"tools":[wt("get_weather",{"location":{"type":"string"},"start_date":{"type":"string"},"end_date":{"type":"string"}},["location"])]}
        sv(i,f"Time range weather {c} week",ex)
    elif idx%4==2:
        call = tc("get_stock_price",{"ticker":t,"since":"2025-01-01","until":"2026-01-01"})
        ex={"messages":[m("user",f"Stock {t} 2025"),m("assistant",tc_val=call)],"tools":[wt("get_stock_price",{"ticker":{"type":"string"},"since":{"type":"string"},"until":{"type":"string"}},["ticker","since","until"])]}
        sv(i,f"Time range stock {t} 2025",ex)
    else:
        call = tc("search_web",{"query":f"research {an}","year_start":2020,"year_end":2026})
        ex={"messages":[m("user",f"Research {an} 2020-2026"),m("assistant",tc_val=call)],"tools":[wt("search_web",{"query":{"type":"string"},"year_start":{"type":"integer"},"year_end":{"type":"integer"}},["query","year_start","year_end"])]}
        sv(i,f"Time range {an} 2020-2026",ex)

# ── 21-30: Scheduling/delayed execution
for idx in range(10):
    i=21+idx; c=C[idx%25]; an=ANIMALS[idx%15]; top=TO[idx%15]
    if idx%4==0:
        call = tc("set_reminder",{"text":f"check weather {c}","time":"2026-08-01T08:00"})
        ex={"messages":[m("user",f"Remind me to check weather in {c} tomorrow"),m("assistant",tc_val=call)],"tools":[wt("set_reminder",{"text":{"type":"string"},"time":{"type":"string"}},["text","time"])]}
        sv(i,f"Scheduled reminder weather {c}",ex)
    elif idx%4==1:
        call = tc("schedule_email",{"to":"a@b.com","subject":"Report","body":"Weekly report","send_at":"2026-08-05T09:00"})
        ex={"messages":[m("user","Schedule email for next week"),m("assistant",tc_val=call)],"tools":[wt("schedule_email",{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"},"send_at":{"type":"string"}},["to","subject","body","send_at"])]}
        sv(i,"Scheduled email weekly",ex)
    elif idx%4==2:
        call = tc("get_news",{"topic":top,"schedule":"daily","count":5})
        ex={"messages":[m("user",f"Daily news briefing {top}"),m("assistant",tc_val=call)],"tools":[wt("get_news",{"topic":{"type":"string"},"schedule":{"type":"string"},"count":{"type":"integer"}},["topic","schedule"])]}
        sv(i,f"Scheduled news {top} daily",ex)
    else:
        call = tc("book_flight",{"origin":"BKK","destination":c[:3].upper(),"date":"flexible","prefer":"morning"})
        ex={"messages":[m("user","Book flexible flight morning"),m("assistant",tc_val=call)],"tools":[wt("book_flight",{"origin":{"type":"string"},"destination":{"type":"string"},"date":{"type":"string"},"prefer":{"type":"string"}},["origin","destination"])]}
        sv(i,f"Scheduled flexible flight",ex)

# ── 31-40: Aggregation tools (sum, average, count, group)
for idx in range(10):
    i=31+idx; t=T[idx%15]; top=TO[idx%15]; c=C[idx%25]
    if idx%3==0:
        call = tc("aggregate",{"metric":"average","target":f"{t} stock price","period":"30d"})
        ex={"messages":[m("user",f"Average {t} stock price 30 days"),m("assistant",tc_val=call)],"tools":[wt("aggregate",{"metric":{"type":"string","enum":["sum","average","count","max","min"]},"target":{"type":"string"},"period":{"type":"string"}},["metric","target"])]}
        sv(i,f"Aggregate avg {t} 30d",ex)
    elif idx%3==1:
        call = tc("aggregate",{"metric":"count","target":f"articles about {top}","category":"news"})
        ex={"messages":[m("user",f"Count articles about {top}"),m("assistant",tc_val=call)],"tools":[wt("aggregate",{"metric":{"type":"string","enum":["sum","average","count","max","min"]},"target":{"type":"string"},"category":{"type":"string"}},["metric","target"])]}
        sv(i,f"Aggregate count {top}",ex)
    else:
        call = tc("aggregate",{"metric":"max","target":f"temperature in {c}","month":"July"})
        ex={"messages":[m("user",f"Max temp in {c} July"),m("assistant",tc_val=call)],"tools":[wt("aggregate",{"metric":{"type":"string","enum":["sum","average","count","max","min"]},"target":{"type":"string"},"month":{"type":"string"}},["metric","target"])]}
        sv(i,f"Aggregate max temp {c}",ex)

# ── 41-50: Geo-spatial queries
for idx in range(10):
    i=41+idx; c=C[idx%25]; an=ANIMALS[idx%15]
    if idx%3==0:
        call = tc("get_nearby",{"location":c,"radius_km":10,"type":"restaurant"})
        ex={"messages":[m("user",f"Restaurants within 10km of {c}"),m("assistant",tc_val=call)],"tools":[wt("get_nearby",{"location":{"type":"string"},"radius_km":{"type":"number"},"type":{"type":"string"}},["location","type"])]}
        sv(i,f"Geo nearby {c} restaurants",ex)
    elif idx%3==1:
        call = tc("get_nearby",{"location":c,"radius_km":50,"type":"hospital"})
        ex={"messages":[m("user",f"Hospitals near {c}"),m("assistant",tc_val=call)],"tools":[wt("get_nearby",{"location":{"type":"string"},"radius_km":{"type":"number"},"type":{"type":"string"}},["location","type"])]}
        sv(i,f"Geo nearby {c} hospitals",ex)
    else:
        call = tc("get_distance",{"from":C[(idx+5)%25],"to":c,"unit":"km"})
        ex={"messages":[m("user",f"Distance from {C[(idx+5)%25]} to {c}"),m("assistant",tc_val=call)],"tools":[wt("get_distance",{"from":{"type":"string"},"to":{"type":"string"},"unit":{"type":"string","enum":["km","miles"]}},["from","to"])]}
        sv(i,f"Geo distance {C[(idx+5)%25]}->{c}",ex)

# ── 51-60: Conditional tool calls (if-then-else patterns)
for idx in range(10):
    i=51+idx; c=C[idx%25]; t=T[idx%15]; top=TO[idx%15]
    if idx%3==0:
        c1 = tc("get_weather",{"location":c})
        c2 = tc("get_restaurant_info",{"name":"outdoor_cafe","location":c})
        ex={"messages":[m("user",f"Weather in {c}? If sunny find outdoor cafe"),m("assistant",tc_val=c1)],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"]),wt("get_restaurant_info",{"name":{"type":"string"},"location":{"type":"string"}},["name"])]}
        sv(i,f"Conditional weather->cafe {c}",ex)
    elif idx%3==1:
        c1 = tc("get_stock_price",{"ticker":t})
        c2 = tc("search_web",{"query":f"{t} earnings report"})
        ex={"messages":[m("user",f"Check {t} price, if dropped find earnings"),m("assistant",tc_val=c1)],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        sv(i,f"Conditional stock->earnings {t}",ex)
    else:
        c1 = tc("translate_text",{"text":"hello","target_lang":"th"})
        c2 = tc("search_web",{"query":"thai greeting etiquette"})
        ex={"messages":[m("user","Translate hello to Thai and find greetings etiquette"),m("assistant",tc_val=c1)],"tools":[wt("translate_text",{"text":{"type":"string"},"target_lang":{"type":"string"}},["text","target_lang"]),wt("search_web",{"query":{"type":"string"}},["query"])]}
        sv(i,"Conditional translate->etiquette",ex)

# ── 61-70: Collaborative/sharing tools
for idx in range(10):
    i=61+idx; c=C[idx%25]; t=T[idx%15]; top=TO[idx%15]; an=ANIMALS[idx%15]; lg=L[idx%15]
    if idx%5==0:
        call = tc("share",{"content":f"weather report for {c}","target":"team@co.com","permission":"view"})
        ex={"messages":[m("user",f"Share weather report for {c} with team"),m("assistant",tc_val=call)],"tools":[wt("share",{"content":{"type":"string"},"target":{"type":"string"},"permission":{"type":"string","enum":["view","edit","admin"]}},["content","target"])]}
        sv(i,f"Share weather {c}",ex)
    elif idx%5==1:
        call = tc("assign",{"task":f"research {top}","assignee":"colleague","due":"2026-09-01"})
        ex={"messages":[m("user",f"Assign {top} research to colleague"),m("assistant",tc_val=call)],"tools":[wt("assign",{"task":{"type":"string"},"assignee":{"type":"string"},"due":{"type":"string"}},["task","assignee","due"])]}
        sv(i,f"Assign research {top}",ex)
    elif idx%5==2:
        call = tc("tag",{"item":f"report {an}","tags":["biology","rare_species","research"]})
        ex={"messages":[m("user",f"Tag {an} report with biology tags"),m("assistant",tc_val=call)],"tools":[wt("tag",{"item":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}}},["item","tags"])]}
        sv(i,f"Tag report {an}",ex)
    elif idx%5==3:
        call = tc("notify",{"users":["team_a@c.com","team_b@c.com"],"message":f"update on {t}","priority":"high"})
        ex={"messages":[m("user",f"Notify teams about {t} update"),m("assistant",tc_val=call)],"tools":[wt("notify",{"users":{"type":"array","items":{"type":"string"}},"message":{"type":"string"},"priority":{"type":"string","enum":["low","normal","high"]}},["users","message"])]}
        sv(i,f"Notify about {t}",ex)
    else:
        call = tc("subscribe",{"topic":top,"frequency":"weekly","channels":["email","slack"]})
        ex={"messages":[m("user",f"Subscribe to weekly {top} updates"),m("assistant",tc_val=call)],"tools":[wt("subscribe",{"topic":{"type":"string"},"frequency":{"type":"string","enum":["daily","weekly","monthly"]},"channels":{"type":"array","items":{"type":"string"}}},["topic","frequency"])]}
        sv(i,f"Subscribe {top} weekly",ex)

# ── 71-80: Tool metadata/introspection
for idx in range(10):
    i=71+idx; c=C[idx%25]; an=ANIMALS[idx%15]
    if idx%3==0:
        call = tc("list_tools",{"category":"weather","version":"latest"})
        ex={"messages":[m("user","What weather tools are available?"),m("assistant",tc_val=call)],"tools":[wt("list_tools",{"category":{"type":"string"},"version":{"type":"string"}},["category"])]}
        sv(i,"Tool metadata list weather",ex)
    elif idx%3==1:
        call = tc("get_tool_info",{"tool_name":"get_weather"})
        ex={"messages":[m("user","How does get_weather work?"),m("assistant",tc_val=call)],"tools":[wt("get_tool_info",{"tool_name":{"type":"string"}},["tool_name"])]}
        sv(i,"Tool metadata get_weather info",ex)
    else:
        call = tc("get_tool_history",{"tool_name":f"get_stock_price","limit":5})
        ex={"messages":[m("user","Recent stock price calls?"),m("assistant",tc_val=call)],"tools":[wt("get_tool_history",{"tool_name":{"type":"string"},"limit":{"type":"integer"}},["tool_name"])]}
        sv(i,"Tool metadata history stock",ex)

# ── 81-90: Tool versioning (v1, v2, deprecated)
for idx in range(10):
    i=81+idx; c=C[idx%25]; t=T[idx%15]; top=TO[idx%15]
    if idx%4==0:
        call = tc("get_weather",{"location":c,"version":"v2","unit":"celsius"})
        ex={"messages":[m("user",f"Weather {c} using v2 API"),m("assistant",tc_val=call)],"tools":[wt("get_weather",{"location":{"type":"string"},"version":{"type":"string","enum":["v1","v2"]},"unit":{"type":"string"}},["location","version"])]}
        sv(i,f"Tool version v2 weather {c}",ex)
    elif idx%4==1:
        call = tc("get_stock_price",{"ticker":t,"version":"v2","source":"realtime"})
        ex={"messages":[m("user",f"Stock {t} realtime v2"),m("assistant",tc_val=call)],"tools":[wt("get_stock_price",{"ticker":{"type":"string"},"version":{"type":"string","enum":["v1","v2"]},"source":{"type":"string","enum":["delayed","realtime"]}},["ticker","version"])]}
        sv(i,f"Tool version v2 stock {t}",ex)
    elif idx%4==2:
        call = tc("search_web",{"query":top,"version":"v1","filter":"basic"})
        ex={"messages":[m("user",f"Search {top} legacy v1"),m("assistant",tc_val=call)],"tools":[wt("search_web",{"query":{"type":"string"},"version":{"type":"string","enum":["v1","v2"]},"filter":{"type":"string","enum":["basic","advanced"]}},["query","version"])]}
        sv(i,f"Tool version v1 search {top}",ex)
    else:
        call = tc("get_news",{"topic":"tech","version":"deprecated","fallback":"v2"})
        ex={"messages":[m("user","Tech news using old API"),m("assistant",tc_val=call)],"tools":[wt("get_news",{"topic":{"type":"string"},"version":{"type":"string","enum":["v1","v2","deprecated"]},"fallback":{"type":"string"}},["topic","version"])]}
        sv(i,"Tool version deprecated news",ex)

# ── 91-100: Natural language explanations as fallback
for idx in range(10):
    i=91+idx; c=C[idx%25]; t=T[idx%15]; top=TO[idx%15]; an=ANIMALS[idx%15]
    if idx%4==0:
        call = tc("get_weather",{"location":c})
        ex={"messages":[m("user",f"Explain weather in {c} simply"),m("assistant",tc_val=call),m("tool","25C sunny"),m("assistant",content=f"The weather in {c} is warm and sunny at 25C.")],"tools":[wt("get_weather",{"location":{"type":"string"}},["location"])]}
        sv(i,f"Explain weather {c}",ex)
    elif idx%4==1:
        call = tc("get_stock_price",{"ticker":t})
        ex={"messages":[m("user",f"What does {t} stock price mean?"),m("assistant",tc_val=call),m("tool","$450"),m("assistant",content=f"The stock price of {t} is currently $450 per share.")],"tools":[wt("get_stock_price",{"ticker":{"type":"string"}},["ticker"])]}
        sv(i,f"Explain stock {t}",ex)
    elif idx%4==2:
        call = tc("search_web",{"query":f"what is {an}"})
        ex={"messages":[m("user",f"What is a {an}?"),m("assistant",tc_val=call),m("tool","Result: unique animal..."),m("assistant",content=f"A {an} is a fascinating creature.")],"tools":[wt("search_web",{"query":{"type":"string"}},["query"])]}
        sv(i,f"Explain animal {an}",ex)
    else:
        call = tc("get_news",{"topic":top,"count":3})
        ex={"messages":[m("user",f"Summarize {top} news"),m("assistant",tc_val=call),m("tool","3 articles..."),m("assistant",content=f"Here are the latest developments in {top}.")],"tools":[wt("get_news",{"topic":{"type":"string"},"count":{"type":"integer"}},["topic"])]}
        sv(i,f"Explain news {top}",ex)

print(f"\n{'='*60}")
print(f"CYCLE 100 V5 COMPLETE: {len(res)} new improvements")
print(f"{'='*60}")
with open(OUT/"all-100-v5.jsonl","w") as f:
    for p in sorted(OUT.glob("iter-*.jsonl")):
        f.write(p.read_text(encoding="utf-8"))
pc=Counter(r["phase"] for r in res)
for p in PHASES: print(f"  {p:15s}: {pc.get(p,0)}")
print(f"\n5-round total: 500 improvements")
print(f"  50 unique categories across v1-v5")
print(f"  Every phase: 50 examples each")
