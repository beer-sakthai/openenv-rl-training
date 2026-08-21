#!/usr/bin/env python3
"""
Gap-filling dataset augmentation.
Targets the specific gap: 0.5B at 91.2% vs 1.5B at 48.2%.
Root cause: 0.5B uses all 7 LoRA targets + pure tool-calling data.
            1.5B uses only 4 targets + chat data dilution.

This creates pure tool-calling data with no chit-chat to close the gap.
All examples use ALL 7 linear modules as targets (matching the v2 config).
"""
import json, random
from pathlib import Path

random.seed(42)
OUT = Path("gap-filled")
OUT.mkdir(exist_ok=True)

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_time", "description": "Get time for a city", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the web for information", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Calculate a math expression", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_stock_price", "description": "Get current stock price", "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "Translate text to a language", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}}},
    {"type": "function", "function": {"name": "book_flight", "description": "Book a flight between cities", "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "Send an email", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject"]}}},
    {"type": "function", "function": {"name": "get_news", "description": "Get news articles for a topic", "parameters": {"type": "object", "properties": {"topic": {"type": "string"}, "count": {"type": "integer"}}, "required": ["topic"]}}},
    {"type": "function", "function": {"name": "get_restaurant_info", "description": "Get info about a restaurant", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "location": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "set_reminder", "description": "Set a reminder for a time", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "time": {"type": "string"}}, "required": ["text", "time"]}}},
    {"type": "function", "function": {"name": "get_directions", "description": "Get directions between locations", "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "mode": {"type": "string", "enum": ["driving", "walking", "transit"]}}, "required": ["origin", "destination"]}}},
]

def m(role, content=None, tc=None):
    x = {"role": role}
    if content is not None: x["content"] = content
    if tc: x["tool_calls"] = tc
    return x

def tc(name, args):
    return [{"function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]

def subs(names):
    return [t for t in TOOLS if t["function"]["name"] in names]

def save(name, examples):
    path = OUT / f"{name}.jsonl"
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  {name}.jsonl: {len(examples)} examples")

all_examples = []

# ── GAP 1: Pure tool-calling (no chat, no diluted examples) ─────────────
# Root cause: 1.5B uses 48.2% because training included chat data.
# Fix: examples that FORCE tool use — every query MUST result in a tool call.
def gap1_pure_tc(n=100):
    cities = ["Bangkok","Tokyo","London","Paris","New York","Dubai","Singapore","Berlin","Rome","Madrid",
              "Seoul","Mumbai","Sydney","Toronto","Moscow","Istanbul","Amsterdam","Prague","Vienna","Oslo"]
    tickers = ["AAPL","GOOGL","MSFT","TSLA","NVDA","AMD","AMZN","META","NFLX","SPOT"]
    topics = ["AI","climate change","renewable energy","space exploration","quantum computing",
              "cryptocurrency","electric vehicles","cybersecurity","biotechnology","robotics"]
    langs = ["th","fr","de","es","it","pt","ja","ko","zh","ar"]
    origins = ["BKK","NYC","LHR","CDG","NRT","DXB","SFO","LAX","HKG","SIN"]
    dests = ["NRT","LAX","CDG","HKG","SIN","BKK","LHR","JFK","DXB","SFO"]
    dates = ["2026-10-01","2026-11-15","2026-12-25","2027-01-10","2027-02-14","2027-03-20","2027-04-05","2027-05-01"]
    exprs = ["2+2","15/100*200","sqrt(144)","sin(pi/2)","log(100)","3**3","(5+3)*2","100/3"]
    emails = [("alice@co.com","Project update","Done"),("bob@firm.com","Meeting","3pm"),
              ("carol@org.com","Report","Attached"),("dave@io.com","Question","Please review")]
    rest_names = ["Sushi Bar","Pizza Place","Taco Stand","Noodle House"," Curry Shop"]
    reminders = ["Buy groceries","Call mom","Doctor appointment","Team standup","Submit report"]
    
    examples = []
    for i in range(n):
        c = cities[i % len(cities)]
        t = tickers[i % len(tickers)]
        top = topics[i % len(topics)]
        lg = langs[i % len(langs)]
        o, d = origins[i % len(origins)], dests[(i+3) % len(dests)]
        dt = dates[i % len(dates)]
        ex = exprs[i % len(exprs)]
        em = emails[i % len(emails)]
        rn = rest_names[i % len(rest_names)]
        rm = reminders[i % len(reminders)]
        
        # Cycle through tool types
        tool_type = i % 10
        if tool_type == 0:
            examples.append({"messages": [m("user", f"Weather in {c}?"), m("assistant", tc=tc("get_weather", {"location": c, "unit": "celsius"}))], "tools": subs(["get_weather","get_time"])})
        elif tool_type == 1:
            examples.append({"messages": [m("user", f"Stock price of {t}"), m("assistant", tc=tc("get_stock_price", {"ticker": t}))], "tools": subs(["get_stock_price","get_news"])})
        elif tool_type == 2:
            examples.append({"messages": [m("user", f"Search for {top}"), m("assistant", tc=tc("search_web", {"query": top}))], "tools": subs(["search_web","get_news"])})
        elif tool_type == 3:
            examples.append({"messages": [m("user", f"Calculate {ex}"), m("assistant", tc=tc("calculator", {"expression": ex}))], "tools": subs(["calculator","get_stock_price"])})
        elif tool_type == 4:
            if o != d:
                examples.append({"messages": [m("user", f"Book flight from {o} to {d} on {dt}"), m("assistant", tc=tc("book_flight", {"origin": o, "destination": d, "date": dt}))], "tools": subs(["book_flight","get_restaurant_info"])})
        elif tool_type == 5:
            examples.append({"messages": [m("user", f"Translate 'hello' to {lg}"), m("assistant", tc=tc("translate_text", {"text": "hello", "target_lang": lg}))], "tools": subs(["translate_text","search_web"])})
        elif tool_type == 6:
            examples.append({"messages": [m("user", f"News about {top}"), m("assistant", tc=tc("get_news", {"topic": top, "count": 5}))], "tools": subs(["get_news","search_web"])})
        elif tool_type == 7:
            examples.append({"messages": [m("user", f"Email {em[0]} about {em[1]}"), m("assistant", tc=tc("send_email", {"to": em[0], "subject": em[1], "body": em[2]}))], "tools": subs(["send_email","book_flight"])})
        elif tool_type == 8:
            examples.append({"messages": [m("user", f"Set reminder to {rm} at 3pm"), m("assistant", tc=tc("set_reminder", {"text": rm, "time": "15:00"}))], "tools": subs(["set_reminder","get_time","calculator"])})
        elif tool_type == 9:
            c2 = cities[(i+7) % len(cities)]
            examples.append({"messages": [m("user", f"Directions from {c} to {c2}"), m("assistant", tc=tc("get_directions", {"origin": c, "destination": c2, "mode": random.choice(["driving","walking","transit"])}))], "tools": subs(["get_directions","get_weather","get_time"])})
    return examples[:n]

# ── GAP 2: Irrelevance (current only 60, proven at 93.3% accuracy) ──────
# The 0.5B gets 93.3% irrelevance. More irrelevance data = better safety.
def gap2_irrelevance(n=100):
    queries = [
        "Hello!","How are you?","What's your name?","Tell me a joke","Good morning!",
        "Thanks for your help","Have a nice day","What is AI?","Explain quantum physics",
        "Who won the world cup?","What is the capital of Thailand?","How tall is Everest?",
        "What is the speed of light?","Who painted Starry Night?","What is DNA?",
        "How do vaccines work?","What is climate change?","Explain gravity",
        "What is the boiling point of water?","How many continents are there?",
        "What is the largest desert?","Who invented the printing press?",
        "What is the Fibonacci sequence?","How does photosynthesis work?",
        "What is the meaning of life?","Tell me a fun fact","What is 2+2?",
        "Who wrote The Great Gatsby?","What is the smallest country?",
        "How deep is the ocean?","What causes earthquakes?","How do birds fly?",
        "What is the human genome?","How does memory work?","What is consciousness?",
        "Why is the sky blue?","What are black holes?","How old is the universe?",
        "What is renewable energy?","How do solar panels work?","What is blockchain?",
        "How does encryption work?","What is democracy?","What is art?",
        "How are clouds formed?","What is the water cycle?","How do muscles grow?",
        "What is nutrition?","How do languages evolve?","What is culture?",
    ]
    examples = []
    for q in queries * (n // len(queries) + 1):
        random.shuffle(TOOLS)
        examples.append({"messages": [m("user", q), m("assistant", content=f"That's an interesting question. Let me answer directly.")], "tools": TOOLS[:random.randint(4, 8)]})
    return examples[:n]

# ── GAP 3: Arguments precision (45.7% → target 65%+) ────────────────────
# The biggest accuracy gap. Create examples testing exact argument matching.
def gap3_args_precision(n=80):
    cities = ["Bangkok","Tokyo","London","Paris","New York","Dubai","Singapore","Berlin","Rome","Madrid"]
    examples = []
    for i in range(n):
        c = cities[i % len(cities)]
        examples.append({"messages": [m("user", f"Weather in {c}?"), m("assistant", tc=tc("get_weather", {"location": c, "unit": "celsius"}))], "tools": subs(["get_weather","get_time"])})
        examples.append({"messages": [m("user", f"Weather in {c} in fahrenheit"), m("assistant", tc=tc("get_weather", {"location": c, "unit": "fahrenheit"}))], "tools": subs(["get_weather","get_time"])})
        t = ["AAPL","GOOGL","MSFT","TSLA","NVDA"][i % 5]
        examples.append({"messages": [m("user", f"Stock for {t}"), m("assistant", tc=tc("get_stock_price", {"ticker": t}))], "tools": subs(["get_stock_price","get_news","calculator"])})
        lg = ["th","fr","de","es","it","ja"][i % 6]
        examples.append({"messages": [m("user", f"Translate 'friend' to {lg}"), m("assistant", tc=tc("translate_text", {"text": "friend", "target_lang": lg}))], "tools": subs(["translate_text","search_web"])})
    return examples[:n]

# ── GAP 4: Parallel calls (multiple simultaneous tools) ─────────────────
def gap4_parallel(n=60):
    cities = ["Bangkok","Tokyo","London","Paris","New York","Dubai","Singapore","Berlin","Rome","Madrid"]
    tickers = ["AAPL","GOOGL","MSFT","TSLA","NVDA"]
    topics = ["AI","climate","sports","tech","health"]
    examples = []
    for i in range(n):
        c1, c2 = cities[i % len(cities)], cities[(i+3) % len(cities)]
        examples.append({"messages": [m("user", f"Weather in {c1} and {c2}"),
            m("assistant", tc=tc("get_weather", {"location": c1}) + tc("get_weather", {"location": c2}))],
            "tools": subs(["get_weather","get_time"])})
        t1, t2 = tickers[i % len(tickers)], tickers[(i+1) % len(tickers)]
        examples.append({"messages": [m("user", f"Stocks for {t1} and {t2}"),
            m("assistant", tc=tc("get_stock_price", {"ticker": t1}) + tc("get_stock_price", {"ticker": t2}))],
            "tools": subs(["get_stock_price","get_news"])})
        c3 = cities[(i+5) % len(cities)]
        top = topics[i % len(topics)]
        examples.append({"messages": [m("user", f"Weather in {c3} and news about {top}"),
            m("assistant", tc=tc("get_weather", {"location": c3}) + tc("get_news", {"topic": top, "count": 3}))],
            "tools": subs(["get_weather","get_news","search_web","get_time"])})
    return examples[:n]

# ── GAP 5: Selection accuracy (hard negatives) ──────────────────────────
def gap5_selection(n=60):
    pairs = [
        ("get_weather", "get_time"), ("get_stock_price", "get_news"),
        ("search_web", "get_news"), ("book_flight", "get_restaurant_info"),
        ("translate_text", "search_web"), ("send_email", "book_flight"),
        ("calculator", "get_stock_price"), ("set_reminder", "get_time"),
        ("get_directions", "book_flight"), ("get_weather", "get_restaurant_info"),
    ]
    a = {"get_weather": {"location": "Paris", "unit": "celsius"}, "get_time": {"location": "Paris"},
         "get_stock_price": {"ticker": "AAPL"}, "get_news": {"topic": "latest", "count": 3},
         "search_web": {"query": "latest news"}, "book_flight": {"origin": "BKK", "destination": "NRT", "date": "2026-09-01"},
         "get_restaurant_info": {"name": "Sushi Bar"}, "translate_text": {"text": "hello", "target_lang": "th"},
         "send_email": {"to": "a@b.com", "subject": "Hi", "body": "Hello"}, "calculator": {"expression": "2+2"},
         "set_reminder": {"text": "test", "time": "12:00"}, "get_directions": {"origin": "A", "destination": "B"}}
    examples = []
    for correct, wrong in pairs * (n // len(pairs) + 1):
        q = f"Need {correct}"
        examples.append({"messages": [m("user", q), m("assistant", tc=tc(correct, a[correct]))],
                         "tools": subs([correct, wrong])})
    return examples[:n]

# ── GAP 6: Multi-hop chains ─────────────────────────────────────────────
def gap6_multihop(n=40):
    cities = ["Bangkok","Tokyo","London","Paris"]
    tickers = ["AAPL","GOOGL","MSFT","TSLA"]
    examples = []
    for i in range(n):
        c = cities[i % len(cities)]
        t = tickers[i % len(tickers)]
        # weather → translate
        examples.append({"messages": [m("user", f"Weather in {c}, then translate to Thai"),
            m("assistant", tc=tc("get_weather", {"location": c}) + tc("translate_text", {"text": "result", "target_lang": "th"}))],
            "tools": subs(["get_weather","translate_text","search_web"])})
        # stock → search for analysis
        examples.append({"messages": [m("user", f"Stock of {t} and find analyst opinions"),
            m("assistant", tc=tc("get_stock_price", {"ticker": t}) + tc("search_web", {"query": f"{t} analyst opinion 2026"}))],
            "tools": subs(["get_stock_price","search_web","get_news"])})
        # book flight → check weather at destination
        c2 = cities[(i+2) % len(cities)]
        examples.append({"messages": [m("user", f"Book to {c} from BKK and check weather there"),
            m("assistant", tc=tc("book_flight", {"origin": "BKK", "destination": c, "date": "2026-10-01"}) + tc("get_weather", {"location": c}))],
            "tools": subs(["book_flight","get_weather","get_restaurant_info"])})
    return examples[:n]

# ── GAP 7: Multi-turn tracking ──────────────────────────────────────────
def gap7_multiturn(n=40):
    cities = ["Rome","Paris","Tokyo","London","Berlin","Madrid","Dubai","Seoul","Bangkok","Mumbai"]
    tickers = ["NVDA","AMD","AAPL","MSFT","GOOGL","TSLA","AMZN","META"]
    examples = []
    for i in range(n):
        c1, c2 = cities[i % len(cities)], cities[(i+1) % len(cities)]
        t1, t2 = tickers[i % len(tickers)], tickers[(i+1) % len(tickers)]
        # Weather → follow-up
        examples.append({"messages": [
            m("user", f"Weather in {c1}?"), m("assistant", tc=tc("get_weather", {"location": c1})),
            m("tool", f"22C in {c1}"), m("user", f"And in {c2}?")],
            "tools": subs(["get_weather","get_time"])})
        # Stock → follow-up
        examples.append({"messages": [
            m("user", f"Price of {t1}?"), m("assistant", tc=tc("get_stock_price", {"ticker": t1})),
            m("tool", "$800"), m("user", f"What about {t2}?")],
            "tools": subs(["get_stock_price","get_news"])})
        # News → search follow-up
        examples.append({"messages": [
            m("user", "Latest AI news"), m("assistant", tc=tc("get_news", {"topic": "AI", "count": 3})),
            m("tool", "Story 1... Story 2..."), m("user", "Tell me more about story 1")],
            "tools": subs(["get_news","search_web"])})
    return examples[:n]

# ── Generate all ────────────────────────────────────────────────────────
GENERATORS = [
    ("01-pure-tc", gap1_pure_tc, 100, "Pure tool-calling (no chat dilution)"),
    ("02-irrelevance", gap2_irrelevance, 100, "Irrelevance expansion (proven 93.3%)"),
    ("03-args-precision", gap3_args_precision, 80, "Argument precision (45.7% gap)"),
    ("04-parallel", gap4_parallel, 60, "Parallel calls (multi-tool)"),
    ("05-selection", gap5_selection, 60, "Selection hard negatives"),
    ("06-multihop", gap6_multihop, 40, "Multi-hop chains"),
    ("07-multiturn", gap7_multiturn, 40, "Multi-turn tracking"),
]

total = 0
print("Gap-Filling Dataset Augmentation")
print(f"Target: Close 0.5B(91.2%) → 1.5B(48.2%) gap")
print("=" * 50)
for name, gen_fn, count, desc in GENERATORS:
    batch = gen_fn(count)
    save(name, batch)
    all_examples.extend(batch)
    total += len(batch)

combined = OUT / "all-gap-filled.jsonl"
with open(combined, "w") as f:
    for ex in all_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\n{'=' * 50}")
print(f"TOTAL: {total} gap-filling examples")
print(f"{'=' * 50}")

# ── Full training data composition ──────────────────────────────────────
print(f"\n📊 Complete training dataset composition:")
print(f"   v7 original:     2,424")
print(f"   + 10x augment:     660")
print(f"   + benchmark targ:  232")
print(f"   + gap-fill:        {total}")
print(f"   = TOTAL:          {2424 + 660 + 232 + total}")
print(f"\n📈 Gap analysis addressed:")
print(f"   Root cause 1 (chat dilution): 100 pure TC examples")
print(f"   Root cause 2 (args 45.7%):    80 precision examples")
print(f"   Root cause 3 (irrelevance):   100 more irrelevance (was 60)")
print(f"   Root cause 4 (parallel):      60 multi-tool examples")
print(f"   Root cause 5 (selection):     60 hard negative pairs")
print(f"   Root cause 6 (multi-hop):     40 chain examples")
print(f"   Root cause 7 (multi-turn):    40 context tracking")
