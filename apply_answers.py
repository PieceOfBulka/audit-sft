import json, glob, os

MAIN = "data/claude_answers.json"

def main():
    data = json.load(open(MAIN, encoding="utf-8"))
    applied = 0
    for pf in sorted(glob.glob("data/patches/*.json")):
        patch = json.load(open(pf, encoding="utf-8"))
        for k, v in patch.items():
            i = int(k)
            if not v.strip():
                continue
            data[i]["answer"] = v
            applied += 1
    json.dump(data, open(MAIN, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    empty = sum(1 for x in data if not x.get("answer", "").strip())
    print(f"applied {applied} answers; remaining empty: {empty}")

if __name__ == "__main__":
    main()
