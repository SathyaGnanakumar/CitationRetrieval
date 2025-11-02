"""
CiteME Benchmark Example Verifier
----------------------------------------------------------
Evaluates candidate examples against the CiteME criteria:
1. Attributable
2. Unambiguous
3. Non-Trivial
4. Reasonable
5. GPT-4o Closed-Book Filtering (optional)
"""

import csv, json, time, os
import openai

# ---------------- MODEL CONFIGURATION ----------------
MODEL = "gpt-4o"
TEMPERATURE = 0.5
N_TRIES = 5
DELAY = 1.2

INPUT_FILE = "data.csv"          # columns: id,excerpt,target_paper_title,target_paper_url,source_paper_title,source_paper_url,year,split
OUTPUT_JSON = "verified_examples.json"
OUTPUT_CSV = "verified_examples.csv"

PERFORM_FILTERING = True               # skip closed-book filtering if False
# ------------------------------------------------

def ask_gpt(system, user):
    """Send a request to GPT-4o and return its response text."""
    try:
        response = openai.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        content = response.choices[0].message.content.strip()
        print(f"💬 API call succeeded (tokens used: {response.usage.total_tokens})")
        return content
    except Exception as e:
        print("⚠️ API error:", e)
        return ""


def evaluate_criteria(example):
    """Ask GPT-4o to evaluate the example on CiteME’s 4 human curation criteria."""
    excerpt = example["excerpt"]
    correct = f"{example['target_paper_title']} ({example['year']})"

    system = (
        "You are an expert ML researcher applying the CiteME benchmark curation criteria. "
        "Evaluate whether the provided excerpt satisfies all four of the following:\n"
        "1. Attributable: The citation directly supports the claim.\n"
        "2. Unambiguous: Only one paper could reasonably be cited.\n"
        "3. Non-Trivial: Does not mention authors, acronyms, or titles explicitly.\n"
        "4. Reasonable: Clear, readable, and contextually sound.\n\n"
        "Respond ONLY with valid JSON using exactly these boolean keys:\n"
        '{"Attributable": true/false, "Unambiguous": true/false, '
        '"NonTrivial": true/false, "Reasonable": true/false}. '
        "Do not include explanations outside the JSON object."
    )

    user = f"Excerpt: {excerpt}\n\nProposed Citation: {correct}"

    raw = ask_gpt(system, user)
    if not raw:
        return {"Attributable": None, "Unambiguous": None, "NonTrivial": None, "Reasonable": None}

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        json_str = raw[start:end]
        return json.loads(json_str)
    except Exception as e:
        print(f"⚠️ Error parsing JSON: {e}\nResponse was:\n{raw}\n")
        return {"Attributable": None, "Unambiguous": None, "NonTrivial": None, "Reasonable": None}


def filtering_stage(example):
    """Implements GPT-4o closed-book filtering (remove memorized examples)."""
    excerpt = example["excerpt"]

    print(f"\n🔍 Filtering example {example['id']} ...")
    for i in range(N_TRIES):
        response = ask_gpt(
            "You are a closed-book scientific assistant. You cannot use the internet or external tools. "
            "Based only on your internal knowledge, identify which paper this excerpt most likely refers to.",
            excerpt
        )
        if not response:
            continue
        print(f"Run {i+1}: {response[:120]}...")
        if example["target_paper_title"].lower() in response.lower():
            print("✅ Model recalled it — removing.")
            return False
        time.sleep(DELAY)
    print("❌ Model failed to recall — keep.")
    return True


def main():
    results = []

    with open(INPUT_FILE, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(f"\n🧠 Evaluating example {row['id']} ...")
            row_results = evaluate_criteria(row)
            passes_all = all(row_results.values())

            keep = False
            if passes_all:
                if PERFORM_FILTERING:
                    keep = filtering_stage(row)
                else:
                    keep = True

            # Only append verified examples that pass all criteria and filtering
            if passes_all and keep:
                row.update(row_results)
                row["PassesAll"] = True
                row["KeepAfterFiltering"] = True
                results.append(row)

    # Save verified results only
    if results:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=2, ensure_ascii=False)

        with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as cf:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print("\n-------------------- SUMMARY --------------------")
    print(f"Total verified examples: {len(results)}")
    print(f"Saved to: {OUTPUT_JSON} and {OUTPUT_CSV}")


if __name__ == "__main__":
    print(f"🧠 Using Python: {os.sys.executable}")
    print(f"🔑 API key loaded: {'OPENAI_API_KEY' in os.environ}")
    print(f"📦 openai version: {openai.__version__}\n")
    main()
