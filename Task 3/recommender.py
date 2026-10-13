"""
Project 3 - AI Recommendation Logic: Tech Stack Recommender
DecodeLabs | Content-Based Filtering with TF-IDF + Cosine Similarity

Pipeline (Input -> Process -> Output):
    1. Ingestion : capture >= 3 user skills
    2. Scoring   : cosine similarity of user vector vs every job-role vector
    3. Sorting   : descending by score
    4. Filtering : keep Top-N

Usage:
    python recommender.py                       # interactive
    python recommender.py Python "Cloud Computing" Automation   # one-shot
"""

import csv
import math
import sys
from pathlib import Path

DATASET = Path(__file__).with_name("raw_skills.csv")
MIN_INPUTS = 3
TOP_N = 3

# Shared vocabulary guard: map common alternate names to the dataset's canonical
# skill names so user features and item features use the exact same vocabulary.
ALIASES = {
    "cloud": "cloud computing",
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "machine learning",
    "js": "javascript",
    "node": "node.js",
    "nodejs": "node.js",
    "cicd": "ci/cd",
    "ci cd": "ci/cd",
    "k8s": "kubernetes",
    "sklearn": "scikit-learn",
    "web design": "css",
    "frontend development": "react",
    "api": "apis",
    "rest": "apis",
    "dsa": "data structures",
    "algorithm": "algorithms",
    "natural language processing": "nlp",
    "powerbi": "tableau",
    "power bi": "tableau",
    "scripting": "automation",
    "shell": "bash",
    "tf": "tensorflow",
}


# ---------------------------------------------------------------- data loading
def load_roles(path=DATASET):
    roles = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            skills = [s.strip().lower() for s in row["skills"].split(";") if s.strip()]
            roles.append({"role": row["role"], "demand": int(row["demand"]), "skills": skills})
    return roles


def normalize(skill):
    s = skill.strip().lower()
    return ALIASES.get(s, s)


# ------------------------------------------------------------------- TF-IDF
def build_vocabulary(roles):
    return sorted({s for r in roles for s in r["skills"]})


def compute_idf(roles, vocab):
    n = len(roles)
    idf = {}
    for term in vocab:
        df = sum(1 for r in roles if term in r["skills"])
        idf[term] = math.log(n / df)          # IDF = log(total docs / docs with term)
    return idf


def tfidf_vector(skills, vocab, idf):
    """TF = count(term) / total terms; weight = TF * IDF; vector over shared vocab."""
    total = len(skills)
    if total == 0:
        return [0.0] * len(vocab)
    return [(skills.count(t) / total) * idf[t] for t in vocab]


# ------------------------------------------------------------ cosine similarity
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:      # cold-start guard: zero vector -> no similarity
        return 0.0
    return dot / (na * nb)


# ------------------------------------------------------------------- pipeline
def recommend(user_skills, roles, top_n=TOP_N):
    vocab = build_vocabulary(roles)
    idf = compute_idf(roles, vocab)

    # Step 1: Ingestion - normalize, de-duplicate, keep only known vocabulary
    cleaned = []
    for s in user_skills:
        n = normalize(s)
        if n not in cleaned:
            cleaned.append(n)
    known = [s for s in cleaned if s in vocab]
    unknown = [s for s in cleaned if s not in vocab]
    user_vec = tfidf_vector(known, vocab, idf)

    # Cold start: nothing recognised -> trending fallback (global demand)
    if not known:
        trending = sorted(roles, key=lambda r: r["demand"], reverse=True)[:top_n]
        return [(r["role"], 0.0, []) for r in trending], known, unknown, True

    # Step 2: Scoring
    scored = []
    for r in roles:
        score = cosine_similarity(user_vec, tfidf_vector(r["skills"], vocab, idf))
        matched = [s for s in known if s in r["skills"]]
        scored.append((r["role"], score, matched, r["demand"]))

    # Step 3: Sorting (score desc; demand breaks ties)
    scored.sort(key=lambda x: (x[1], x[3]), reverse=True)

    # Step 4: Filtering (Top-N)
    return [(n, s, m) for n, s, m, _ in scored[:top_n]], known, unknown, False


# ------------------------------------------------------------------------ UI
def show(results, known, unknown, cold):
    print("\n" + "=" * 56)
    print("  TOP CAREER MATCHES")
    print("=" * 56)
    if unknown:
        print(f"  (Ignored unknown skills: {', '.join(unknown)})")
    if cold:
        print("  No recognised skills - showing trending roles instead.\n")
    for i, (role, score, matched) in enumerate(results, 1):
        print(f"  {i}. {role:<30} {score * 100:5.1f}% match")
        if matched:
            print(f"     matched skills: {', '.join(matched)}")
    print("=" * 56)


def collect_inputs(roles):
    vocab = build_vocabulary(roles)
    print("Tech Stack Recommender  |  DecodeLabs Project 3")
    print("Available skills:\n  " + ", ".join(vocab) + "\n")
    skills = []
    while len(skills) < MIN_INPUTS:
        s = input(f"Enter skill #{len(skills) + 1}: ").strip()
        if s:
            skills.append(s)
    return skills


def main():
    roles = load_roles()
    if len(sys.argv) > 1:
        skills = sys.argv[1:]
    else:
        skills = collect_inputs(roles)
    if len(skills) < MIN_INPUTS:
        sys.exit(f"Please provide at least {MIN_INPUTS} skills.")
    results, known, unknown, cold = recommend(skills, roles)
    show(results, known, unknown, cold)


if __name__ == "__main__":
    main()
