import numpy as np
import pandas as pd

np.random.seed(42)

FEATURES = ["python", "math", "creativity", "problem_solving",
            "security", "web_design", "data_analysis"]

PROFILES = {
    "AI / ML":         [4.5, 4.5, 3.0, 4.5, 2.0, 2.0, 4.5],
    "Data Science":    [4.0, 4.5, 3.0, 4.0, 2.0, 2.0, 5.0],
    "Cybersecurity":   [4.0, 3.0, 2.5, 4.5, 5.0, 3.0, 2.0],
    "Web Development": [4.0, 2.5, 3.5, 3.5, 2.5, 5.0, 2.0],
    "UI/UX Design":    [2.5, 2.0, 5.0, 3.0, 2.0, 4.5, 2.5],
}

STUDENTS_PER_DOMAIN = 60
SPREAD = 0.9

rows = []
for domain, typical in PROFILES.items():
    ratings = np.random.normal(loc=typical, scale=SPREAD,
                               size=(STUDENTS_PER_DOMAIN, len(FEATURES)))
    ratings = np.clip(np.round(ratings), 1, 5).astype(int)
    for r in ratings:
        rows.append(list(r) + [domain])

df = pd.DataFrame(rows, columns=FEATURES + ["career"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("careers.csv", index=False)
print("Created careers.csv with", len(df), "rows")