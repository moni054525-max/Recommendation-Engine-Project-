from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(data_dir):
    data_dir = Path(data_dir)
    students = pd.read_csv(data_dir / "students.csv").fillna("")
    internships = pd.read_csv(data_dir / "internships.csv").fillna("")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    return students, internships, ratings

def _text_similarity(student, internships):
    student_text = " ".join([
        str(student.get("skills", "")), str(student.get("interests", "")),
        str(student.get("preferred_domain", "")), str(student.get("preferred_location", ""))
    ])
    internship_texts = (
        internships["required_skills"].astype(str) + " " +
        internships["domain"].astype(str) + " " +
        internships["location"].astype(str) + " " +
        internships["description"].astype(str)
    ).tolist()
    corpus = [student_text] + internship_texts
    matrix = TfidfVectorizer(stop_words="english").fit_transform(corpus)
    return cosine_similarity(matrix[0:1], matrix[1:]).ravel()

def _collaborative_scores(student_id, internships, ratings):
    item_ids = internships["internship_id"].astype(str).tolist()
    matrix = ratings.pivot_table(index="student_id", columns="internship_id", values="rating", aggfunc="mean")
    matrix = matrix.reindex(columns=item_ids)
    if student_id not in matrix.index or matrix.drop(index=student_id, errors="ignore").empty:
        return np.zeros(len(item_ids))
    user_vectors = matrix.fillna(0).to_numpy(dtype=float)
    ids = list(matrix.index)
    target_idx = ids.index(student_id)
    target = user_vectors[target_idx]
    sims = cosine_similarity(user_vectors)[target_idx]
    sims[target_idx] = 0
    weighted = np.zeros(len(item_ids))
    weights = np.zeros(len(item_ids))
    for i, sim in enumerate(sims):
        if sim <= 0:
            continue
        observed = matrix.iloc[i].notna().to_numpy()
        weighted += np.where(observed, user_vectors[i] * sim, 0)
        weights += observed * sim
    scores = np.divide(weighted, weights, out=np.zeros_like(weighted), where=weights > 0)
    # Normalize 1–5 ratings to 0–1. Unrated items remain neutral/zero.
    return np.clip(scores / 5.0, 0, 1)

def recommend_for_student(student_id, students, internships, ratings, method="Hybrid", top_k=5):
    match = students[students["student_id"].astype(str) == str(student_id)]
    if match.empty:
        return pd.DataFrame()
    student = match.iloc[0]
    content = _text_similarity(student, internships)
    collaborative = _collaborative_scores(str(student_id), internships, ratings)
    if method == "Content-Based":
        scores = content
    elif method == "Collaborative":
        scores = collaborative
    else:
        # Give content-based signals more weight for sparse educational demo data.
        scores = 0.65 * content + 0.35 * collaborative
    result = internships.copy()
    result["score"] = np.round(scores, 3)
    return result.sort_values("score", ascending=False).head(int(top_k)).reset_index(drop=True)

def add_student(csv_path, student_id, name, skills, interests, domain, location):
    student_id, name = student_id.strip(), name.strip()
    if not student_id or not name:
        return False, "Student ID and name are required."
    path = Path(csv_path)
    df = pd.read_csv(path).fillna("")
    if student_id in df["student_id"].astype(str).values:
        return False, "That student ID already exists."
    row = {
        "student_id": student_id, "name": name, "skills": skills,
        "interests": interests, "preferred_domain": domain,
        "preferred_location": location
    }
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    return True, "Student profile saved."
