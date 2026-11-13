import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from recommender import recommend_for_student

def calculate_metrics(students, internships, ratings, k=3):
    if ratings.empty:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "rmse": 0.0, "mae": 0.0}
    relevant_total = 0
    hits_total = 0
    precisions, recalls = [], []
    actual, predicted = [], []
    for _, row in ratings.iterrows():
        actual.append(float(row["rating"]))
        # Simple transparent baseline for rating prediction: student's mean rating.
        user_rows = ratings[ratings["student_id"] == row["student_id"]]
        predicted.append(float(user_rows["rating"].mean()))
    for sid in students["student_id"].astype(str):
        recs = recommend_for_student(sid, students, internships, ratings, "Hybrid", k)
        relevant = set(ratings[(ratings["student_id"].astype(str) == sid) & (ratings["rating"] >= 4)]["internship_id"].astype(str))
        recommended = set(recs["internship_id"].astype(str))
        hits = len(relevant & recommended)
        relevant_total += len(relevant)
        hits_total += hits
        precisions.append(hits / k if k else 0)
        recalls.append(hits / len(relevant) if relevant else 0)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted))) if actual else 0.0
    mae = float(mean_absolute_error(actual, predicted)) if actual else 0.0
    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "rmse": rmse, "mae": mae
    }
