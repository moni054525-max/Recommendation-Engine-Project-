import streamlit as st
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from recommender import load_data, recommend_for_student, add_student
from evaluation import calculate_metrics

st.set_page_config(page_title="AI Internship Recommendation Engine", page_icon="🎯", layout="wide")
st.title("🎯 AI Internship Recommendation Engine")
st.caption("AI-SS-002 | Student Support & Internship Management")
st.write("Find suitable internships using student skills, interests, and collaborative preferences.")

students, internships, ratings = load_data(ROOT / "data")
tab1, tab2, tab3, tab4 = st.tabs(["🎓 Recommendations", "👤 Student Profiles", "💼 Internship Database", "📊 Evaluation"])

with tab1:
    st.subheader("Personalized internship recommendations")
    student_options = {f'{r["student_id"]} — {r["name"]}': r["student_id"] for _, r in students.iterrows()}
    selected_label = st.selectbox("Choose a student", list(student_options.keys()))
    method = st.selectbox("Recommendation method", ["Hybrid", "Content-Based", "Collaborative"])
    top_k = st.slider("Number of recommendations", 1, 10, 5)
    sid = student_options[selected_label]
    if st.button("Generate recommendations", type="primary"):
        results = recommend_for_student(sid, students, internships, ratings, method=method, top_k=top_k)
        if results.empty:
            st.info("No recommendations available for this student yet.")
        else:
            st.dataframe(results[["internship_id", "title", "company", "domain", "required_skills", "location", "score"]].rename(columns={"score":"match_score"}), use_container_width=True, hide_index=True)
            st.bar_chart(results.set_index("title")["score"])
            st.success("Recommendations generated successfully.")

with tab2:
    st.subheader("Student profile management")
    st.dataframe(students, use_container_width=True, hide_index=True)
    with st.expander("Add a new student profile"):
        with st.form("add_student_form"):
            sid_new = st.text_input("Student ID (unique)")
            name = st.text_input("Full name")
            skills = st.text_input("Skills (comma-separated)", placeholder="Python, Pandas, SQL")
            interests = st.text_input("Interests (comma-separated)", placeholder="Data Science, AI")
            domain = st.text_input("Preferred domain", "Data Science")
            location = st.text_input("Preferred location", "Remote")
            submitted = st.form_submit_button("Save student")
            if submitted:
                ok, message = add_student(ROOT / "data" / "students.csv", sid_new, name, skills, interests, domain, location)
                if ok:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

with tab3:
    st.subheader("Internship database")
    st.dataframe(internships, use_container_width=True, hide_index=True)
    st.caption("Internship information is loaded from data/internships.csv. Edit the CSV to add or update opportunities.")

with tab4:
    st.subheader("Recommendation evaluation")
    st.write("Metrics are calculated using the sample student–internship ratings as held-out relevance feedback.")
    metrics = calculate_metrics(students, internships, ratings)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision@3", f'{metrics["precision_at_k"]:.3f}')
    c2.metric("Recall@3", f'{metrics["recall_at_k"]:.3f}')
    c3.metric("RMSE", f'{metrics["rmse"]:.3f}')
    c4.metric("MAE", f'{metrics["mae"]:.3f}')
    st.info("These values are illustrative for the included sample dataset. Evaluate with a larger, representative dataset for meaningful results.")
