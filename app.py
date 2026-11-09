"""Streamlit interface: enter student details -> predicted learner level + learning plan."""
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Personalized Learning System", page_icon="🎓", layout="wide")

@st.cache_resource
def load():
    return joblib.load("best_model.joblib"), json.load(open("model_meta.json"))

model, meta = load()
labels = meta["labels"]

st.title("🎓 Personalized Learning System")
st.caption(f"Predicts a student's performance level and suggests a learning plan. "
           f"Model: **{meta['best_model']}** (trained on Kaggle *Student Performance Factors*).")

st.header("Student information")
inputs = {}

st.subheader("📊 Academic & study habits")
num_grid = st.columns(3)
for i, c in enumerate(meta["num_cols"]):
    lo, hi, med = meta["num_ranges"][c]
    label = c.replace("_", " ")
    with num_grid[i % 3]:
        inputs[c] = st.slider(label, int(lo), int(hi), int(med))

st.subheader("🏠 Background & environment")
cat_grid = st.columns(3)
for i, c in enumerate(meta["cat_cols"]):
    opts = meta["cat_options"][c]
    with cat_grid[i % 3]:
        inputs[c] = st.selectbox(c.replace("_", " "), opts,
                                 index=opts.index(meta["cat_default"][c]))

def recommend(d):
    tips = []
    if d["Hours_Studied"] < 15:
        tips.append("📚 Increase weekly study time with short, scheduled study blocks.")
    if d["Attendance"] < 80:
        tips.append("🏫 Improve class attendance - recorded sessions help for missed classes.")
    if d["Sleep_Hours"] < 7:
        tips.append("😴 Aim for 7-9 hours of sleep to improve focus and memory.")
    if d["Tutoring_Sessions"] == 0:
        tips.append("👩‍🏫 Join tutoring or peer-study sessions for difficult topics.")
    if d["Previous_Scores"] < 65:
        tips.append("🔁 Revise foundation topics with adaptive practice before new content.")
    if d["Motivation_Level"] == "Low":
        tips.append("🎯 Set small weekly goals with rewards to build motivation.")
    if d["Access_to_Resources"] == "Low":
        tips.append("💻 Use free online resources / library materials to close the resource gap.")
    if d["Internet_Access"] == "No":
        tips.append("📥 Download offline lessons and use school/library internet.")
    if d["Parental_Involvement"] == "Low":
        tips.append("👪 Share a weekly progress report with a parent or guardian.")
    return tips or ["✅ Great habits! Keep going and try advanced / enrichment material."]

if st.button("🔮 Predict performance", type="primary"):
    row = pd.DataFrame([inputs])
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)

    icon = {"Needs Support": "🔴", "On Track": "🟡", "Advanced": "🟢"}[pred]
    st.subheader(f"{icon} Predicted level: {pred}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Prediction confidence**")
        st.bar_chart(pd.Series(proba, index=classes))
    with c2:
        st.markdown("**Personalized learning plan**")
        for t in recommend(inputs):
            st.write(t)
else:
    st.info("Fill in the student details above and click **Predict performance**.")
