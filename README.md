# Personalized Learning System – Student Performance Prediction

## 1. Application
Early identification of each student's learning level (**Needs Support / On Track / Advanced**)
so the system can recommend a personalized learning plan (study time, tutoring, sleep, resources...).

## 2. Dataset (Kaggle)
**Student Performance Factors** – https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
- File: `StudentPerformanceFactors.csv` (6,607 rows, 20 columns)
- Inputs: Hours_Studied, Attendance, Parental_Involvement, Access_to_Resources, Sleep_Hours,
  Previous_Scores, Motivation_Level, Tutoring_Sessions, Teacher_Quality, etc.
- Target: `Exam_Score` -> converted into 3 levels (balanced tertiles).

## 3. Data-mining technique: **Classification**
Three+ models compared on the same split and preprocessing:
1. Logistic Regression  2. Decision Tree  3. Random Forest  4. Gradient Boosting

Evaluation: 5-fold CV accuracy, test accuracy, macro precision/recall/F1, confusion matrix.
The best model (by macro F1) is saved and used by the interface.

## 4. How to run
```bash
pip install -r requirements.txt
# download the CSV from Kaggle and put it in data/StudentPerformanceFactors.csv
python train.py            # trains 4 models, saves charts + best_model.joblib
streamlit run app.py       # opens the prediction interface in the browser
```
Optional: `pip install kaggle` then
`kaggle datasets download -d lainguyn123/student-performance-factors -p data --unzip`

## 5. Interface
Sidebar form for student details -> **Predict** -> predicted level, confidence chart,
and a rule-based personalized learning plan.
