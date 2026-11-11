import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv("careers.csv")
X = data.drop("career", axis=1)   
y = data["career"]                
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=30)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
skills = {
    "AI / ML": ["Python", "Statistics", "NumPy and Pandas", "Machine Learning basics"],
    "Data Science": ["Python", "Statistics", "Pandas", "Data visualisation"],
    "Cybersecurity": ["Networking", "Linux", "Python scripting", "Ethical hacking basics"],
    "Web Development": ["HTML and CSS", "JavaScript", "Git and GitHub", "Flask or React"],
    "UI/UX Design": ["Design basics", "Figma", "User research", "Colours and fonts"],
}
st.title("Career Path Recommender")
st.write("Rate yourself from 1 (low) to 5 (high)")
python = st.slider("Python", 1, 5, 3)
math = st.slider("Maths", 1, 5, 3)
creativity = st.slider("Creativity", 1, 5, 3)
problem_solving = st.slider("Problem solving", 1, 5, 3)
security = st.slider("Security", 1, 5, 3)
web_design = st.slider("Web design", 1, 5, 3)
data_analysis = st.slider("Data analysis", 1, 5, 3)
if st.button("Get my recommendation"):
    student = pd.DataFrame(
        [[python, math, creativity, problem_solving, security, web_design, data_analysis]],
        columns=X.columns)

    probs = model.predict_proba(student)[0]

    
    best_domain = ""
    best_score = 0
    for career, p in zip(model.classes_, probs):
        st.write(career + " : " + str(round(p * 100)) + " %")
        if p > best_score:
            best_score = p
            best_domain = career

    st.subheader("Recommended domain: " + best_domain)
    st.write("Skills to learn:")
    for skill in skills[best_domain]:
        st.write("- " + skill)

st.write("Model accuracy: " + str(round(accuracy * 100)) + " %")