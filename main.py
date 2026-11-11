import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# load the data
data = pd.read_csv("careers.csv")
X = data.drop("career", axis=1)   # the 7 ratings
y = data["career"]                # the domain

# hide 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# train the model
model = KNeighborsClassifier(n_neighbors=30)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))

# ask the student
print("Rate yourself from 1 to 5")
python = int(input("Python: "))
math = int(input("Maths: "))
creativity = int(input("Creativity: "))
problem_solving = int(input("Problem solving: "))
security = int(input("Security: "))
web_design = int(input("Web design: "))
data_analysis = int(input("Data analysis: "))

student = pd.DataFrame(
    [[python, math, creativity, problem_solving, security, web_design, data_analysis]],
    columns=X.columns)

# show the match for each domain
probs = model.predict_proba(student)[0]
for career, p in zip(model.classes_, probs):
    print(career, ":", round(p * 100), "%")