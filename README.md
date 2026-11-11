# Career Compass

A career path recommender for first-year students. A student rates 7 skills
and interests from 1 to 5, and the program suggests the best-matching tech
domain with a match percentage for each domain.

## How it works
- make.py creates careers.csv (300 synthetic students, 5 domains)
- main.py loads the data, trains a KNN model (30 neighbours) and asks the 7 questions
- app.py is the same model with a Streamlit web page (sliders and a button)

## Libraries
NumPy, Pandas, Scikit-learn (and Streamlit for the web page)

## Run
    pip install numpy pandas scikit-learn streamlit
    python make.py
    python main.py
    streamlit run app.py

## Limits
- The dataset is synthetic, so this is a prototype of the idea.
- AI / ML and Data Science overlap, so the model sometimes mixes them up.
- Future work: collect real student data and compare other models.
