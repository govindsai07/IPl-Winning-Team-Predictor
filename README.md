 🏏 IPL Team Winning Predictor using Machine Learning

This project aims to predict the winning team of an IPL match using historical match data and machine learning classification algorithms. It considers factors like playing teams, venue, toss results, and decisions to estimate match outcomes.

---

 📂 Dataset

- Sourced from IPL match history (Kaggle or ESPN Cricinfo)
- Key Features:
  - Team1, Team2
  - Toss Winner & Toss Decision
  - Venue, City, Season
  - Match Winner, Match Result Type

---

 🧹 Data Preprocessing

- Handled missing values and irrelevant columns
- Encoded categorical variables like team names and venues
- Simplified team names for consistency
- Focused on matches with a clear winner (excluded ties/no result)

---

 📊 Exploratory Data Analysis (EDA)

- Win percentages by team
- Toss impact on match results
- Venue-wise team performance
- Head-to-head team analysis

---

 🧠 Machine Learning Models

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machine (SVM)

---

 📈 Model Evaluation

- Accuracy Score  
- Confusion Matrix  
- Classification Report  
- Cross-validation to ensure generalizability

---

⚙️ Technologies Used

Python,Pandas, NumPy,Scikit-learn,Matplotlib, Seaborn


 🚀 How to Run

 Run the model
python ipl_predictor.py

🙌 Acknowledgements

IPL dataset from Kaggle / ESPN Cricinfo

Scikit-learn for classification models

Matplotlib & Seaborn for data visualization
