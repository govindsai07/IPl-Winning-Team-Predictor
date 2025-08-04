# 1. Importing Necessary Libraries
import pandas as pd
import numpy as np

# 2. Mount Google Drive and Load Dataset (For Colab Users)
# from google.colab import drive
# drive.mount('/content/drive')

data = pd.read_csv('/content/ipl_colab.csv')
print(f"Dataset successfully Imported of Shape : {data.shape}")

# 3. Exploratory Data Analysis
data.head()
data.describe()
data.info()
data.nunique()
data.dtypes

# 4. Data Cleaning - Removing Irrelevant Columns
irrelevant = ['mid', 'date', 'venue','batsman', 'bowler', 'striker', 'non-striker']
print(f'Before Removing Irrelevant Columns : {data.shape}')
data = data.drop(irrelevant, axis=1)
print(f'After Removing Irrelevant Columns : {data.shape}')
data.head()

# 5. Keeping only Consistent Teams
const_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
              'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
              'Delhi Daredevils', 'Sunrisers Hyderabad']
data = data[(data['batting_team'].isin(const_teams)) & (data['bowling_team'].isin(const_teams))]

# 6. Remove First 5 Overs
data = data[data['overs'] >= 5.0]

# 7. Select only numeric columns for correlation
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# 8. Correlation Matrix
from seaborn import heatmap
import matplotlib.pyplot as plt
heatmap(data=numeric_data.corr(), annot=True)
plt.show()

# 9. Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ['batting_team', 'bowling_team']:
  data[col] = le.fit_transform(data[col])
data.head()

# 10. One Hot Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
columnTransformer = ColumnTransformer([
    ('encoder', OneHotEncoder(), [0, 1])
], remainder='passthrough')
data = np.array(columnTransformer.fit_transform(data))

# 11. Create DataFrame with Transformed Columns
cols = ['batting_team_Chennai Super Kings', 'batting_team_Delhi Daredevils', 'batting_team_Kings XI Punjab',
        'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals',
        'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad',
        'bowling_team_Chennai Super Kings', 'bowling_team_Delhi Daredevils', 'bowling_team_Kings XI Punjab',
        'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals',
        'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad',
        'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'total']
df = pd.DataFrame(data, columns=cols)

# 12. Train-Test Split
from sklearn.model_selection import train_test_split
features = df.drop(['total'], axis=1)
labels = df['total']
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, shuffle=True)
print(f"Training Set : {train_features.shape}\nTesting Set : {test_features.shape}")

# 13. Train Models and Evaluate
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
models = dict()

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
tree.fit(train_features, train_labels)
print("Decision Tree Test Score:", tree.score(test_features, test_labels))
models["tree"] = str(tree.score(test_features, test_labels) * 100)

# Linear Regression
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(train_features, train_labels)
print("Linear Regression Test Score:", linreg.score(test_features, test_labels))
models["linreg"] = str(linreg.score(test_features, test_labels) * 100)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(train_features, train_labels)
print("Random Forest Test Score:", forest.score(test_features, test_labels))
models["forest"] = str(forest.score(test_features, test_labels) * 100)

# Lasso
from sklearn.linear_model import LassoCV
lasso = LassoCV()
lasso.fit(train_features, train_labels)
print("Lasso Test Score:", lasso.score(test_features, test_labels))
models["lasso"] = str(lasso.score(test_features, test_labels) * 100)

# SVM
from sklearn.svm import SVR
svm = SVR()
svm.fit(train_features, train_labels)
print("SVM Test Score:", svm.score(test_features, test_labels))
models["svm"] = str(svm.score(test_features, test_labels) * 100)

# Neural Network
from sklearn.neural_network import MLPRegressor
neural_net = MLPRegressor(activation='logistic', max_iter=500)
neural_net.fit(train_features, train_labels)
print("Neural Net Test Score:", neural_net.score(test_features, test_labels))
models["neural_net"] = str(neural_net.score(test_features, test_labels) * 100)

# 14. Barplot for Best Model
from seaborn import barplot
model_names = list(models.keys())
accuracy = list(map(float, models.values()))
barplot(x=model_names, y=accuracy)
plt.show()

# 15. Prediction Function using Random Forest
def predict_score(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5, model=forest):
    teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 'Kolkata Knight Riders',
             'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad']
    prediction_array = [1 if batting_team == team else 0 for team in teams] + \
                       [1 if bowling_team == team else 0 for team in teams] + \
                       [runs, wickets, overs, runs_last_5, wickets_last_5]
    prediction_array = np.array([prediction_array])
    return int(round(model.predict(prediction_array)[0]))

# 16. Example Test
batting_team='Delhi Daredevils'
bowling_team='Chennai Super Kings'
score = predict_score(batting_team, bowling_team, overs=10.2, runs=68, wickets=3, runs_last_5=29, wickets_last_5=1)
print(f'Predicted Score : {score} || Actual Score : 147')

# 17. Export Best Model
from joblib import dump
dump(forest, "forest_model.pkl")
dump(tree, "tree_model.pkl")
dump(neural_net, "neural_nets_model.pkl")
