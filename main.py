# This is a sample Python script.
import pandas as pd
df = pd.read_csv('student_performance.csv')
import pandas as pd
df = pd.read_csv('student_performance.csv')
print(df.head())
#convert categorical data into numerical data A is the best grade F is bad grade
grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}    #dictionary
df.isnull()
df['grade_num'] = df['grade'].map(grade_map)
print(df['grade_num'])
import matplotlib.pyplot as plt
import seaborn as sns
df.isnull()
df['grade_num'] = df['grade'].map(grade_map)
print(df['grade_num'])
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df.sample(1000), x='attendance_percentage', y='total_score', hue='grade')
plt.title('Impact of Attendance on Total Score')
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='grade', y='weekly_self_study_hours', order=['A', 'B', 'C', 'D', 'F'])
plt.title('Weekly Study Hours by Grade')
plt.show()
numeric_columns = df[['weekly_self_study_hours', 'attendance_percentage', 'class_participation', 'total_score']]
correlation = numeric_columns.corr()
sns.heatmap(correlation, annot=True, cmap='Blues')
plt.title('Pattern Map: Which factors correlate most?')
plt.show()
X = df[['weekly_self_study_hours', 'attendance_percentage', 'class_participation']]
y = df['grade_num']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#logistic regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
# 2. Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
# 3. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
svm = SVC(kernel='linear')
svm.fit(X_train[:5000], y_train[:5000])
print("--- Model Accuracy Results ---")
print(f"Logistic Regression: {accuracy_score(y_test, lr.predict(X_test)):.2%}")
print(f"Decision Tree: {accuracy_score(y_test, dt.predict(X_test)):.2%}")
print(f"Naive Bayes:{accuracy_score(y_test, nb.predict(X_test)):.2%}")
print(f"SVM (Linear):{accuracy_score(y_test[:1000], svm.predict(X_test[:1000])):.2%}")
final_predictions = lr.predict(X_test)
# 2. Create a clean results table
results_df = X_test.copy()
results_df['Predicted_Grade_Num'] = final_predictions
# Map numbers back to Letters for the report
reverse_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
results_df['Predicted_Grade'] = results_df['Predicted_Grade_Num'].map(reverse_map)
at_risk_students = results_df[results_df['Predicted_Grade'].isin(['D', 'F'])]
print(f"\nTotal Students Identified as 'At-Risk': {len(at_risk_students)}")
print("Top 5 people who need attention")
print(at_risk_students.head())
plt.figure(figsize=(10, 6))
sns.barplot(x='Predicted_Grade', y='attendance_percentage', data=results_df, order=['A', 'B', 'C', 'D', 'F'])
plt.title('Academic Trend: Attendance vs Predicted Grade')
plt.ylabel('Average Attendance %')
plt.show()