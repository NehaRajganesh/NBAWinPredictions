import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


df = pd.read_csv("final_data.csv")
#print(df.head())
y = df['Outcome'] # let y be the outcome (Win=1/lose=0)
x = df.drop(columns=['Outcome']) # x is all the features, drop the outcome column

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# visualization confusion matrix
y_pred = model.predict(x_test)

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
disp.ax_.set_title('Confusion Matrix (Logistic Regression)')
plt.show()

