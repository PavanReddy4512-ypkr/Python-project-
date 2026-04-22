import pandas as pd

# Load dataset
data = pd.read_csv("spam.csv")

# Input and Output
X = data["message"]
y = data["label"]

# Convert text to numbers
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_vector = cv.fit_transform(X)

# Train model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_vector, y)

# Save model
import pickle
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(cv, open("vectorizer.pkl","wb"))

print("Training Completed")