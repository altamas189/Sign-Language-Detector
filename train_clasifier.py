import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
label_map = data_dict.get('label_map', None)  # safe load

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)#80-20 split 

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
score = accuracy_score(y_pred, y_test)

print(f"\n Accuracy: {score*100:.2f}%")
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and label map
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_map': label_map}, f)

print("\n Model saved to 'model.p'")
