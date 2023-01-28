from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

# Load the dataset
X, y = fetch_lfw_people(min_faces_per_person=50, resize=0.4)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create the KNN model
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
  
# Predict on dataset which model has not seen before
print(knn.predict(X_test))
