from Imports import *
import config

class LSVMClassifier:
    def __init__(self, data):
        self.model = None
        self.X , self.Y = get_data(data)
        
        
    
    def train_model(self):
        self.model = SGDClassifier(loss='hinge', max_iter=10000, penalty='l2')
        self.model.fit(self.X, self.Y)
        filename = 'finalized_model.sav'
        pickle.dump(self.model, open(filename, 'wb'))
        print("Saved model to disk")

    def load_model(self):
        filename = 'finalized_model.sav'
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)
        print("Loaded model from disk")

    def test_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y,  test_size=0.25, random_state=42)
        y_pred = self.model.predict(X_test)
        ac = accuracy_score(y_test, y_pred) * 100
        print("LSVM-Classifier Binary Set-Accuracy is ", ac)

# Assuming bin_data, X, and Y are already defined



# Creating an instance of the LSVMClassifier class
lsvm_classifier = LSVMClassifier()

# Training the model
lsvm_classifier.train_model(X, Y)

# Loading the trained model
lsvm_classifier.load_model()

# Testing the model
lsvm_classifier.test_model(X, Y)
