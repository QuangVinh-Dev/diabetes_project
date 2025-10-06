from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class DiabetesModel:
    def __init__(self):  
        self.model = LogisticRegression(max_iter=1000)  
    
    def train(self, X_train, y_train): 
        self.model.fit(X_train, y_train)  
    
    def evaluate(self, X_test, y_test): 
        y_pred = self.model.predict(X_test)  
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
    
    def predict(self, input_data):
        prob = self.model.predict_proba(input_data)[:, 1]  
        return prob[0]