import numpy as np

class DiabetesApp:
    def __init__(self, model):
        self.model = model
    
    def run(self):
        # Ví dụ nhập dữ liệu từ console
        print("Nhập thông số của bệnh nhân:")
        Pregnancies = float(input("Số lần mang thai: "))
        Glucose = float(input("Glucose: "))
        BloodPressure = float(input("BloodPressure: "))
        SkinThickness = float(input("SkinThickness: "))
        Insulin = float(input("Insulin: "))
        BMI = float(input("BMI: "))
        DPF = float(input("Diabetes Pedigree Function: "))
        Age = float(input("Age: "))
        
        user_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]])
        risk_prob = self.model.predict(user_data)
        
        if risk_prob > 0.5:
            print(f"Nguy cơ tiểu đường cao: {risk_prob:.2f}")
        else:
            print(f"Nguy cơ thấp: {risk_prob:.2f}")
