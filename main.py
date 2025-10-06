from DataSet import DiabetesDataset
from Model import DiabetesModel
from App import DiabetesApp
if __name__ == "__main__":
    # Bước 1: Chuẩn bị dữ liệu
    dataset = DiabetesDataset("./diabetes.csv") 
    dataset.load_data()
    dataset.preprocess()

    # Bước 2: Huấn luyện mô hình
    model = DiabetesModel()
    model.train(dataset.X_train, dataset.y_train)  

    # Bước 3: Đánh giá mô hình
    model.evaluate(dataset.X_test, dataset.y_test)  
    

     # Bước 4: Cho người dùng nhập dữ liệu
    app = DiabetesApp(model)
    app.run()
