import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler   
class DiabetesDataset:
    # Ham khởi tạo
    def __init__(self,file_path):
        self.file_path=file_path
        self.data=None
        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None

    # Doc du lieu
    def load_data(self):
        self.data=pd.read_csv(self.file_path)
        
    # Tien xu ly du lieu
    def preprocess(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first")
        
        # Scale the features
        scaler = StandardScaler()
        columns_to_scale = self.data.columns.drop('Outcome')
        self.data[columns_to_scale] = scaler.fit_transform(self.data[columns_to_scale])
        
        X=self.data.drop('Outcome',axis=1)
        y=self.data['Outcome']
    # Chuan hoa  data
        scaler= StandardScaler()
        X_scaled=scaler.fit_transform(X)
    # chia du lieu thanh 80 train 20 test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(  
            X_scaled, y, test_size=0.2, random_state=42)
