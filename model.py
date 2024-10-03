import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# อ่านข้อมูลจากไฟล์ survey lung cancer.csv
data = pd.read_csv('data/survey_lung_cancer.csv')

# สำรวจข้อมูลเพื่อดูว่ามีตัวแปรอะไรบ้าง
print(data.head())

# สมมติว่าคอลัมน์ที่ใช้เป็นฟีเจอร์สำหรับการทำนายคือ 'AGE', 'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE'
# และตัวแปรที่เราจะทำนายคือ 'LUNG_CANCER'

# เตรียมข้อมูลสำหรับการฝึกโมเดล
X = data[['AGE', 'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE']]
y = data['LUNG_CANCER']

# แปลงข้อมูลประเภทข้อความเป็นค่าตัวเลขถ้าจำเป็น
X = pd.get_dummies(X, columns=['GENDER'])

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# บันทึกโมเดลลงไฟล์ .pkl
with open('model/lung_cancer_predictor_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# ทดสอบโมเดล
accuracy = model.score(X_test, y_test)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")
