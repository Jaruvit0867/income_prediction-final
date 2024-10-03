from flask import Flask, render_template, request
import pandas as pd
import pickle

# โหลดโมเดลที่ฝึกแล้วจากไฟล์ lung_cancer_predictor_model.pkl
with open('model/lung_cancer_predictor_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# อ่านข้อมูลจากไฟล์ survey lung cancer.csv
data = pd.read_csv('data/survey_lung_cancer.csv')

# ฟังก์ชันสำหรับแปลง 1 = No, 2 = Yes
def translate_values(value):
    if value == 1:
        return "No"
    elif value == 2:
        return "Yes"
    return value

# แปลงข้อมูลในคอลัมน์ที่ต้องการให้เป็น Yes/No
data['SMOKING'] = data['SMOKING'].apply(translate_values)
data['YELLOW_FINGERS'] = data['YELLOW_FINGERS'].apply(translate_values)
data['ANXIETY'] = data['ANXIETY'].apply(translate_values)
data['PEER_PRESSURE'] = data['PEER_PRESSURE'].apply(translate_values)
data['CHRONIC DISEASE'] = data['CHRONIC DISEASE'].apply(translate_values)
data['LUNG_CANCER'] = data['LUNG_CANCER'].apply(translate_values)

@app.route('/')
def home():
    return render_template('index.html', data=data.to_dict('records'))

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = request.form['gender']
    smoking = int(request.form['smoking'])
    yellow_fingers = int(request.form['yellow_fingers'])
    anxiety = int(request.form['anxiety'])
    peer_pressure = int(request.form['peer_pressure'])
    chronic_disease = int(request.form['chronic_disease'])

    # เตรียมข้อมูลที่จะใช้สำหรับการทำนาย
    input_data = pd.DataFrame([[age, gender, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease]],
                              columns=['AGE', 'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE'])

    # แปลงข้อมูล gender ให้เป็นแบบตัวเลข
    input_data = pd.get_dummies(input_data, columns=['GENDER'])

    # เติมคอลัมน์ที่อาจหายไป
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # ทำการทำนาย
    prediction = model.predict(input_data)[0]

    # แปลงค่าผลการทำนายเป็น Yes/No
    prediction_text = translate_values(prediction)

    if translate_values(prediction) == "YES":
        prediction_text = "คุณมีความเสี่ยงที่จะเป็นมะเร็งปอด'สูง'"
    else:
        prediction_text = "คุณมีความเสี่ยงที่จะเป็นมะเร็งปอด'ต่ำ'"
        

    return render_template('index.html', prediction=prediction_text, data=data.to_dict('records'))

if __name__ == "__main__":
    app.run(debug=True)