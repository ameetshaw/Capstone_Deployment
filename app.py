import numpy as np
from flask import Flask, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def get_test_data():
    test_data = pd.read_csv(r"test_data.csv")
    test_data.drop(columns=["index"], inplace=True)
    test_data = test_data.drop_duplicates(subset="device_id", keep="first")
    test_data["Total_No_events"] = StandardScaler().fit_transform(test_data[["Total_No_events"]])
    test_data["Change_in_LatLong"] = StandardScaler().fit_transform(test_data[["Change_in_LatLong"]])
    test_data["longitude_median"] = StandardScaler().fit_transform(test_data[["longitude_median"]])
    test_data["latitude_median"] = StandardScaler().fit_transform(test_data[["latitude_median"]])
    return test_data

@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')

@app.route('/age')
def age_prediction():
    test_data = get_test_data()
    age_model = pickle.load(open('age_pred.pkl', 'rb'))
    x_test = test_data.drop(
        columns=["gender", "age", "group_train", "super_category", "train_test_flag", "device_id", "phone_brand"])
    age_predict = age_model.predict(x_test.values)
    x_test["predicted_age"]= age_predict

    campain4=test_data[x_test["predicted_age"]<25]["device_id"][:15].to_list()
    campain5=test_data[x_test["predicted_age"].between(24,32)]["device_id"][15:35].to_list()
    campain6=test_data[x_test["predicted_age"]>32]["device_id"][:10].to_list()
    headings=("campaign","device_id","description")
    data_list=[]
    for i in np.unique(campain4):
        temp=(4,i,"Bundled smartphone offers for the age group [0-24]")
        data_list.append(temp)
    for i in np.unique(campain5):
        temp=(5,i,"Special offers for payment wallet offers [24-32]")
        data_list.append(temp)
    for i in np.unique(campain6):
        temp=(6,i,"Special cashback offers for Privilege Membership [32+]")
        data_list.append(temp)
    data=tuple(data_list)
    return render_template('prediction.html', headings=headings, data=data)

@app.route('/gender')
def gender_prediction():
    test_data = get_test_data()
    gender_model = pickle.load(open('gender_pred.pkl', 'rb'))
    x_test = test_data.drop(
        columns=["gender", "age", "group_train", "super_category", "train_test_flag", "device_id", "phone_brand"])
    gender_predict = gender_model.predict_proba(x_test)
    y_pred = [1 if x > 0.7 else 0 for x in gender_predict[:, 1]]
    x_test["predicted_gender"] = y_pred
    #sampled_df = x_test.sample(50)
    #print(sampled_df.dtypes)
    campain1 = test_data[x_test["predicted_gender"] == 0]["device_id"][:15].to_list()
    campain2 = test_data[x_test["predicted_gender"] == 0]["device_id"][15:35].to_list()
    campain3 = test_data[x_test["predicted_gender"] == 1]["device_id"][:10].to_list()
    data_list = []
    for i in np.unique(campain1):
        temp = (1, i, "Specific personalised fashion-related campaigns targeting female customers")
        data_list.append(temp)
    for i in np.unique(campain2):
        temp = (
        2, i, "Specific cashback offers on special days (International Women’s Day etc) targeting female customers")
        data_list.append(temp)
    for i in np.unique(campain3):
        temp = (3, i, "Personalised call and data packs targeting male customers")
        data_list.append(temp)
    data = tuple(data_list)
    headings = ("campaign", "device_id", "Description")
    return render_template('prediction.html', headings=headings, data=data)


if __name__ == '__main__':
    app.run()