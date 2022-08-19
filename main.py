# -*- coding: utf-8 -*-
"""


@author: Irfan
"""

# importing libraries
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from markupsafe import escape

# flask app
app = Flask(__name__)
df_sample = pd.read_excel(r"C:/Users/Irfan/Desktop/project69/sample data.xlsx")

# loading model
model = pickle.load(open('C:/Users/Irfan/Desktop/project69/model.pkl', 'rb'))

@app.route('/')
def index():
    test_name = sorted(df_sample['Test_Name'].unique())
    sample = sorted(df_sample['Sample'].unique())
    # way_of_storage name dont need to be here used radio buttons insted
    schedule = sorted(df_sample['Cut_off_Schedule'].unique())
    traffic = sorted(df_sample['Traffic_Conditions'].unique())
    return render_template('index.html', test_name= test_name, sample= sample, 
                            # way_of_storage= way_of_storage, 
                            schedule= schedule, traffic= traffic)

@app.route('/prediction' ,methods = ["GET", "POST"])
def prediction():
    test_name = request.form.get('testname')
    sample = request.form.get('samplename')
    way_of_storage = request.form.get('newradio')
    test_booking_time = request.form.get('test_booking_time_hh_mm')
    scheduled_sample_collection = request.form.get('scheduled_sample_collection_time_hh_mm')
    cut_off_schedule = request.form.get('schedule')
    cut_off_time = request.form.get('cut_off_time_hh_mm')
    # Agent_ID = int(request.form.get('Agent_ID'))
    traffic_conditions = request.form.get('traffic')
    agents_location = request.form.get('agent_location_km')
    # time_taken_to_reach_patient_mm = request.form.get('time_taken_to_reach_patient_mm')
    time_for_sample_collection_mm = request.form.get('time_for_sample_collection_mm')
    lab_location = request.form.get('lab_location_km')
    time_taken_to_reach_lab_mm = request.form.get('time_taken_to_reach_lab_mm')

    data = {
        'Test_Name': test_name,
        'Sample': sample,
        'Way_Of_Storage_Of_Sample': way_of_storage,
        'Test_Booking_Time_HH_MM' : test_booking_time,
        'Scheduled_Sample_Collection_Time_HH_MM' : scheduled_sample_collection,
        'Cut_off_Schedule': cut_off_schedule,
        'Cut_off_time_HH_MM': cut_off_time,
        'Traffic_Conditions': traffic_conditions,
        'Agent_Location_KM': agents_location,
        # 'Time_Taken_To_Reach_Patient_MM' : time_taken_to_reach_patient_mm,
        'Time_For_Sample_Collection_MM': time_for_sample_collection_mm,
        'Lab_Location_KM': lab_location,
        'Time_Taken_To_Reach_Lab_MM': time_taken_to_reach_lab_mm
        }
    
    print('data:' , data)
    features = pd.DataFrame(data, index=[0])
    print('Features:' , features)
    
    print('predictiND MODEL')
    prediction = model.predict(features)
    print('prediction:' , prediction)
    
    if prediction == 'Y':
        return render_template("predict.html", output='YES')
    else:
        return render_template("predict.html", output='NO')

if __name__ == "__main__":
    app.run(debug=True)