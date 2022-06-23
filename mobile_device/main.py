import time
from pydantic import BaseModel
import requests



class Record(BaseModel):
    attitude_roll	: float
    attitude_pitch	: float
    attitude_yaw	: float
    userAcceleration_x	: float
    userAcceleration_y	: float
    userAcceleration_z	: float
    act	: float
    id	: float
    weight	: float
    height	: float
    age	: float
    gender	: float
    trial	: float

if __name__ == "__main__":
    
    # device = open('motion_final_part_1.csv', 'r')
    device = open('test.csv', 'r')
    Smart_device_datastream = device.readlines()

    count = 0 

    for data in Smart_device_datastream:
      count += 1
      time.sleep(2)
      temp_data = data.split(',')
      record_data = {
      "a":float(temp_data[0]),
      # "attitude_roll":float(temp_data[0]),
      # "attitude_pitch":float(temp_data[1]),
      # "attitude_yaw":float(temp_data[2]),
      # "userAcceleration_x":float(temp_data[3]),
      # "userAcceleration_y":float(temp_data[4]),
      # "userAcceleration_z":float(temp_data[5]),
      # "act":float(temp_data[6]),
      # "id":float(temp_data[7]),
      # "weight":float(temp_data[8]),
      # "height":float(temp_data[9]),
      # "age":float(temp_data[10]),
      # "gender":float(temp_data[11]),
      # "trial":float(temp_data[12])
      }
      url = 'http://127.0.0.1:8000/stream'
      data = {'attitude_roll': 0}
      headers = {'Content-Type': 'application/json','accept':'application/json'}
      response = requests.post(url=url, json=record_data, headers=headers)
