"""
This file converts ros2 bag messages to csv along with calculating
and changing the lat and long for gps if the gps condition = "spoofed"
on the metadata.

To use:
1) In the if __name__ ... Change the directory to your specific directory
   for the /dataset and csv location.
  -> dataset folder: example /drone_basics/dataset
  -> finished csv location: example /home/ros_csv

2) source ros2 and .venv.
 -> run . ./source_and_start_ros2.sh
 -> Should source and start .venv for ros2

3) Run python3 <filename>.py

"""

import csv
from pathlib import Path
import json
import pandas as pd

from data_collection_module.spoof_types import RunGPSCondition 

#Ros2 and px4 translation items
from rosbag2_py  import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message

from px4_msgs.msg import SensorGps
from px4_msgs.msg import VehicleGlobalPosition
from px4_msgs.msg import VehicleOdometry

"""
------------- CLASS FOR READING AND CONVERTING ROS 2 MESSAGES -------------
"""    

class ReadAndConvert():
    #Need to read the folder and open each flight run_00n to csv
        #source_dir = dataset/run_00n/ros_bag
        #dataset = destination 
    def __init__(self, source_dir: Path, csv_destination: Path, j_son: Path):
        self.source_dir = source_dir
        self.csv_destination = csv_destination
        self.run_folder = self.csv_destination / "run_csv"
        self.j_son = j_son

    """
------------- COLLECT PATHS FOR FLIGHT DATA AND CREATE FOLDER -------------
    """   

    #collects all the flight data from the folders
    def collect_flight_data(self):
        flight_data = list(item for item in self.source_dir.rglob("*.mcap"))

        #Safety check
        if not flight_data:
            raise ValueError(f"No file with .mcap in {self.source_dir}")
        
        return flight_data
    
    #Makes the run folders for the ones containing the run_00n.csv
    def create_folder(self):
        self.run_folder.mkdir(parents=True, exist_ok=True)

    """
------------- READ FROM .JSON FILE -------------
    """  

    def read_json(self, j_file):
        with open(j_file, "r") as file:
            flight_metadata = json.load(file)
        
        flatten_json = pd.json_normalize(flight_metadata)
        
        gps_condition = flatten_json.loc[0, "scenario.gps condition"]
        if gps_condition == "normal":
            flatten_json["label"] = 0
        elif gps_condition == "spoofed":
            flatten_json["label"] = 1
        
        return flatten_json

    """
------------- RED FROM THE ROS2 BAGS -------------
    """     

    #Read the messages from the bag
    def read_bag(self, current_mcap):
        bag_reader = SequentialReader()

        storage = StorageOptions(
            uri=str(current_mcap.parent),
            storage_id="mcap"
        )

        converter = ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        )

        bag_reader.open(storage, converter)

        #Goes through every messages recorded.
        #Extracts it for csv to read and save into there own row.
        while bag_reader.has_next():
            topic, serialized_msg, timestamp = bag_reader.read_next()
            if topic == "/fmu/out/vehicle_gps_position":  
                message = deserialize_message(serialized_msg, SensorGps)
                yield topic, message, timestamp
            elif topic == "/fmu/out/vehicle_global_position":  
                message = deserialize_message(serialized_msg, VehicleGlobalPosition)
                yield topic, message, timestamp                
            
            elif topic == "/fmu/out/vehicle_odometry":  
                message = deserialize_message(serialized_msg, VehicleOdometry)
                yield topic, message, timestamp

    """
------------- SET PATHS AND CREATE THE CSVS -------------
    """     

    #Get the name of the run parent file to use for the files
    def get_name(self, current_mcap):
        parent_flight_data = current_mcap.parent.parent.parent
        return parent_flight_data.name

    #Estabish the csv paths
    def get_csv_path(self, current_mcap):
        name = self.get_name(current_mcap)

        return {
            "gps": self.run_folder / f"{name}_gps.csv",
            "global": self.run_folder / f"{name}_global.csv",
            "odometry": self.run_folder / f"{name}_odometry.csv",
            "merged": self.run_folder / f"{name}_merged.csv",
        }
    
    #Create the csv and write each row from the bag
    def create_csv(self, current_mcap, ros_bag):
        #create the csvs with the headers
        paths = self.get_csv_path(current_mcap)

        gps_csv = paths["gps"]
        global_csv = paths["global"]
        odometry_csv = paths["odometry"]
        merged_csv = paths["merged"]

        #Open four seperate csv.
        #Each have their own set of data.
        #Will be used to merge later
        with open(gps_csv, "w", newline="") as gps_file, \
             open(global_csv, "w", newline="") as global_file, \
             open(odometry_csv, "w", newline="") as odometry_file:
            
            gps_writer = csv.writer(gps_file)
            gps_writer.writerow([
                "timestamp",
                "latitude",
                "longitude",
                "altitude",
                "vel_n",
                "vel_e",
                "vel_d",
                "fix_type",
                "satellites"
            ])

            global_writer = csv.writer(global_file)
            global_writer.writerow([
                "timestamp",
                "lat",
                "lon",
                "alt"
            ])
            
            odometry_writer = csv.writer(odometry_file)
            odometry_writer.writerow([
                "timestamp",
                "position_x", "position_y", "position_z",
                "velocity_x", "velocity_y", "velocity_z",
                "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
                "q0", "q1", "q2", "q3"
            ])

            #In the ros bag depending on the topic
            #Write into the given csv with the message from the ros bag
            for topic, msg, timestamp in ros_bag:
                if topic == "/fmu/out/vehicle_gps_position":  
                    gps_writer.writerow([
                            timestamp,
                            msg.latitude_deg,
                            msg.longitude_deg,
                            msg.altitude_msl_m,
                            msg.vel_n_m_s,
                            msg.vel_e_m_s,
                            msg.vel_d_m_s,
                            msg.fix_type,
                            msg.satellites_used
                        ])

                elif topic == "/fmu/out/vehicle_global_position":
                    global_writer.writerow([
                        timestamp,
                        msg.lat,
                        msg.lon,
                        msg.alt
                    ])
                
                elif topic == "/fmu/out/vehicle_odometry":
                    odometry_writer.writerow([
                        timestamp,
                        msg.position[0],
                        msg.position[1], 
                        msg.position[2],
                        msg.velocity[0], 
                        msg.velocity[1], 
                        msg.velocity[2],
                        msg.angular_velocity[0], 
                        msg.angular_velocity[1], 
                        msg.angular_velocity[2], 
                        msg.q[0],
                        msg.q[1],
                        msg.q[2],
                        msg.q[3],

                    ])
        print(f"\n-------- Created CSVs --------")
        return {
            "gps": gps_csv,
            "global": global_csv,
            "odometry": odometry_csv,
            "merged": merged_csv
        }

    """
------------- CSV MERGEING AND HELPER FUNCTIONS FOR MERGE -------------
    """      

    #Helper for the merge section
    #Finds the smallest time from gps and the row of the other
    #types from the ros bag.
    def find_difference(self, timestamp, rows):
        close = None
        small_difference = float("inf")

        for row in rows:
            difference = abs(float(timestamp) - float(row["timestamp"]))

            if difference < small_difference:
                small_difference = difference
                close = row

        return close
    
    #Reads the row indiviusdaly
    def read_rows(self, csv_path):
        with open(csv_path, "r", newline="") as file:
            reader = csv.DictReader(file)
            return list(reader)

    #Checks to make sure there is some value in the data
    def check_value(self, row, key):
        if row is None:
            return ""
        return row.get(key, "")
    
    #Merge the four csvs together per row.
    def merge_csv(self, gps_csv, global_csv, odometry_csv, j_file,
                  lat_meters, long_meters, drift, jump,
                  attack_start, attack_end):
        gps_rows = self.read_rows(gps_csv)
        global_rows = self.read_rows(global_csv)
        odometry_rows = self.read_rows(odometry_csv)

        #JSON metadata
        metadata = self.read_json(j_file)
        total_rows = len(gps_rows)

        merged = []
        print(f"\n-------- Starting Merge in Merge_CSV --------")
        print(f"\n-------- Looping on {total_rows} Rows --------")
        
        #Need timestamp to get the timestamps and calculate
        #the elapsed
        first_row = gps_rows[0]
        first_gps_timestamp = float(first_row["timestamp"])

        #Counts and keep track for user.
        for index, rows in enumerate(gps_rows, start=1):
            print(f"\n-------- Loop {index} out of {total_rows} --------")
            merge_global = self.find_difference(rows["timestamp"], global_rows)
            merge_odometry = self.find_difference(rows["timestamp"], odometry_rows)

            #Spoof math if metadata says spoof
            timestamp = float(rows["timestamp"])
            elapsed_time = (timestamp - first_gps_timestamp) / 1000000000
            latitude = float(rows["latitude"])
            longitude = float(rows["longitude"])
            spoof_type = metadata.loc[0, "scenario.spoof types"]
            spoof_cond = metadata.loc[0, "scenario.gps condition"]

            #If it's spoofed calcuate the new lat and long
            if spoof_cond == "spoofed":
                if attack_start <= elapsed_time and elapsed_time <= attack_end:
                    gps_conditions = RunGPSCondition(rows)
                    new_lat, new_long = gps_conditions.select_condition(
                        lat_meters=lat_meters, long_meters=long_meters,
                        drift_amount=drift, jump_amount=jump,
                        spoofed_type=spoof_type, time_elapsed=elapsed_time)
                    
                    #Change the lat and long to the spoof calculation 
                    latitude = new_lat
                    longitude = new_long

            print(f"\n-------- Starting Append --------")
            merged.append ({
                "gps_timestamp": rows["timestamp"],
                "gps_latitude_deg": latitude,
                "gps_longitude_deg": longitude,
                "gps_altitude_msl_m": rows["altitude"],
                "gps_vel_n_m_s": rows["vel_n"],
                "gps_vel_e_m_s": rows["vel_e"],
                "gps_vel_d_m_s": rows["vel_d"],
                "gps_fix_type": rows["fix_type"],
                "gps_satellites_used": rows["satellites"], 
                
                "global_lat": self.check_value(merge_global,"lat"),
                "global_lon": self.check_value(merge_global, "lon"),
                "global_alt": self.check_value(merge_global, "alt"),
                
                "odom_pos_x": self.check_value(merge_odometry, "position_x"),
                "odom_pos_y": self.check_value(merge_odometry, "position_y"),
                "odom_pos_z": self.check_value(merge_odometry, "position_z"),
                "odom_vel_x": self.check_value(merge_odometry, "velocity_x"),
                "odom_vel_y": self.check_value(merge_odometry, "velocity_y"),
                "odom_vel_z": self.check_value(merge_odometry, "velocity_z"),

                "run id": metadata.loc[0,"scenario.run id"],
                "location name": metadata.loc[0,"scenario.location name"],
                "lat": metadata.loc[0,"scenario.location.lat"],
                "long": metadata.loc[0,"scenario.location.long"],
                "wind condition": metadata.loc[0,"scenario.wind condition"],
                "label": metadata.loc[0,"label"],
                "gps condition": metadata.loc[0,"scenario.gps condition"],
                "mission type": metadata.loc[0,"scenario.mission type"],
                "flight duration":  metadata.loc[0,"scenario.flight duration"]
            })
            #End Append

        #Return the merged csv
        print(f"\n-------- Finished Merge --------")
        return merged

    """
------------- MERGE THE FINISH CSV -------------
    """    

    def save_merged_csv(self, merged_rows, output_path):
        if not merged_rows:
            return
        print(f"\n-------- Writing into Merged CSV --------")
        with open(output_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=merged_rows[0].keys())
            writer.writeheader()
            writer.writerows(merged_rows)

    """
------------- MAIN ROS2 TO CSV CONVERTER -------------
    """    
    #Run everything here.
    #Creates new folder for all merged data
    #runs the csv write and merges them
    def convert_ros2_csv(self, lat_meters, long_meters, drift, jump,
                         attack_start, attack_end):
        #Get director to us across different file directory formats

        #Collect data
        collected_data = self.collect_flight_data()
        print(f"\n-------- Collected Data --------")

        #Created folder
        self.create_folder()
        print(f"\n-------- Created Folder --------")

        #Read and start conversion
        print(f"\n-------- Starting Read on Ros Bag --------")
        for mcap in collected_data:
            ros_bag = self.read_bag(mcap)
            print(f"\n-------- Starting Create CSV --------")
            created_csv = self.create_csv(mcap, ros_bag)
            print(f"\n-------- Starting merge --------")
            merged = self.merge_csv(created_csv["gps"], created_csv["global"],
                                     created_csv["odometry"], self.j_son,
                                     lat_meters, long_meters, drift, jump,
                                     attack_start=attack_start, attack_end=attack_end)

            #Merged csvs 
            print(f"\n-------- Saved Merged Items --------")
            self.save_merged_csv(merged, created_csv["merged"])

        print(f"\n-------- Finished --------")
            
"""
------------- CLIENT HANDLER -------------
"""    

def client(source_dir, csv_destination, j_son,
           lat_meters, long_meters, 
           drift, jump,
           attack_start, attack_end):
    read_folders = ReadAndConvert(source_dir, csv_destination, j_son)
    read_folders.convert_ros2_csv(lat_meters=lat_meters, long_meters=long_meters,
                                  drift=drift, jump=jump,
                                  attack_start=attack_start, attack_end=attack_end)

"""
------------- RUN APPLICATION HERE -------------
"""    

if __name__ == "__main__":

    #Change with your file path
    source = Path("/home/mudsk/ros2_drone_ws/src/drone_basics/dataset")
    destination = Path("/home/mudsk/ros_csv")
    j_son = Path("/home/mudsk/ros2_drone_ws/src/drone_basics/dataset/run_001/metadata.json")
    client_test = client(source, destination, j_son,
                         lat_meters=10, long_meters=15,
                         drift=5, jump=10,
                         attack_start=120, attack_end=350)