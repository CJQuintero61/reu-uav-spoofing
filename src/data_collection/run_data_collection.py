import os
import json
import random
import subprocess
import signal
from pathlib import Path
from datetime import datetime
import shutil

#import setup variables
import config_vars as variables

"""
This file is to run an autonomous flight sim of n nuumber of flights.
The goal is to have it be autonomous while storing the ulogs and 
  getting the ros2 information seperately. Getting specificly the 
  specific gps information from flight.
"""

#Selects the scenarios from dictionary
class ScenarioSelection():
    def create(self, run_id):
        selected_location = random.choice(variables.locations)
        scenario = {
            "run id": run_id,
            "location": selected_location,
            "location name": selected_location["name"],
            "altitude": random.choice(variables.altitude),
            "gps condition": random.choice(variables.gps_conditions),
            "mission type": random.choice(variables.mission_type),
            "flight duration" : 600,
            "spoof types" : random.choice(variables.spoofing_profiles)
        }

        return scenario
    
#Runs the flight sim and data collection
class RunFlightHandler():
    def __init__(self, scenario):
        self.scenario = scenario
        self.run_path = variables.dataset_path / self.scenario["run id"]
        self.ros_bag = self.run_path / "ros_bag"
        self.px4_ulg = self.run_path / "px4_ulg" 
    
    #Create the folders for ros and px4.
    #Automatically makes the parent directories and bypasses the error for existing folder
    def create_folder(self):
        self.ros_bag.mkdir(parents=True, exist_ok=True)
        self.px4_ulg.mkdir(parents=True, exist_ok=True)

    def save_data(self):
        meta_data = {
            "created_at" : datetime.now().isoformat(),
            "scenario" : self.scenario
        }

        with open(self.run_path / "metadata.json", "w") as f:
            json.dump(meta_data, f, indent=4)

    #Starts up px4 and gazebo
    def start_px4_gazebo(self):
        location = self.scenario["location"]
        
        #Enviornment variables
        env = os.environ.copy()
        
        #Sets the px4 location for this specific env.
        env["SIM_GZ_HOME_LAT"] = str(location["lat"])
        env["SIM_GZ_HOME_LON"] = str(location["long"])
        env["SIM_GZ_HOME_ALT"] = str(location["alt"])  

        env["PX4_HOME_LAT"] = str(location["lat"])
        env["PX4_HOME_LON"] = str(location["long"])
        env["PX4_HOME_ALT"] = str(location["alt"])
        
        print("PX4 home set to:", location)
        
        command = ["make", "px4_sitl", "gz_x500"]

        #returns that command from the given directory and envronment variables
        return subprocess.Popen(
            command,
            cwd=variables.px4_dir,
            env=env,
            preexec_fn=os.setsid
        )
    
    #Starts ros bag to collect data.
    def start_ros_bag(self):
        ros_bag_output = self.ros_bag / "flight_data"

        #If the folder exists delete it
        if ros_bag_output.exists():
            shutil.rmtree(ros_bag_output)

        #To collect the bag, run this command
        command = [
            "ros2",
            "bag",
            "record",
            "-o",
            str(ros_bag_output),
        ] + variables.topics

        return subprocess.Popen(command, preexec_fn=os.setsid)
    
    #Copy Ulog
    def copy_ulog(self):
        #Set ulog director
        ulog_dir = list(variables.px4_logs.rglob("*.ulg"))

        #Safety check
        if not ulog_dir:
            print("No .ulg files")
            return
        
        #Get the latest ulog and "copy" and place in a different directory.
        latest_ulog = max(ulog_dir, key=os.path.getmtime)
        destination = self.px4_ulg / latest_ulog.name

        destination.write_bytes(latest_ulog.read_bytes())

        print("Ulog copied")
        

    #Terminate the ros bag process (might still be wiritng in the files)
    def stop_ros_process(self, process):
        process.terminate()

        try:
            process.wait(timeout=10)
        except:
            process.kill()
            process.wait()

    #Kill the px4 process
    def stop_px4_process(self, process):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
        except:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)