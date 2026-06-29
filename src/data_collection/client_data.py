#import files
from data_collection_module.run_data_collection import ScenarioSelection, RunFlightHandler
from data_collection_module.autonomous_flight import DroneControl
from data_collection_module.config_vars import spoofing_profiles

import asyncio
import time

"""
This file runs an autonomous flight for n number of times to collect normal
data for Machine Learning models.

During flights it will randomly selecte conditions for each scenario
  -> Mission_types: circle, square, hover
  -> GPS conditions: spoofed, normal
  -> spoof types: offset, drift, jump
  -> etc. See run_data_collection.py for specifics

During each flight it will record and store ros2 messages and ulg data
that can be converted into a csv file which is done thorugh convert_ros2_csv.py.

Flights are timed at 600 seconds (10 mins) each and will automaticlly kill the process
of px4 and gazebo. You do not need to restart QGroundControl or the MicroXRCE Agent.

How to use:
In one terminal:
  1) cd ~/ros2_drone_ws/src/drone_basics
  2) source .venv/bin/activate
  3) Check that it's in the venv
     1) which python3
     2) should say something like 
        /scr/drone_basics/.venv/bin/python3
  4) source /opt/ros/jazzy/setup.bash
  5) source ~/ros2_drone_ws/install/setup.bash
Our to do this with one file
  1) run
     . ./source_and_start_ros2.sh
  2) python3 <file_name.py>

In another terminal:
  Run: MicroXRCEAgent udp4 -p 8888

In another terminal:
  Run your QGroundControl
  Example: ./QgroundControl-x86_64.AppImage

In the ros2 terminal from the top run the script
python3 client_data.py

To stop everything mid flight
    OR
To ensure everthing is terminated

Run these items first
1) ctrl + c
2)
    pkill -f px4
    pkill -f gz
    pkill -f gazebo
    pkill -f ros2
    pkill -f mavsdk_server
    pkill -f "ros2 bag record"

To check if everything was terminated
ps aux | grep -E "px4|gz|gazebo|ros2"

If any of them refuse to terminate
--> kill -9 <PID>
    - (PID equals the number in the second column 
    - when using ps aux | grep -E "px4|gz|gazebo|ros2"

If everything is not terminated it will lead to issues when trying to
run the script again.
"""

#Handlers and sets up all needed functions from classes
class RunHandlers():
    def __init__(self):
        #initlizae the classes
        self.scenarios = ScenarioSelection()

    async def run(self, total_flights):
        #Per flight run this

        print(f"\n-------- Running {total_flights} flight(s) --------")
        for i in range(1, total_flights + 1):
            run_id = f"run_{i:03d}"
            print(f"\n-------- Flight {i} out of {total_flights}")
            
            drone_scenario = self.scenarios.create(run_id)
            print(f"\n-------- Mission Type: {drone_scenario["mission type"]} --------")    
            
            handler = RunFlightHandler(drone_scenario)
            spoof = spoofing_profiles

            #Create and save data and folders
            handler.create_folder()
            handler.save_data()
            print("\n-------- Successfully Created Dataset Folders --------")

            print("\n-------- Started PX4 and Gazebo --------")
            px4_handler = handler.start_px4_gazebo()

            #Waits for 10 seconds and checks to see if PX4 fails.
            time.sleep(10)
            if px4_handler.poll() is not None:
                print("\n-------- PX4 failed to start --------")
                return
            
            #Setup drone control and automous.
            #Sets up missions
            print("Scenario mission:", drone_scenario["mission type"])
            drone = DroneControl(
            drone_scenario["location name"], 
            drone_scenario["altitude"],
            drone_scenario["mission type"]
            )

            print("\n-------- Drone Safety Started Procedures --------")

            print("\n-------- Connecting to Drone--------")
            await drone.connect_to_vehicle()
            print("\n-------- Waiting for Drone to Connect --------")
            await drone.wait_until_connected()

            print("\n-------- Wait for Ready --------")
            await drone.wait_until_ready()

            #Starts ros2 if PX4 and drone connection = work
            print("\n-------- Started Ros2 Process Bag --------")
            ros2_bag_handler = handler.start_ros_bag()

            print("\n-------- Started Running Mission --------")
            await drone.run_mission(drone_scenario["flight duration"])

            #Stop all processes and copy the ulog

            print("\n-------- Stopping Process --------")
            handler.stop_ros_process(ros2_bag_handler)
            handler.copy_ulog()
            handler.stop_px4_process(px4_handler)
            
            await asyncio.sleep(20)

            print("\n-------- All Processes stopped please check data --------")

#Runs n number amounts of flights using runner.run(n)
async def main_client():
    runner = RunHandlers()
    await runner.run(2)

#In this specfic file runs the main client function
if __name__ == "__main__":
    print("\n-------- Starting Data Collection System --------")
    asyncio.run(main_client())
    print("\n-------- Finish Sim Collection data --------")    