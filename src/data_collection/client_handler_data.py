#import files

from run_data_collection import ScenarioSelection, RunFlightHandler
from autonomous_flight import DroneControl
from spoof_attack import GPSConditionHandler
from config_vars import spoofing_profiles

"""
    Pipeline
    -> loop n number of times
    -> create scenario
    -> create runflighthandler
    -> create folders
    -> save metadata
    -> start px4/gazebo
    -> wait for sim startup/start ros
    -> create dronecontrol
    -> run autonomous mission
    -> stop rosbag
    -> copy ulog
    -> stop px4/gazebo

"""

class RunHandlers():
    def __init__(self):
        #initlizae the classes
        self.scenarios = ScenarioSelection()

    async def run(self, total_flights):
        for i in total_flights:
            #Setup
            run_id = f"run_{i:03d}"
            drone_scenario = self.scenarios.create(run_id)
            handler = self.run_flight(drone_scenario)
            spoof = spoofing_profiles

            self.run_flight = RunFlightHandler(drone_scenario)
            self.spoof_hander = GPSConditionHandler()

            #Create and save data and folders
            handler.create_folder()
            handler.save_data()

            px4_handler = handler.start_px4_gazebo()
            ros2_bag_handler = handler.start_ros_bag()
            
            #Setup drone control and automous.
            #Sets up missions
            drone = DroneControl(
            drone_scenario["location"], 
            drone_scenario["altitude"],
            drone_scenario["mission type"]
            )

            await drone.connect_to_vehicle()
            await drone.wait_until_connected()
            await drone.wait_until_ready()
            await drone.run_mission(drone_scenario["location name"],
                                    drone_scenario["flight_duration"]            
                                    )

            handler.stop_ros_process(ros2_bag_handler)
            handler.copt_ulog()
            handler.stop_px4_process(px4_handler)

#main client caller
def main_client():

    RunHandlers()

if __name__ == "__main__":
    main_client()