
#Autonomous drone control
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan
import asyncio

#Math for flight pathers
import math
import config_vars as variables

"""
Main flight control code to fly the drone autonomously.
Handles all mission generation and px4 flight controls.
"""

class DroneControl():
    #initialize the drone and know the location, alt, and mission type
    def __init__(self, location, altitude, mission):
        self.drone = System()
        self.location = location
        self.altitude = altitude
        self.mission = mission

    """
------------- PX4 HELPERS -------------
    """  

    #Connect
    async def connect_to_vehicle(self):
       await self.drone.connect(system_address="udpin://0.0.0.0:14540")

    #Wait for conenction
    async def wait_until_connected(self):
       async for state in self.drone.core.connection_state():
        if state.is_connected:
            print("Drone Connected")
            break

    #Wait for ready
    async def wait_until_ready(self):
       async for health in self.drone.telemetry.health():
           if health.is_global_position_ok and health.is_home_position_ok:
               print("Drone ready")
               break

    #Arms the drone ready to fly
    async def arm_vehicle(self):
        await self.drone.action.arm()
    
    #Takeoff
    async def takeoff(self):
        await self.drone.action.set_takeoff_altitude(self.altitude)
        await self.drone.action.takeoff()
        await asyncio.sleep(5)

    #Land
    async def land_vehicle(self):
        await self.drone.action.land()
    
    #Upload/Start mission
    async def upload_mission(self, mission_plan):
        await self.drone.mission.upload_mission(mission_plan)
        
    async def start_mission(self):
        await self.drone.mission.start_mission()

    #Waits for the mission to be completed
    async def wait_mission_complete(self):
        async for progress in self.drone.mission.mission_progress():
            if progress.total > 0 and progress.current == progress.total:
                break
    
    """
------------- GENERATE SQUARE MISSION -------------
    """  
        
    #Square mission
    async def generate_square_mission(self, speed_m_s):
        #Square mission needs four points
        #dictory for waypoints
        
        mission_map = {
            "tamu-cc": 
                    variables.waypoint_tamucc,
            "downtown corpus christi": 
                    variables.waypoint_downtown,
            "corpus christi international airport": 
                    variables.waypoint_airport
        }

        #Selects the location from the map.
        selected_location = mission_map[self.location]

        mission_item = []

        for _ in range(30):
            for coords in selected_location:
                waypoint = MissionItem(
                    latitude_deg=coords["lat"],
                    longitude_deg=coords["long"],
                    relative_altitude_m=self.altitude,
                    speed_m_s=speed_m_s,
                    is_fly_through=True,
                    gimbal_pitch_deg=float("nan"),
                    gimbal_yaw_deg=float("nan"),
                    camera_action=MissionItem.CameraAction.NONE,
                    loiter_time_s=0,
                    camera_photo_interval_s=0,
                    acceptance_radius_m=1,
                    yaw_deg=float("nan"),
                    camera_photo_distance_m=0,
                    vehicle_action=MissionItem.VehicleAction.NONE
                )

                #Combined the list of coords from config_vars.py
                mission_item.append(waypoint)

        #Returnt hat mission plan
        return MissionPlan(mission_item)

    """
------------- GENERATE HOVER MISSION -------------
    """  

    #Hover Mission
    async def fly_hover_mission(self, duration_sec):
        #Take off and hover for duration time
        await self.arm_vehicle()
        await self.takeoff()
        await asyncio.sleep(duration_sec)

        #lands drone after the duration time
        await self.land_vehicle()

    """
------------- GENERATE CIRCLE MISSION -------------
    """  
    #Circle Mission
    async def generate_circle_mission(self, center_lat,
                                center_long, radius_m,
                                num_points, speed_m_s):
        circle_points = []
        mission_waypoints = []

        for i in range(num_points):
            angle = 2 * math.pi * i / num_points

            north_offset = radius_m * math.cos(angle)
            east_offset = radius_m * math.sin(angle)

            new_long = center_long + (
                                      east_offset / (111111 * math.cos(math.radians(center_lat))))
            new_lat = center_lat + (north_offset/111111)

            #Sets the mission item in the loop for the circle
            #for each 
            waypoint = MissionItem(
                latitude_deg=new_lat,
                longitude_deg=new_long,
                relative_altitude_m=self.altitude,
                speed_m_s=speed_m_s,
                is_fly_through=True,
                gimbal_pitch_deg=float("nan"),
                gimbal_yaw_deg=float("nan"),
                camera_action=MissionItem.CameraAction.NONE,
                loiter_time_s=0,
                camera_photo_interval_s=0,
                acceptance_radius_m=1,
                yaw_deg=float("nan"),
                camera_photo_distance_m=0,
                vehicle_action=MissionItem.VehicleAction.NONE
            )
            
            circle_points.append(waypoint)

        #Loop to store 20 waypoints for missions.
        for _ in range(30):
            for waypoint in circle_points:
                mission_waypoints.append(waypoint)

        return MissionPlan(mission_waypoints)

    """
------------- RUN MISSION -------------
    """  

    #Selects the mission type
    async def run_mission(self, duration_sec):
        #Calls hover
        print("Selected mission:", self.mission)
        if self.mission == "hover":
            await self.fly_hover_mission(duration_sec)

        #Preforms the square mission.
        #Got to four points starting form location
        elif self.mission == "square":
            #Testing for time.
            await self.drone.mission.clear_mission()

            #Generate the mission and upload it
            mission_plan = await self.generate_square_mission(speed_m_s=15)    
            await self.upload_mission(mission_plan)

            #Arm the drone and set or takeoff
            await self.arm_vehicle()
            await self.takeoff()

            #Start the mission and land adter duration_sec
            await self.start_mission()
            await asyncio.sleep(duration_sec)               

            await self.land_vehicle()
            await asyncio.sleep(10)

        #Perfroms the circle mission
        #From startpoint go to starting circle point and fly in circle
        elif self.mission == "circle":
            #Gets the location and the lat and long for the center.
            center = None

            for location in variables.locations:
                if location["name"] == self.location:
                    center = location
                    break

            #Fail safe in case no location is found.
            if center is None:
                raise ValueError (f"Unknown location: {self.location}")

            #Generate Mission
            #Clear the first mission
            await self.drone.mission.clear_mission()

            mission_plan = await self.generate_circle_mission(
                center_lat=center["lat"],
                center_long=center["long"],
                radius_m=25,
                num_points=12,
                speed_m_s=15
            )
            await self.upload_mission(mission_plan)

            #Arm and ready for takeoff
            await self.arm_vehicle()
            await self.takeoff()

            #Start and land drone after duration_sec
            await self.start_mission()     
            await asyncio.sleep(duration_sec)
            await self.land_vehicle()
            await asyncio.sleep(10)
            