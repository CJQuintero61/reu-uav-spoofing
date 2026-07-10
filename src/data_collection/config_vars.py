from pathlib import Path

"""
Storage of dictories and lists and variables for easy access.
"""
#Set paths for data set and the px4
dataset_path = Path("dataset")
px4_dir = Path.home() / "PX4-autopilot"

px4_logs = (
    px4_dir
    / "build"
    / "px4_sitl_default"
    / "rootfs"
    / "log"
)

#Ros2 topics to record data from px4 and ros2
topics = [
    "/fmu/out/vehicle_gps_position",
    "/fmu/out/vehicle_global_position",
    "/fmu/out/vehicle_local_position",
    "/fmu/out/vehicle_odometry",
]

#Locations for code selection
#Can add more with the same format/
locations = [
    {
        "name":"tamu-cc",
        "lat": 27.7098,
        "long": -97.3148,
        "alt": 5
    },

    {
        "name":"downtown corpus christi",
        "lat": 27.800583,
        "long": -97.396378,
        "alt": 7
    },
    
    {
        "name":"corpus christi international airport",
        "lat": 27.7722,
        "long": -97.5024,
        "alt": 14
    }
]

#Add more if adding more altitude and locations
altitude = [10, 30, 40, 50]
mission_type = ["hover", "square", "circle"]
gps_conditions = ["normal", "spoofed"] #Can add more types

#Starting area TAMU-CC
waypoint_tamucc = [
    {"lat": 27.71030, "long": -97.31530},
    {"lat": 27.71030, "long": -97.31430},
    {"lat": 27.70930, "long": -97.31430},
    {"lat": 27.70930, "long": -97.31530}
]
#Starting area Corpus Christi downtown
waypoint_downtown = [
    {"lat": 27.80108, "long": -97.39688},
    {"lat": 27.80108, "long": -97.39588},
    {"lat": 27.80008, "long": -97.39588},
    {"lat": 27.80008, "long": -97.39688}
]

#Starting area Corpus Christi Airport
waypoint_airport = [
    {"lat": 27.77270, "long": -97.50290},
    {"lat": 27.77270, "long": -97.50190},
    {"lat": 27.77170, "long": -97.50190},
    {"lat": 27.77170, "long": -97.50290}
]

#Attack items
spoofing_profiles = [
    "offset",
    "drift",
    "sudden jump"
]