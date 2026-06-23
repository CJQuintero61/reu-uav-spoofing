from pathlib import Path

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
wind_condition = ["none", "low", "moderate", "strong", "very strong"]
mission_type = ["hover", "square", "circle"]
gps_conditions = ["normal", "spoofed"] #Can add more types

#Starting area TAMU-CC
waypoint_tamucc = [
    {"lat": 27.71351,"long":-97.32961},
    {"lat": 27.69005,"long":-97.33524},
    {"lat": 27.70592,"long":-97.36645},
    {"lat": 27.72808,"long":-97.35170}
]
#Starting area Corpus Christi downtown
waypoint_downtown = [
    {"lat": 27.80090,"long":-97.39637},
    {"lat": 27.80390,"long":-97.46478},
    {"lat": 27.76386,"long":-97.46770},
    {"lat": 27.76697,"long":-97.38487}
]

#Starting area Corpus Christi Airport
waypoint_airport = [
    {"lat": 27.77407,"long":-97.50225},
    {"lat": 27.67169,"long":-97.50023},
    {"lat": 27.67169,"long":-97.1922},
    {"lat": 27.77504,"long":-97.41955}
]

#Attack items
spoofing_profiles = [
    "constant_offset",
    "slow_drift",
    "sudden_jump"
]