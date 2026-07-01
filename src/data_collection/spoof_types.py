import math

class RunGPSCondition():
    def __init__(self, gps_row):
        self.gps_row = gps_row
    
    def select_condition(self, lat_meters, long_meters, 
                         drift_amount, jump_amount,
                         spoofed_type, time_elapsed):
        spoof_attacks = SpoofIdentifers(gps_row=self.gps_row)
        match spoofed_type:
            case "offset":
                lat_offset, long_offset = spoof_attacks.offset(lat_meters=lat_meters, long_meters=long_meters)
                return lat_offset, long_offset
            case "drift":
                lat_drift, long_drift = spoof_attacks.drift(drift_amount=drift_amount,time_elapsed=time_elapsed)
                return lat_drift, long_drift
            case "sudden jump":
                lat_jump, long_jump = spoof_attacks.sudden_jump(jump_amount=jump_amount)
                return lat_jump, long_jump
            
class SpoofIdentifers():
    #initilize variable for gps information
    def __init__(self, gps_row):
        self.location = gps_row
        self.spoofed = self.location.copy()
        self.lat = float(self.location["latitude"])
        self.long = float(self.location["longitude"])
        
    
    #Set lat and long
    def lat_meters(self, meters):
        new_lat = meters / 111111
        return new_lat
    
    def long_meters(self, lat, meters):
        new_long = meters / (111111*math.cos(math.radians(lat)))
        return new_long
    
    #Attack surfaves

    #Sets an offset of the lat and long
    #---------
    #         -------- <- new offset
    def offset(self, lat_meters, long_meters):
        #Set offset for both lat and long
        north_offset = lat_meters
        east_offset = long_meters

        self.spoofed["latitude"] = self.lat + self.lat_meters(north_offset)
        self.spoofed["longitude"] = self.long + self.long_meters(self.lat, east_offset)

        return self.spoofed["latitude"], self.spoofed["longitude"]
    
    #Drift drone an amount over time.
    """ 
            - <- new gps location after drift
           -
          -
         -
    -----
    """
    def drift(self, drift_amount, time_elapsed):
        drift_rate = drift_amount * time_elapsed

        self.spoofed["latitude"] = self.lat + self.lat_meters(drift_rate)
        self.spoofed["longitude"] = self.long + self.long_meters(self.lat, drift_rate)

        return self.spoofed["latitude"], self.spoofed["longitude"]
    

    # Jumps to that amount
    #      -
    #    -
    # ---
    def sudden_jump(self, jump_amount):
        jump_distance = jump_amount

        self.spoofed["latitude"] = self.lat + self.lat_meters(jump_distance)
        self.spoofed["longitude"] = self.long + self.long_meters(self.lat, jump_distance)

        return self.spoofed["latitude"], self.spoofed["longitude"]