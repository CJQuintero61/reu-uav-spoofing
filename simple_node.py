#Need these to have a node for the simulation
import rclpy
from rclpy.node import Node

#Need these to talk to the PX4 for commands through
#PX4 messaging.
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand


#Creates your own node.
#Create a new type of Node called SImpleDroneNode
class SimpleDroneNode(Node):
    #Runs automatically when node is created.
    #What should it do at the start
    def __init__(self):
        #Creates the ROS node
        #The name becomes simple_drone_node
        super().__init__('simple_drone_node')
        
        #Inits the communication between ROS2 and PX4
        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            10
        )

        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            10
        )

        self.command_pub = self. create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            10
        )

        #IDK what the counter is for
        #Will look for later
        self.counter = 0

        #Tells ROS
        #Every 1 second run timer_callback
        self.timer = self.create_timer(
            1.0,
            self.timer_callback
        )

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.from_external = True
        self.command_pub.publish(msg)

    #timer first this code
    #Same as Print statement
    def timer_callback(self):
       #Need to tell PX4 that we want the position control
       #I want to control position set = True
       #I do not want to control velocity set = False
       offboard_msg = OffboardControlMode()
       offboard_msg.position = True
       offboard_msg.velocity = False
       
       self.offboard_pub.publish(offboard_msg)

       self.get_logger().info(
           f"Counter = {self.counter}")
       
       #Desired position
       #AKA GO HERE AT X = 0, Y = 0, Z = -5.0
       setpoint = TrajectorySetpoint()
       setpoint.position = [0.0, 0.0, -5.0]

       self.setpoint_pub.publish(setpoint)


       #After sending a few setpoints,
       #switch to offboard and arm
       if self.counter == 10:
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                1.0,
                6.0
            )

            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                1.0
            )
       
       self.counter += 1

#Where the main program starts
def main(args=None):
    rclpy.init(args=args)
    node = SimpleDroneNode() #Creates an instance of the node
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
