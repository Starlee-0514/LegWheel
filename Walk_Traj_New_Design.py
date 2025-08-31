import Trajectory_Planning as TJ
import LegModel
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\starl\\NTU\\Birola_LAB\\Code\\LegWheel')
import LegModel
import PlotLeg
from matplotlib.animation import FuncAnimation
import pandas as pd

# initialize
Traj = TJ.TrajectoryPlanner(stand_height=0.25,
                            step_length=0.4,
                            leg=LegModel.LegModel(),
                            step_height=0.08,
                            period=10,
                            dt=0.001,
                            duty=1 / 8,
                            overlap=0.2)

Traj.set_speed(0.3) # set speed to 0.1 m/s or 10cm /s

Traj.move()
Traj_Curve = Traj["cmd"]
Target_dist = 3     # target distance to walk, meter
total_time = Target_dist / Traj.V
data_len = int(total_time / Traj.dt)    #target cmd data length must output

BL = 0.444  # body length, 44.4 cm
BH = 0.2     # body height, 20 cm
BW = 0.33    # body width, 33 cm
CoM_bias = 0.0    # x bias of center of mass
# gait_select = "walk"
gait_select = "trot"

legs = [PlotLeg.PlotLeg() for _ in range(4)]
# ==<leg Number>
#       1 | 2 
#       ----- 
#       4 | 3
# ==============
# larger number means lift ooff earlier
if gait_select == "walk":
    # walking gait
    gait = [4,2,3,1]
elif gait_select == "trot":
    # trotting gait
    gait = [1,3,1,3]
else:
    gait = [4,2,3,1]

offset_x = [BL/2, BL/2, -BL/2, -BL/2]
offset_z = [-BW/2, BW/2, BW/2, -BW/2]

t = 0                                           # Initialize time
init_index = [0]*4                              # Initialize the index for each leg
traj_quad_point = [len(Traj_Curve)//4 * i for i in range(4)]                         # Initialize trajectory points for each leg
cmd_record = [[] for _ in range(8)]             # Initialize command record for each leg

# animation settings
fig,ax = plt.subplots(figsize=(14, 4))  
ani_interval = 100                              # Animation interval in milliseconds
speed = ani_interval /( Traj.dt *1000)          # Frame rate in milliseconds
frame_len = int(data_len/speed) +1               # Total frame length

# calculate initial point:
for i, leg in enumerate(legs):    
    init_index[i] = traj_quad_point[gait[i]-1]

# animation function
def update(frame):
    global t, ax, init_index, color, offset_x, offset_z, gait, cmd_record, speed, traj_quad_point
    # fig setting
    ax.clear()
    ax.set_xlim(-BL/2, Target_dist+BL)
    ax.set_ylim(0, BH*2)
    ax.set_aspect('equal')
    ax.set_title('Walking Animation'+f'T = {t:.2f}s'+
                 '\n' + f'Stand Height: {Traj.stand_height}m, Step Length: {Traj.step_length}m, Step Height: {Traj.step_height}m\n'+
                  f'velocity: {Traj.V:.2f}m/s')
    ax.grid()
    
    for i, leg in enumerate(legs):
        index = (init_index[i]+int(frame*speed))%len(Traj_Curve)
            
        if index >= traj_quad_point[3]:
            leg.setting(line_width=2.5, color='red')
        else:
            leg.setting(line_width=2, color='black')

        leg.plot_by_angle(Traj_Curve[index][0],
                        Traj_Curve[index][1],
                        O=np.array([offset_x[i] + Traj.V*t+BL/2, Traj.stand_height]),
                        ax=ax)
        
        cmd_record[i*2].append(Traj_Curve[index][0])
        cmd_record[i*2+1].append(Traj_Curve[index][1])
    t += Traj.dt*speed


# func_animation = FuncAnimation(fig, update, frames=len(Traj_Curve)*5, interval=Traj["dt"]*1000, repeat=False)
func_animation = FuncAnimation(fig, update, frames=frame_len, interval=ani_interval, repeat=False)

# transform function
def transform(joint):
    # build cmd to transfer from [17 deg, 0] to [joint] in 5000 steps
    theta, beta = joint[0], joint[1]
    T = np.linspace(np.deg2rad(17), theta, 5000).tolist()
    B = np.linspace(0, beta, 5000).tolist()
    # stack into two columns: [[theta0, beta0], ...]
    cmd = [[T[i], B[i]] for i in range(5000)]
    # cmd = [[T], [B]]
    return cmd

# save_data = True
save_data = False

# create output file name
path = "Output_datas/"
output_file_name = "".join([f"{gait_select}_Traj_",
                            "_Hip_height"+str(Traj.stand_height),
                            "_Step_Length"+str(Traj.step_length),
                            "_Step_Height"+str(Traj.step_height),
                            # "_Period"+str(Traj.T ),
                            "_Velocity"+str(round(Traj.V, 2)),
                            "_Distance"+str(Target_dist)])
if save_data:    
    # # saving animation
    print("Saving animation to", path+'Videos/'+output_file_name+".mp4")
    func_animation.save(path+'Videos/'+output_file_name + ".mp4", fps=10, writer='ffmpeg')
    
    # save command into csv
    cmd_temp = [[] for i in range(4)] #traj cmd for each legs
    cmd_out = [[] for _ in range(8)]
    for i in range(len(cmd_temp)):
        cmd_temp[i] = Traj_Curve[init_index[i]::]+Traj_Curve[:init_index[i]]
        print("data length after clip:\t", len(cmd_temp[i]))
        cmd_temp[i] = cmd_temp[i] * (data_len//len(cmd_temp[i])) + cmd_temp[i][:data_len%len(cmd_temp[i])]
        print("data length after extension:\t", len(cmd_temp[i]))
        cmd_temp[i] = transform(cmd_temp[i][0])+cmd_temp[i]
        print("data length after adding transform:\t", len(cmd_temp[i]))
        
        cmd_out[i*2] = list(map(lambda x: x[0] ,cmd_temp[i] ))
        cmd_out[i*2+1] = list(map(lambda x: x[1] *(-1 if i in [0,3] else 1) ,cmd_temp[i] ))
    
    print("Saving to", path+"CSV/"+output_file_name+".csv")
    # df = pd.DataFrame(np.array(cmd_out).T, columns=['theta1', 'beta1', 'theta2', 'beta2', 'theta3', 'beta3', 'theta4', 'beta4'])
    df = pd.DataFrame(np.array(cmd_out).T)
    df.to_csv(path+"CSV/"+output_file_name+".csv", index=False)
else:
    plt.show()
    # save command into csv
    cmd_temp = [[] for i in range(4)] #traj cmd for each legs
    cmd_out = [[] for _ in range(8)]
    for i in range(len(cmd_temp)):
        cmd_temp[i] = Traj_Curve[init_index[i]::]+Traj_Curve[:init_index[i]]
        print("data length after clip:\t", len(cmd_temp[i]))
        cmd_temp[i] = cmd_temp[i] * (data_len//len(cmd_temp[i])) + cmd_temp[i][:data_len%len(cmd_temp[i])]
        print("data length after extension:\t", len(cmd_temp[i]))
        cmd_temp[i] = transform(cmd_temp[i][0])+cmd_temp[i]
        print("data length after adding transform:\t", len(cmd_temp[i]))
        
        cmd_out[i*2] = list(map(lambda x: x[0] ,cmd_temp[i] ))
        cmd_out[i*2+1] = list(map(lambda x: x[1] *(-1 if i in [0,3] else 1) ,cmd_temp[i] ))
    
    print("Saving to", path+"CSV/"+output_file_name+".csv")
    # df = pd.DataFrame(np.array(cmd_out).T, columns=['theta1', 'beta1', 'theta2', 'beta2', 'theta3', 'beta3', 'theta4', 'beta4'])
    df = pd.DataFrame(np.array(cmd_out).T)
    df.to_csv(path+"CSV/"+output_file_name+".csv", index=False)
    