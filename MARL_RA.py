import numpy as np
import matplotlib.pyplot as plt
import time


class AOIPacket:
    def __init__(self, start_step):
        self.start_step = start_step
        self.service_time = np.random.exponential(1 / service_rate)

    def get_age(self, current_step):
        return (current_step - self.start_step) + 1

def get_server_state(queue, current_step):
    '''
    instead of get AOI of only one packet in queue, get AOI of ALL the packets in the queue 
    return the aoi in the queue and the queue legnth

    later use this to calculate the avergae AOI of system (use summation of queu_len , all over summation of queruelen)
    '''
    queue_aoi = 0
    for i in range(len(queue)):
        try:
            i_aoi= queue[i].get_age(current_step)
        except:
            i_aoi =0
        queue_aoi+=i_aoi
      # Queue length bins
    
    queue_len = len(queue)

    return queue_aoi, queue_len

# def bin_aaoi(total_aoi, total_queue_lengths=None):
#     if total_queue_lengths is not None:
  
#         try:
#             aaoi = total_aoi/total_queue_lengths
#         except:
#             aaoi= 1
#     else:
#         aaoi = total_aoi

#     # print("AAOI", aaoi)
    
#     if aaoi <= 2.0:
#         aaoi_bin = 0
#     elif aaoi <= 5.0:
#         aaoi_bin = 1
#     elif aaoi <= 8.0:
#         aaoi_bin = 2
#     else:
#         aaoi_bin = 3

#     return aaoi_bin # since getting AAOI, dont need to include queue length as a state
#     # so there are 4 bins for AAOI

# def system_aaoi(queues, current_step):
#     total_aoi, total_len = 0, 0
#     for q in queues:
#         q_aoi, q_len = get_server_state(q, current_step)
#         total_aoi += q_aoi
#         total_len += q_len
#     return total_aoi / total_len if total_len > 0 else 0.0

    
def get_state(queue, current_step):
    if len(queue) == 0:
        aoi = 0
    else:
        aoi = queue[0].get_age(current_step)


    # AoI bins
    if aoi <= 1:
        aoi_bin = 0
    elif aoi <= 3:
        aoi_bin = 1
    elif aoi <= 7:
        aoi_bin = 2
    else:
        aoi_bin = 3


    return aoi_bin


def get_power_bin(total_power_sat, power_limit):
    remaining_ratio = (power_limit - total_power_sat) / power_limit

    if remaining_ratio <= 0.20:
        return 0
    elif remaining_ratio <= 0.40:
        return 1
    elif remaining_ratio <= 0.60:
        return 2
    elif remaining_ratio <= 0.80:
        return 3
    else:
        return 4

# the below is new code to add server aware AAOI
num_gt_aoi_bins = 4
# num_aaoi_bins = 4
num_power_bins = 5
num_states = num_gt_aoi_bins  * num_power_bins
start_time = time.time()
print_interval = 1000

def state_to_index(gt_aoi_bin, power_bin):
    return gt_aoi_bin * num_power_bins + power_bin

#################################### below is older code 
# num_aoi_bins = 4
# num_queue_bins = 3
# num_power_bins = 5
# num_states = num_aoi_bins * num_queue_bins * num_power_bins
# there are 4 x 3 x5 states, but this state only look at the age of the packet at head of current queue

# def state_to_index(aoi_bin, queue_bin,power_bin):
    # return (aoi_bin * num_queue_bins*num_power_bins) + (queue_bin *num_power_bins) + power_bin

# params
num_satellites = 60
num_gts = 20
channels_per_satellite = 10
power_per_mbps = 5
base_power_budget = 500
num_iterations = 20_000

max_queue_length = 10
arrival_rate = 1.5 # lambda
service_rate = 0.5


# variables init
aoi_marl = np.zeros(num_iterations)
aoi_greedy = np.zeros(num_iterations)

aoi_ind_marl = []
aoi_ind_greedy = []

throughput_marl = np.zeros(num_iterations)
throughput_greedy = np.zeros(num_iterations)

power_marl = np.zeros(num_iterations)
power_greedy = np.zeros(num_iterations)

latency_marl = np.zeros(num_iterations)
latency_greedy = np.zeros(num_iterations)


# q_tables[sat][gt, ch]
q_tables = []

for sat in range(num_satellites):
    q_tables.append(np.zeros((num_states,num_gts,channels_per_satellite)))

# Q-learning / bandit parameters
alpha = 0.01
epsilon_start = 1.0
epsilon_end = 0.001
epsilon_decay = (epsilon_end / epsilon_start) ** (1 / num_iterations)

discount_factor = 0.99

# Generate consistent GT demands
half = num_gts // 2

high_demand_gts = np.arange(0, half)
low_demand_gts = np.arange(half, num_gts)

mean_high = 15
mean_low = 7

gt_demand_all = np.zeros((num_iterations, num_gts), dtype=int)

for i in range(num_iterations):
    gt_demand_all[i, high_demand_gts] = np.random.randint(
        mean_high - 2, mean_high + 3, size=len(high_demand_gts)
    )

    gt_demand_all[i, low_demand_gts] = np.random.randint(
        mean_low - 2, mean_low + 3, size=len(low_demand_gts)
    )


# Packet queues
packet_queues_marl = [[[] for _ in range(num_gts)] for _ in range(num_satellites)]
packet_queues_greedy = [[[] for _ in range(num_gts)] for _ in range(num_satellites)]  # capacity is 1

#Removed this so that the initial queues are empty. 
# Adds initial packets with arrival times -5 to -1
# for sat in range(num_satellites):
#     for gt in range(num_gts):
#         for packet_time in range(-5, 0):
#             packet_queues_marl[sat][gt].append(AOIPacket(packet_time))
#             packet_queues_greedy[sat][gt].append(AOIPacket(packet_time))


# Main sim loop
epsilon = epsilon_start
rewards=[]

total_packets_served_marl = 0
total_packets_served_greedy = 0
total_packets = 0

busy_until_marl = np.zeros((num_satellites, channels_per_satellite))
busy_until_greedy = np.zeros((num_satellites, channels_per_satellite))


for iteration in range(num_iterations):
    iter_step = iteration + 1

    # Add new packets
    for sat in range(num_satellites):
        for gt in range(num_gts):
            num_arrivals = np.random.poisson(arrival_rate)
            if num_arrivals > 0:
                packet_queues_marl[sat][gt] = [AOIPacket(iter_step)]
                packet_queues_greedy[sat][gt] = [AOIPacket(iter_step)]
                total_packets +=num_arrivals


    # dynamically change power budgets
    power_budget = base_power_budget + np.random.randint(50, 101, size=num_satellites)

    # Greedy Algorithm

    total_throughput_greedy = 0
    total_power_greedy = 0
    total_aoi_greedy = 0
    packets_served_greedy = 0

    gt_demand_iter = gt_demand_all[iteration].copy()

    for sat in range(num_satellites):
        gt_demand = gt_demand_iter.copy()

        total_power_sat = 0
        total_throughput_sat = 0
        total_aoi_sat = 0

        power_limit = power_budget[sat]

        # find aoi of oldest packet for each GT
        gt_aois = np.zeros(num_gts)

        for gt in range(num_gts):
            if len(packet_queues_greedy[sat][gt]) > 0:
                packet = packet_queues_greedy[sat][gt][0]
                gt_aois[gt] = packet.get_age(iter_step)
            else:
                gt_aois[gt] = -np.inf

        # sort GTs based on AoI, highest first
        gt_indices = np.argsort(gt_aois)[::-1]

        ch = 0
        index = 0

        while ch < channels_per_satellite and index < num_gts:
            gt = gt_indices[index]

            if len(packet_queues_greedy[sat][gt]) > 0:
                demand = gt_demand[gt]
                power_consumed = demand * power_per_mbps

                if total_power_sat + power_consumed <= power_limit:
                    packet = packet_queues_greedy[sat][gt][0]
                    packet_aoi = packet.get_age(iter_step)

                    packets_served_greedy += 1

                    total_power_sat += power_consumed
                    total_throughput_sat += demand
                    total_aoi_sat += packet_aoi

                    # Remove served packet
                    packet = packet_queues_greedy[sat][gt][0]
                    busy_until_greedy[sat][ch] = iter_step + packet.service_time
                    packet_queues_greedy[sat][gt] = []


                    aoi_ind_greedy.append(packet_aoi)

            ch += 1
            index += 1

        total_throughput_greedy += total_throughput_sat
        total_power_greedy += total_power_sat
        total_aoi_greedy += total_aoi_sat
    total_packets_served_greedy +=packets_served_greedy
    throughput_greedy[iteration] = total_throughput_greedy
    power_greedy[iteration] = total_power_greedy
    latency_greedy[iteration] = num_gts / (total_throughput_greedy + 1e-5)


    # MARL Simulation - Bandit Approach ( E greedy)

    total_throughput_marl = 0
    total_power_marl = 0
    total_aoi_marl = 0
    packets_served_marl = 0

    gt_demand_iter_marl = gt_demand_all[iteration].copy()

    for sat in range(num_satellites):

        gt_demand = gt_demand_iter_marl.copy()

        total_power_sat = 0
        total_throughput_sat = 0
        total_aoi_sat = 0
        total_queue_lengths = 0 

        power_limit = power_budget[sat]

        q_table = q_tables[sat]

        served_gts = np.zeros(num_gts, dtype=bool)

        for ch in range(channels_per_satellite):
            if iter_step < busy_until_marl[sat][ch]:
                continue

            # only consider unserved GTs that actually have packets
            available_gts = np.array([gt for gt in range(num_gts)  if not served_gts[gt] and len(packet_queues_marl[sat][gt]) > 0])

            if len(available_gts) == 0:
                break

            queues = packet_queues_marl[sat]
            power_bin = get_power_bin(total_power_sat,power_limit)
            # aaoi_bin = bin_aaoi(total_aoi_sat, total_queue_lengths)
            # aaoi_before = system_aaoi(queues, iter_step)
            # aaoi_bin = bin_aaoi(aaoi_before)
            # exploration
            if np.random.rand() < epsilon:
                action = np.random.choice(available_gts)

            # exploitation
            else:
                q_values = np.full(num_gts, -np.inf)

                for gt in available_gts:
                    aoi_bin = get_state(packet_queues_marl[sat][gt],iter_step)
                    state_index = state_to_index(aoi_bin, power_bin=power_bin)
                    q_values[gt] = q_table[state_index,gt,ch]

                action = np.argmax(q_values)
            # State of selected GT BEFORE action
            aoi_bin = get_state(packet_queues_marl[sat][action],iter_step)

            state_index = state_to_index(aoi_bin,power_bin)

            demand = gt_demand[action]
            power_consumed = demand * power_per_mbps

            served_gts[action] = True

            q_len = len(packet_queues_marl[sat][action])
            if (q_len > 0 and total_power_sat + power_consumed <= power_limit):

                packet = packet_queues_marl[sat][action][0]
                packet_aoi = packet.get_age(iter_step)
                packets_served_marl += 1

                total_power_sat += power_consumed
                total_throughput_sat += demand
                total_aoi_sat += packet_aoi
                total_queue_lengths += q_len

                packet = packet_queues_marl[sat][action][0]
                busy_until_marl[sat][ch] = iter_step + packet.service_time
                packet_queues_marl[sat][action] = []
                

                # aaoi_after = system_aaoi(queues, iter_step)
                
                reward = -(packet_aoi + power_consumed*0.1)    # if we helped the system then before will be greaater than after and get pos reward
                # print(power_consumed , packet_aoi)
                aoi_ind_marl.append(packet_aoi)


            else:
                # reward = 0
                reward= -power_consumed #penalize
                # print("REWARD penalize:", reward)

            rewards.append(reward) # just out of curiousity track the reward, see if trending up?


            #update
            current_q = q_table[state_index,action,ch]

            next_channel = ch+1

            next_available_gts = np.array([gt for gt in range(num_gts)  if not served_gts[gt] and len(packet_queues_marl[sat][gt]) > 0])

            if next_channel >= channels_per_satellite or len(next_available_gts) == 0:
                max_next_q = 0
            else:
                next_power_bin = get_power_bin(total_power_sat, power_limit)
                # next_aaoi = system_aaoi(packet_queues_marl[sat], iter_step)
                # next_aaoi_bin = bin_aaoi(next_aaoi)

                next_q_values = []

                for next_gt in next_available_gts:
                    next_aoi_bin = get_state(packet_queues_marl[sat][next_gt], iter_step)
                    next_state_index = state_to_index(next_aoi_bin,next_power_bin)
                    next_q_values.append(q_table[next_state_index, next_gt, next_channel])

                max_next_q = np.max(next_q_values)

        target = reward + discount_factor * max_next_q

        q_table[state_index, action, ch] = (current_q + alpha * (target - current_q))

        q_tables[sat] = q_table

        total_throughput_marl += total_throughput_sat
        total_power_marl += total_power_sat
        total_aoi_marl += total_aoi_sat

    # store the metrics for plots
    throughput_marl[iteration] = total_throughput_marl
    power_marl[iteration] = total_power_marl
    latency_marl[iteration] = num_gts / (total_throughput_marl + 1e-5)
    total_packets_served_marl +=packets_served_marl
    


    # Calculate average AoI
    if packets_served_marl > 0:
        aoi_marl[iteration] = total_aoi_marl / packets_served_marl
    else:
        aoi_marl[iteration] = 0

    if packets_served_greedy > 0:
        aoi_greedy[iteration] = total_aoi_greedy / packets_served_greedy
    else:
        aoi_greedy[iteration] = 0


    # epsilon decay
    epsilon *= epsilon_decay
    if iteration % print_interval == 0:
        print(f"Iteration {iteration + 1}/{num_iterations} completed. Epsilon: {epsilon:.6f} Average AoI MARL: {np.mean(aoi_marl[iteration-print_interval:iteration]):.2f}, Greedy: {np.mean(aoi_greedy[iteration-print_interval:iteration]):.2f}")



# window = 10000

# smoothed_rewards = np.convolve(rewards,np.ones(window) / window,mode="valid")

# plt.figure()
# plt.plot(smoothed_rewards)
# plt.xlabel("Decision")
# plt.ylabel("Average Reward")
# plt.title("MARL Reward Moving Average")
# plt.grid(True)
# plt.show()

# Plots

iterations = np.arange(1, num_iterations + 1)

# AAoI
plt.figure()

plt.plot(iterations, aoi_marl, linewidth=2, label="MARL AAoI")
plt.plot(iterations, aoi_greedy, linewidth=2, label="Greedy AAoI")

plt.xlabel("Iterations")
plt.ylabel("Average AOI")
plt.title("AoI Comparison: MARL vs Greedy")

plt.legend()
plt.grid(True)
plt.savefig("plots/" + str(int(time.time())) +  "-AAOI.png", dpi=300, bbox_inches="tight")
# plt.show()

# AoI
# plt.figure()

# plt.plot(aoi_ind_marl, label="MARL AoI")
# plt.plot(aoi_ind_greedy, label="Greedy AoI")

# plt.xlabel("Iterations")
# plt.ylabel("AOI")
# plt.title("AoI Comparison: MARL vs Greedy")

# plt.legend()
# plt.grid(True)
# plt.savefig("plots/" + str(int(time.time())) +  "-AOI.png", dpi=300, bbox_inches="tight")
# plt.show()



# Power
plt.figure()

plt.plot(iterations, power_marl, linewidth=2, label="MARL Power Usage")
plt.plot(iterations, power_greedy, linewidth=2, label="Greedy Power Usage")

plt.xlabel("Iterations")
plt.ylabel("Total Power Usage (Watts)")
plt.title("Power Usage Comparison: MARL vs Greedy")

plt.legend()
plt.grid(True)
plt.savefig("plots/" +str(int(time.time()))+  "-Power.png", dpi=300, bbox_inches="tight")

# plt.show()


# Latency
# plt.figure()

# plt.plot(iterations, latency_marl, linewidth=2, label="MARL Latency")
# plt.plot(iterations, latency_greedy, linewidth=2, label="Greedy Latency")

# plt.xlabel("Iterations")
# plt.ylabel("Latency (GTs per Mbps)")
# plt.title("Latency Comparison: MARL vs Greedy")

# plt.legend()
# plt.grid(True)
# plt.savefig("plots/" + str(int(time.time())) + "-Latency.png", dpi=300, bbox_inches="tight")

# plt.show()

#Throughput
plt.figure()
plt.plot(iterations, throughput_marl, linewidth=2, label="MARL Throughput")
plt.plot(iterations, throughput_greedy, linewidth=2, label="Greedy Throughput")

plt.xlabel("Iterations")
plt.ylabel("Throughput")
plt.title("Throughput Comparison: MARL vs Greedy")

plt.legend()
plt.grid(True)
plt.savefig("plots/" + str(int(time.time())) + "-Throughput.png", dpi=300, bbox_inches="tight")


# moving window
# plt.figure()
# window = 200

# aoi_window = []
# for i in range(len(aoi_marl)-window+1):
#     temp = aoi_marl[i:i+window]
#     aoi_window.append(np.mean(temp))

# aoi_np_window = np.array(aoi_window)
# plt.plot(np.arange(window, num_iterations + 1),aoi_np_window,label="MARL AoI (Moving Average)")
# plt.xlabel("Iterations")
# plt.ylabel("Throughput")
# plt.title("Moving window average")
# plt.legend()
# plt.grid(True)
# plt.savefig("plots/" + str(int(time.time())) + "-Window.png", dpi=300, bbox_inches="tight")



print("Completed in {:.2f} seconds".format(time.time() - start_time))
print("Packet loss Greedy: ",total_packets_served_greedy/total_packets , "MARL :",total_packets_served_marl/total_packets)
print("Packets served Greedy: ",total_packets_served_greedy , "MARL :",total_packets_served_marl)


# Average only 1 satellite  
 # Remove global staet 

# Give a write up 
# extension
# keep a single agent baseline
# also change around how much state each satellite sees, also local observation
#SARL , MARL incomplete state, MARL full state, greedy
# Try and extend ot other types of queues
# NS tree for satellite comm
