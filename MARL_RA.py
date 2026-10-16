import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx
import os
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"plots/run_{run_id}/"

os.makedirs(run_dir, exist_ok=True)

class AOIPacket:
    def __init__(self, start_step,destination_gt,destination_sat):
        self.start_step = start_step
        self.destination_gt = destination_gt
        self.destination_sat = destination_sat

    def get_age(self, current_step):
        return current_step - self.start_step
        

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

def get_freshness_bin(freshness_gain):
    if freshness_gain == 0:
        return 0
    elif freshness_gain <= 10:
        return 1
    elif freshness_gain <= 30:
        return 2
    else:
        return 3

def system_aaoi(queues, current_step):
    total_aoi, total_len = 0, 0
    for q in queues:
        q_aoi, q_len = get_server_state(q, current_step)
        total_aoi += q_aoi
        total_len += q_len
    return total_aoi / total_len if total_len > 0 else 0.0

PRIORITY_THRESHOLDS = (20, 40)

def get_priority (sat, gt, current_step, last_received):
    receiver_aoi = current_step - last_received[sat, gt]
    if receiver_aoi >= PRIORITY_THRESHOLDS[1]:
        return 0
    elif receiver_aoi >= PRIORITY_THRESHOLDS[0]:
        return 1
    else:
        return 2
    
def get_state(queue, current_step):
    if len(queue) == 0:
        aoi = 0
    else:
        aoi = queue[0].get_age(current_step) # FIFO queue

    queue_len = len(queue)

    #The AOI is between 2 and 8 mostly after the warmup period
    # AoI bins
    if aoi <= 35:
        aoi_bin = 0
    elif aoi <= 40:
        aoi_bin = 1
    elif aoi <= 45:
        aoi_bin = 2
    else:
        aoi_bin = 3

    # Queue length bins
    if queue_len <= 2:
        queue_bin = 0
    elif queue_len <= 5:
        queue_bin = 1
    else:
        queue_bin = 2

    log_aoi_bin = min(int(np.log2(aoi + 1)), num_aoi_bins - 1)
    return log_aoi_bin, queue_bin


def forward_packets(relay_queues, packet_queues, paths, num_satellites, max_queue_length, link_capacity):
    incoming = [[] for _ in range(num_satellites)]
    packets_lost = 0

    for sat in range(num_satellites):
        num_forward = min(link_capacity, len(relay_queues[sat]))

        for _ in range(num_forward):
            packet = relay_queues[sat].pop(0)
            path = paths[sat][packet.destination_sat]
            next_sat = path[1]
            incoming[next_sat].append(packet)

    for sat in range(num_satellites):
        for packet in incoming[sat]:
            if sat == packet.destination_sat:
                gt = packet.destination_gt

                # If the queue is full, we drop the packet
                if len(packet_queues[sat][gt]) < max_queue_length:
                    packet_queues[sat][gt].append(packet)
                else:
                    packets_lost += 1
            else:
                relay_queues[sat].append(packet)
    return packets_lost

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
# num_gt_aoi_bins = 4
# num_aaoi_bins = 4
# num_power_bins = 5
# num_states = num_gt_aoi_bins * num_aaoi_bins * num_power_bins
start_time = time.time()
print_interval = 1000

# def state_to_index(gt_aoi_bin, aaoi_bin, power_bin):
#     return (gt_aoi_bin * num_aaoi_bins * num_power_bins + aaoi_bin * num_power_bins + power_bin)

#################################### below is older code 
num_aoi_bins = 6
num_queue_bins = 3
num_power_bins = 5
num_states = num_aoi_bins * num_queue_bins * num_power_bins
# there are 4 x 3 x5 states, but this state only look at the age of the packet at head of current queue

def state_to_index(aoi_bin, queue_bin,power_bin):
    return (aoi_bin * num_queue_bins*num_power_bins) + (queue_bin *num_power_bins) + power_bin

#-------------------------- ALL PARAMETERS --------------------------------------------------------
num_satellites = 60
num_gts = 30
channels_per_satellite = 2
power_per_mbps = 5
base_power_budget = 120
num_iterations = 30_000
num_freshness_bins = 4

max_queue_length = 20   # This queue is for transmission to each GT
link_capacity = 20   # How many packets are forwarded every iteration globally 
arrival_rate = 2  # This many every iter per satellite on average (Poisson process)

discount_factor = 0.95


# variables init
aoi_marl = np.zeros(num_iterations)
aoi_greedy = np.zeros(num_iterations)
aoi_greedy_2 = np.zeros(num_iterations)

receiver_aoi_p95_marl = np.zeros(num_iterations)
receiver_aoi_p95_greedy = np.zeros(num_iterations)
receiver_aoi_p95_greedy_2 = np.zeros(num_iterations)


receiver_aoi_marl = np.zeros(num_iterations)
receiver_aoi_greedy = np.zeros(num_iterations)
receiver_aoi_greedy_2 = np.zeros(num_iterations)


# for a given action and gt, it stores the generation time of the packet with the lowest AOI
# Stores it for each sat,gt pair
last_received_marl = np.zeros((num_satellites, num_gts), dtype=int)
last_received_greedy = np.zeros((num_satellites, num_gts), dtype=int)
last_received_greedy_2 = np.zeros((num_satellites, num_gts), dtype=int)



throughput_marl = np.zeros(num_iterations)
throughput_greedy = np.zeros(num_iterations)
throughput_greedy_2 = np.zeros(num_iterations)


power_marl = np.zeros(num_iterations)
power_greedy = np.zeros(num_iterations)
power_greedy_2 = np.zeros(num_iterations)


latency_marl = np.zeros(num_iterations)
latency_greedy = np.zeros(num_iterations)
latency_greedy_2 = np.zeros(num_iterations)


selected_sats = np.random.default_rng(42).choice(num_satellites, size=2, replace=False)
sat_to_idx = {sat: i for i, sat in enumerate(selected_sats)}

aoi_marl_sats = np.full((2, num_iterations), np.nan)
aoi_greedy_sats = np.full((2, num_iterations), np.nan)
aoi_greedy_sats_2 = np.full((2, num_iterations), np.nan)



# q_tables[sat][gt, ch]
shared_q_table = np.zeros((num_states, num_gts, channels_per_satellite, num_freshness_bins))
q_tables = [shared_q_table] * num_satellites

# Q-learning / bandit parameters
alpha = 0.01
epsilon_start = 0.2
epsilon_end = 0.001
epsilon_decay = (epsilon_end / epsilon_start) ** (1 / num_iterations)


packets_lost_marl =0;
packets_lost_greedy = 0
packets_lost_greedy_2 = 0


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
packet_queues_marl = [[[] for gt in range(num_gts)] for sat in range(num_satellites)]
packet_queues_greedy = [[[] for gt in range(num_gts)] for sat in range(num_satellites)]
packet_queues_greedy_2 = [[[] for gt in range(num_gts)] for sat in range(num_satellites)]


relay_queues_marl = [[] for _ in range(num_satellites)]
relay_queues_greedy = [[] for _ in range(num_satellites)]
relay_queues_greedy_2 = [[] for _ in range(num_satellites)]


max_aoi = 70

bin_edges = np.linspace(0, max_aoi, num_aoi_bins)

def get_aoi_state(aoi):
    return min(max(np.searchsorted(bin_edges, aoi, side="right") - 1, 0), num_aoi_bins - 1)

# Main sim loop
epsilon = epsilon_start
rewards=[]
total_packets = 0

#A module that creates the paths as a graph

G = nx.Graph()
G.add_nodes_from(range(num_satellites))

for sat in range(num_satellites):
    for offset in range(3):
        neighbor = (sat + offset) % num_satellites
        G.add_edge(sat, neighbor, weight= 1)

paths = dict(nx.all_pairs_dijkstra_path(G, weight="weight"))
pending_updates = [[] for _ in range(num_satellites)]

for iteration in range(num_iterations):
    iter_step = iteration + 1

    # Add new packets
    for sat in range(num_satellites):
        num_arrivals = np.random.poisson(arrival_rate)

        for _ in range(num_arrivals):
            gt = np.random.randint(num_gts)
            destination_sat = np.random.randint(num_satellites)
            # age = np.random.randint(2)

            packet_marl = AOIPacket(iter_step, gt, destination_sat)
            packet_greedy = AOIPacket(iter_step, gt, destination_sat)
            packet_greedy_2 = AOIPacket(iter_step, gt, destination_sat)

            total_packets+=1

            if sat == destination_sat:
                if len(packet_queues_marl[sat][gt]) < max_queue_length:
                    packet_queues_marl[sat][gt].append(packet_marl)
                else:
                    packets_lost_marl+=1
                if len(packet_queues_greedy[sat][gt]) < max_queue_length:
                    packet_queues_greedy[sat][gt].append(packet_greedy)
                else:
                    packets_lost_greedy += 1

                if len(packet_queues_greedy_2[sat][gt]) < max_queue_length:
                    packet_queues_greedy_2[sat][gt].append(packet_greedy_2)
                else:
                    packets_lost_greedy_2 += 1
            else:   
                relay_queues_marl[sat].append(packet_marl)
                relay_queues_greedy[sat].append(packet_greedy)
                relay_queues_greedy_2[sat].append(packet_greedy_2)



    # dynamically change power budgets
    # power_budget = base_power_budget + np.random.randint(50, 101, size=num_satellites)
    power_budget = np.full(num_satellites, base_power_budget)
    packets_lost_marl += forward_packets(relay_queues_marl, packet_queues_marl, paths, num_satellites, max_queue_length, link_capacity)
    packets_lost_greedy += forward_packets(relay_queues_greedy, packet_queues_greedy, paths, num_satellites, max_queue_length, link_capacity)
    packets_lost_greedy_2 += forward_packets(relay_queues_greedy_2, packet_queues_greedy_2, paths, num_satellites, max_queue_length, link_capacity)


    # if iteration % 100 == 0:
    #     print(f"Iteration {iteration}: MARL relay={sum(len(q) for q in relay_queues_marl)}, Greedy relay={sum(len(q) for q in relay_queues_greedy)}")

    # Greedy Algorithm

    total_throughput_greedy = 0
    total_power_greedy = 0
    total_aoi_greedy = 0
    packets_served_greedy = 0

    total_throughput_greedy_2 = 0
    total_power_greedy_2 = 0
    total_aoi_greedy_2 = 0
    packets_served_greedy_2 = 0

    gt_demand_iter = gt_demand_all[iteration].copy()

    for sat in range(num_satellites):
        packets_served_sat = 0
        gt_demand = gt_demand_iter.copy()

        total_power_sat = 0
        total_throughput_sat = 0
        total_aoi_sat = 0

        # power_limit = power_budget[sat]
        power_limit = power_budget[sat]

        # find aoi of oldest packet for each GT
        gt_aois_2 = np.zeros(num_gts)

        for gt in range(num_gts):
            if len(packet_queues_greedy_2[sat][gt]) > 0:
                packet = packet_queues_greedy_2[sat][gt][0]
                gt_aois_2[gt] = packet.get_age(iter_step)
            else:
                gt_aois_2[gt] = -np.inf

        # sort GTs based on AoI, highest first
        gt_priorities_2 = np.array([get_priority(sat, gt, iter_step, last_received_greedy_2) for gt in range(num_gts)])

        gt_indices_2 = sorted(range(num_gts), key=lambda gt: (gt_priorities_2[gt], -gt_aois_2[gt]))

        # Calculate freshness gain for each GT
        gt_gains = np.full(num_gts, -np.inf)

        for gt in range(num_gts):
            if len(packet_queues_greedy[sat][gt]) > 0:
                packet = packet_queues_greedy[sat][gt][0]
                old_timestamp = last_received_greedy[sat, gt]

                gt_gains[gt] = max(0, packet.start_step - old_timestamp)

        # Sort GTs by maximum freshness gain
        gt_priorities = np.array([get_priority(sat, gt, iter_step, last_received_greedy) for gt in range(num_gts)])

        gt_indices = sorted(range(num_gts), key=lambda gt: (gt_priorities[gt], -gt_gains[gt]))

        ch = 0
        index = 0

        while ch < channels_per_satellite and index < num_gts:
            gt = gt_indices_2[index]

            if len(packet_queues_greedy_2[sat][gt]) > 0:
                demand = gt_demand[gt]
                power_consumed = demand * power_per_mbps

                if total_power_sat + power_consumed <= power_limit:
                    packet = packet_queues_greedy_2[sat][gt][0]
                    packet_aoi = packet.get_age(iter_step)

                    packets_served_greedy_2 += 1
                    packets_served_sat +=1

                    total_power_sat += power_consumed
                    total_throughput_sat += demand
                    total_aoi_sat += packet_aoi

                    # Remove served packet
                    last_received_greedy_2[sat, gt] = max(last_received_greedy_2[sat, gt], packet.start_step)
                    packet_queues_greedy_2[sat][gt].pop(0)

                    ch += 1
            index += 1
        if sat in sat_to_idx and packets_served_sat > 0:
            idx = sat_to_idx[sat]
            aoi_greedy_sats_2[idx, iteration] = total_aoi_sat / packets_served_sat

        total_throughput_greedy_2 += total_throughput_sat
        total_power_greedy_2 += total_power_sat
        total_aoi_greedy_2 += total_aoi_sat

        total_power_sat = 0
        total_throughput_sat = 0
        total_aoi_sat = 0
        packets_served_sat = 0
        gt_demand = gt_demand_iter.copy()


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
                    packets_served_sat +=1

                    total_power_sat += power_consumed
                    total_throughput_sat += demand
                    total_aoi_sat += packet_aoi

                    # Remove served packet
                    last_received_greedy[sat, gt] = max(last_received_greedy[sat, gt], packet.start_step)
                    packet_queues_greedy[sat][gt].pop(0)

                    ch += 1
            index += 1
        if sat in sat_to_idx and packets_served_sat > 0:
            idx = sat_to_idx[sat]
            aoi_greedy_sats[idx, iteration] = total_aoi_sat / packets_served_sat

        total_throughput_greedy += total_throughput_sat
        total_power_greedy += total_power_sat
        total_aoi_greedy += total_aoi_sat

    throughput_greedy[iteration] = total_throughput_greedy
    power_greedy[iteration] = total_power_greedy
    latency_greedy[iteration] = num_gts / (total_throughput_greedy + 1e-5)

    throughput_greedy_2[iteration] = total_throughput_greedy_2
    power_greedy_2[iteration] = total_power_greedy_2
    latency_greedy_2[iteration] = num_gts / (total_throughput_greedy_2 + 1e-5)

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
        packets_served_sat =0

        power_limit = power_budget[sat]

        q_table = q_tables[sat]

        # Update previous iteration's actions using the current iteration's state
        if pending_updates[sat]:
            next_available_gts = np.array([gt for gt in range(num_gts) if len(packet_queues_marl[sat][gt]) > 0 and gt_demand[gt] * power_per_mbps <= power_limit])

            if len(next_available_gts) == 0:
                max_next_q = 0.0
            else:
                priorities = np.array([get_priority(sat, gt, iter_step, last_received_marl) for gt in next_available_gts])
                next_available_gts = next_available_gts[priorities == np.min(priorities)]

                next_power_bin = get_power_bin(0, power_limit)
                next_q_values = []

                for next_gt in next_available_gts:
                    next_aoi_bin, next_queue_bin = get_state(packet_queues_marl[sat][next_gt], iter_step)
                    next_state_index = state_to_index(next_aoi_bin, next_queue_bin, next_power_bin)

                    next_packet = packet_queues_marl[sat][next_gt][0]
                    next_freshness_gain = max(0, next_packet.start_step - last_received_marl[sat, next_gt])
                    next_freshness_bin = get_freshness_bin(next_freshness_gain)

                    next_q_values.append(q_table[next_state_index, next_gt, 0, next_freshness_bin])

                max_next_q = np.max(next_q_values)

            for prev_state, prev_action, prev_ch, prev_freshness_bin, prev_reward in pending_updates[sat]:
                current_q = q_table[prev_state, prev_action, prev_ch, prev_freshness_bin]
                target = prev_reward + discount_factor * max_next_q
                q_table[prev_state, prev_action, prev_ch, prev_freshness_bin] = current_q + alpha * (target - current_q)

            pending_updates[sat] = []

        served_gts = np.zeros(num_gts, dtype=bool)

        for ch in range(channels_per_satellite):
            
            # only consider unserved GTs that actually have packets
            available_gts = np.array([gt for gt in range(num_gts) if not served_gts[gt] and len(packet_queues_marl[sat][gt]) > 0 and total_power_sat + gt_demand[gt] * power_per_mbps <= power_limit])

            if len(available_gts) == 0:
                break

            priorities = np.array([get_priority(sat, gt, iter_step, last_received_marl) for gt in available_gts])

            highest_priority = np.min(priorities)
            available_gts = available_gts[priorities == highest_priority]

            if len(available_gts) == 0:
                break

            queues = packet_queues_marl[sat]
            power_bin = get_power_bin(total_power_sat,power_limit)

            # exploration
            if np.random.rand() < epsilon:
                action = np.random.choice(available_gts)

            # exploitation
            else:
                q_values = np.full(num_gts, -np.inf)

                for gt in available_gts:
                    aoi_bin, queue_bin = get_state(packet_queues_marl[sat][gt],iter_step)
                    state_index = state_to_index(aoi_bin,queue_bin, power_bin) #should be local state now
                                
                    packet = packet_queues_marl[sat][gt][0]
                    freshness_gain = max(0, packet.start_step - last_received_marl[sat, gt])
                    freshness_bin = get_freshness_bin(freshness_gain)
                    q_values[gt] = q_table[state_index,gt,ch,freshness_bin]

                action = np.argmax(q_values)
            # State of selected GT BEFORE action
            aoi_bin, queue_bin = get_state(packet_queues_marl[sat][action],iter_step)

            state_index = state_to_index(aoi_bin,queue_bin,power_bin)
            packet = packet_queues_marl[sat][action][0]
            old_timestamp = last_received_marl[sat, action]
            freshness_gain = max(0, packet.start_step - old_timestamp)
            freshness_bin = get_freshness_bin(freshness_gain)

            demand = gt_demand[action]
            power_consumed = demand * power_per_mbps

            served_gts[action] = True

            q_len = len(packet_queues_marl[sat][action])
            if (q_len > 0 and total_power_sat + power_consumed <= power_limit):

                packet = packet_queues_marl[sat][action][0]

                packet_aoi = packet.get_age(iter_step)
                packets_served_marl += 1
                packets_served_sat +=1

                total_power_sat += power_consumed
                total_throughput_sat += demand
                total_aoi_sat += packet_aoi
                total_queue_lengths += q_len

                # Update receivers freshest information timestamp
                old_timestamp = last_received_marl[sat, action]


                # for a given action and gt, it stores the generation time of the packet with the lowest AOI
                last_received_marl[sat, action] = max(old_timestamp, packet.start_step)  # max start step means freshest packet

                packet_queues_marl[sat][action].pop(0)


                reward = freshness_gain
            else:
                # reward = 0
                reward= - power_consumed #penalize
                # print("REWARD penalize:", reward)



            #update
            pending_updates[sat].append((state_index, action, ch, freshness_bin, reward))

            rewards.append(reward) 

        q_tables[sat] = q_table

        total_throughput_marl += total_throughput_sat
        total_power_marl += total_power_sat
        total_aoi_marl += total_aoi_sat
        if sat in sat_to_idx and packets_served_sat > 0:
            idx = sat_to_idx[sat]
            aoi_marl_sats[idx, iteration] = total_aoi_sat / packets_served_sat


    # store the metrics for plots
    throughput_marl[iteration] = total_throughput_marl
    power_marl[iteration] = total_power_marl
    latency_marl[iteration] = num_gts / (total_throughput_marl + 1e-5)


    # Receiver-side AoI
    receiver_aoi_marl[iteration] = np.mean(iter_step - last_received_marl)
    receiver_aoi_greedy[iteration] = np.mean(iter_step - last_received_greedy)
    receiver_aoi_greedy_2[iteration] = np.mean(iter_step - last_received_greedy_2)
    # Calculate average AoI
    if packets_served_marl > 0:
        aoi_marl[iteration] = total_aoi_marl / packets_served_marl
    else:
        aoi_marl[iteration] = 0

    if packets_served_greedy > 0:
        aoi_greedy[iteration] = total_aoi_greedy / packets_served_greedy
    else:
        aoi_greedy[iteration] = 0

    if packets_served_greedy_2 > 0:
        aoi_greedy_2[iteration] = total_aoi_greedy_2 / packets_served_greedy_2
    else:
        aoi_greedy_2[iteration] = 0


    receiver_aoi_p95_marl[iteration] = np.percentile(iter_step - last_received_marl, 95)
    receiver_aoi_p95_greedy[iteration] = np.percentile(iter_step - last_received_greedy, 95)
    receiver_aoi_p95_greedy_2[iteration] = np.percentile(iter_step - last_received_greedy_2, 95)
    # epsilon decay
    epsilon *= epsilon_decay
    if iteration % print_interval == 0:
        print(f"Iteration {iteration + 1}/{num_iterations} completed. Epsilon: {epsilon:.6f} Average AoI MARL: {np.mean(aoi_marl[iteration-print_interval:iteration]):.2f}, Greedy Freshness: {np.mean(aoi_greedy[iteration-print_interval:iteration]):.2f} Greedy Oldest: {np.mean(aoi_greedy_2[iteration-print_interval:iteration]):.2f}")

# Final updates have no future iteration
for sat in range(num_satellites):
    q_table = q_tables[sat]

    for prev_state, prev_action, prev_ch, prev_freshness_bin, prev_reward in pending_updates[sat]:
        current_q = q_table[prev_state, prev_action, prev_ch, prev_freshness_bin]
        q_table[prev_state, prev_action, prev_ch, prev_freshness_bin] = current_q + alpha * (prev_reward - current_q)


# plot the rewards:
plt.figure()
plt.plot(range(len(rewards)), rewards, label="reward per iteration")
plt.xlabel("Iterations")
plt.ylabel("reward")
plt.title("reward per iter")
plt.legend()
plt.grid(True)
plt.savefig(str(run_dir) + str(int(time.time())) +  "-Reward.png", dpi=300, bbox_inches="tight")
# plt.show()

iterations = np.arange(1, num_iterations + 1)

plt.figure()
plt.plot(iterations, receiver_aoi_p95_marl, label="95th percentile AoI")
plt.plot(iterations, receiver_aoi_p95_greedy, label="95th percentile AoI - Freshness Gain Greedy")
plt.plot(iterations, receiver_aoi_p95_greedy_2, label="95th percentile AoI - Oldest Packet Greedy")


plt.xlabel("Iterations")
plt.ylabel("95th percentile")
plt.title("95th percentile")
plt.legend()
plt.grid(True)
plt.savefig( str(run_dir) + str(int(time.time())) +  "-95thpercentile.png", dpi=300, bbox_inches="tight")



# Plots


# AoI
plt.figure()

plt.plot(iterations, receiver_aoi_marl, linewidth=2, label="MARL Receiver AoI")
plt.plot(iterations, receiver_aoi_greedy, linewidth=2, label="Greedy Receiver AoI - Freshness Gain Greedy ")
plt.plot(iterations, receiver_aoi_greedy_2, linewidth=2, label="Greedy Receiver AoI - Oldest Packet Greedy")


plt.xlabel("Iterations")
plt.ylabel("Average AOI")
plt.title("AoI Comparison: MARL vs Greedy")

plt.legend()
plt.grid(True)
plt.savefig(str(run_dir)+ str(int(time.time())) +  "-AOI.png", dpi=300, bbox_inches="tight")
# plt.show()


# window = 200
# smooth_iterations = np.arange(window, num_iterations + 1)

# plt.figure(figsize=(14, 7))

# for i, sat in enumerate(selected_sats):
#     marl_smooth = np.array([np.nanmean(aoi_marl_sats[i, j:j+window]) for j in range(num_iterations-window+1)])
#     greedy_smooth = np.array([np.nanmean(aoi_greedy_sats[i, j:j+window]) for j in range(num_iterations-window+1)])

#     plt.plot(smooth_iterations, marl_smooth, label=f"MARL - Satellite {sat}", linewidth=2)
#     plt.plot(smooth_iterations, greedy_smooth, label=f"Greedy - Satellite {sat}", linewidth=2, linestyle="--")

# plt.xlabel("Iterations")
# plt.ylabel("Average Delivered Packet Age")
# plt.title("AoI Comparison for Two Satellites")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.savefig(str(run_dir) + str(int(time.time())) +  "-selected-AOI.png", dpi=300, bbox_inches="tight")

window = 200
smooth_iterations = np.arange(window, num_iterations + 1)

plt.figure(figsize=(14, 7))

for i, sat in enumerate(selected_sats):
    marl_smooth = np.array([np.nanmean(aoi_marl_sats[i, j:j+window]) for j in range(num_iterations-window+1)])
    greedy_smooth = np.array([np.nanmean(aoi_greedy_sats[i, j:j+window]) for j in range(num_iterations-window+1)])
    greedy_smooth_2 = np.array([np.nanmean(aoi_greedy_sats_2[i, j:j+window]) for j in range(num_iterations-window+1)])


    plt.plot(smooth_iterations, marl_smooth, label=f"MARL - Satellite {sat}", linewidth=2)
    plt.plot(smooth_iterations, greedy_smooth, label=f"Freshness Gain Greedy - Satellite {sat}", linewidth=2, linestyle="--")
    plt.plot(smooth_iterations, greedy_smooth_2, label=f"Oldest Packet Greedy - Satellite {sat}", linewidth=2, linestyle=":")


plt.xlabel("Iterations")
plt.ylabel("Average Delivered Packet Age")
plt.title(f"AoI Moving Average (Window = {window})")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(str(run_dir)+ str(int(time.time())) + "-selected-AOI-smoothed.png", dpi=300, bbox_inches="tight")


# Power
plt.figure()

plt.plot(iterations, power_marl, linewidth=2, label="MARL Power Usage")
plt.plot(iterations, power_greedy, linewidth=2, label="Freshness Gain Greedy Power Usage")
plt.plot(iterations, power_greedy_2, linewidth=2, label="Oldest Packet Greedy Power Usage")


plt.xlabel("Iterations")
plt.ylabel("Total Power Usage (Watts)")
plt.title("Power Usage Comparison: MARL vs Greedy")

plt.legend()
plt.grid(True)
plt.savefig(str(run_dir) +str(int(time.time()))+  "-Power.png", dpi=300, bbox_inches="tight")

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
plt.plot(iterations, throughput_greedy_2, linewidth=2, label="Greedy Throughput")

plt.xlabel("Iterations")
plt.ylabel("Throughput")
plt.title("Throughput Comparison: MARL vs Greedy")

plt.legend()
plt.grid(True)
plt.savefig(str(run_dir) +str(int(time.time()))+  "-Throughput.png", dpi=300, bbox_inches="tight")

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
# plt.ylabel("AOI")
# plt.title("Moving window average")
# plt.legend()
# plt.grid(True)
# plt.savefig(str(run_dir) + str(int(time.time())) + "-Window.png", dpi=300, bbox_inches="tight")


warmup = 2000

print("MARL Receiver AoI:", np.mean(receiver_aoi_marl[warmup:]))
print("Freshness Gain Greedy Receiver AoI:", np.mean(receiver_aoi_greedy[warmup:]))
print("Oldest Packet Greedy Receiver AoI:", np.mean(receiver_aoi_greedy_2[warmup:]))



print("Completed in {:.2f} seconds".format(time.time() - start_time))
print("Packet Loss MARL: ",packets_lost_marl/total_packets,"Freshness Gain Greedy: ",packets_lost_greedy/total_packets, "Oldest Packet Greedy", packets_lost_greedy_2/total_packets )

print("MARL GTs never updated:", np.sum(last_received_marl == 0))
print("Freshness Gain Greedy GTs never updated:", np.sum(last_received_greedy == 0))
print("Oldest Packet Greedy GTs never updated:", np.sum(last_received_greedy_2 == 0))


print("MARL maximum receiver AoI:", np.max(num_iterations - last_received_marl))
print("Freshness Gain Greedy maximum receiver AoI:", np.max(num_iterations - last_received_greedy))
print("Oldest Packet Greedy maximum receiver AoI:", np.max(num_iterations - last_received_greedy_2))


# Average only 1 satellite  
# Remove global staet 

# Give a write up 
# extension
# keep a single agent baseline
# also change around how much state each satellite sees, also local observation
#SARL , MARL incomplete state, MARL full state, greedy
# Try and extend ot other types of queues
# NS tree for satellite comm
