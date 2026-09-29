import numpy as np
import matplotlib.pyplot as plt


class AOIPacket:
    def __init__(self, start_step):
        self.start_step = start_step

    def get_age(self, current_step):
        return current_step - self.start_step


def get_state(queue, current_step):
    if len(queue) == 0:
        aoi = 0
    else:
        aoi = queue[0].get_age(current_step)

    queue_len = len(queue)

    # AoI bins
    if aoi <= 5:
        aoi_bin = 0
    elif aoi <= 10:
        aoi_bin = 1
    elif aoi <= 20:
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

    return aoi_bin, queue_bin


num_aoi_bins = 4
num_queue_bins = 3
num_states = num_aoi_bins * num_queue_bins


def state_to_index(aoi_bin, queue_bin):
    return aoi_bin * num_queue_bins + queue_bin

# params
num_satellites = 60
num_gts = 20
channels_per_satellite = 10
power_per_mbps = 5
base_power_budget = 500
num_iterations = 5000

max_queue_length = 10
new_packet_arrival = 0.5


# variables init
aoi_marl = np.zeros(num_iterations)
aoi_greedy = np.zeros(num_iterations)

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

#Removed this so that the initial queues are empty. 
# Adds initial packets with arrival times -5 to -1
# for sat in range(num_satellites):
#     for gt in range(num_gts):
#         for packet_time in range(-5, 0):
#             packet_queues_marl[sat][gt].append(AOIPacket(packet_time))
#             packet_queues_greedy[sat][gt].append(AOIPacket(packet_time))


# Main sim loop
epsilon = epsilon_start

for iteration in range(num_iterations):
    iter_step = iteration + 1

    # Add new packets
    for sat in range(num_satellites):
        for gt in range(num_gts):
            if np.random.rand() < new_packet_arrival:

                if len(packet_queues_greedy[sat][gt]) < max_queue_length:
                    packet_queues_greedy[sat][gt].append(AOIPacket(iter_step))

                if len(packet_queues_marl[sat][gt]) < max_queue_length:
                    packet_queues_marl[sat][gt].append(AOIPacket(iter_step))


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
                    packet_queues_greedy[sat][gt].pop(0)

            ch += 1
            index += 1

        total_throughput_greedy += total_throughput_sat
        total_power_greedy += total_power_sat
        total_aoi_greedy += total_aoi_sat

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

        power_limit = power_budget[sat]

        q_table = q_tables[sat]

        served_gts = np.zeros(num_gts, dtype=bool)

        for ch in range(channels_per_satellite):

            # only consider unserved GTs that actually have packets
            available_gts = np.array([gt for gt in range(num_gts)  if not served_gts[gt] and len(packet_queues_marl[sat][gt]) > 0])

            if len(available_gts) == 0:
                break

            # exploration
            if np.random.rand() < epsilon:
                action = np.random.choice(available_gts)

            # exploitation
            else:
                q_values = np.full(num_gts, -np.inf)

                for gt in available_gts:
                    aoi_bin, queue_bin = get_state(packet_queues_marl[sat][gt],iter_step)
                    state_index = state_to_index(aoi_bin,queue_bin)
                    q_values[gt] = q_table[state_index,gt,ch]

                action = np.argmax(q_values)
            # State of selected GT BEFORE action
            aoi_bin, queue_bin = get_state(packet_queues_marl[sat][action],iter_step)

            state_index = state_to_index(aoi_bin,queue_bin)

            demand = gt_demand[action]
            power_consumed = demand * power_per_mbps

            served_gts[action] = True

            if (len(packet_queues_marl[sat][action]) > 0 and total_power_sat + power_consumed <= power_limit):

                packet = packet_queues_marl[sat][action][0]
                packet_aoi = packet.get_age(iter_step)
                packets_served_marl += 1

                total_power_sat += power_consumed
                total_throughput_sat += demand
                total_aoi_sat += packet_aoi

                packet_queues_marl[sat][action].pop(0)

                # Reward
                reward = packet_aoi

            else:
                reward = 0

            # bandit update
            current_q = q_table[state_index,action,ch]

            q_table[state_index,action,ch] = (current_q + alpha * (reward - current_q))

        q_tables[sat] = q_table

        total_throughput_marl += total_throughput_sat
        total_power_marl += total_power_sat
        total_aoi_marl += total_aoi_sat


    # store the metrics for plots
    throughput_marl[iteration] = total_throughput_marl
    power_marl[iteration] = total_power_marl
    latency_marl[iteration] = num_gts / (total_throughput_marl + 1e-5)


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


# Plots

iterations = np.arange(1, num_iterations + 1)


# AoI
plt.figure()

plt.plot(iterations, aoi_marl, linewidth=2, label="MARL AoI")
plt.plot(iterations, aoi_greedy, linewidth=2, label="Greedy AoI")

plt.xlabel("Iterations")
plt.ylabel("Average AOI")
plt.title("AoI Comparison: MARL vs Greedy")

plt.legend()
plt.grid(True)
plt.show()


# Power
plt.figure()

plt.plot(iterations, power_marl, linewidth=2, label="MARL Power Usage")
plt.plot(iterations, power_greedy, linewidth=2, label="Greedy Power Usage")

plt.xlabel("Iterations")
plt.ylabel("Total Power Usage (Watts)")
plt.title("Power Usage Comparison: MARL vs Greedy")

plt.legend()
plt.grid(True)
plt.show()


# Latency
plt.figure()

plt.plot(iterations, latency_marl, linewidth=2, label="MARL Latency")
plt.plot(iterations, latency_greedy, linewidth=2, label="Greedy Latency")

plt.xlabel("Iterations")
plt.ylabel("Latency (GTs per Mbps)")
plt.title("Latency Comparison: MARL vs Greedy")

plt.legend()
plt.grid(True)
plt.show()