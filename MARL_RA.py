import numpy as np
import matplotlib.pyplot as plt
import time

    


class AOIPacket:
    def __init__(self, start_step, service_rate=None):
        self.start_step = start_step
        if service_rate is None:
            service_rate = globals()['service_rate']
        self.service_time = np.random.exponential(1 / rng.uniform(service_rate - 0.2,service_rate+0.2))

    def get_age(self, current_step):
        return (current_step - self.start_step) + 1

rng = np.random.default_rng(50)
np.random.seed(50)
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

    
def get_state(aoi):
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

max_queue_length = 1
arrival_rate = 1.5 # lambda
service_rate = 0.5




def greedy_algorithm_freshness(
    iteration, iter_step, power_budget, gt_demand_all, packet_queues_greedy,
    busy_until_greedy, active_greedy, aoi_ind_greedy, aoi_one_sat_greedy,
    throughput_greedy, power_greedy, latency_greedy, random_sat,
    num_satellites, num_gts, channels_per_satellite, power_per_mbps,receiver_aoi_greedy
):
    # Greedy Algorithm

    total_throughput_greedy = 0
    total_power_greedy = 0
    total_aoi_greedy = 0
    packets_served_greedy = 0

    gt_demand_iter = gt_demand_all[iteration].copy()

    for sat in range(num_satellites):

        packets_served_sat = 0
        gt_demand = gt_demand_iter.copy()

        total_power_sat = sum(
            tx[2] for tx in active_greedy[sat] if tx is not None
        )
        total_throughput_sat = 0
        total_aoi_sat = 0

        power_limit = power_budget[sat]

        # find aoi of oldest packet for each GT
        gt_aois = np.full(num_gts, -np.inf)

        for gt in range(num_gts):
            
            # Skip this sat if its already in active transmission 
            if any(tx is not None and tx[0] == gt for tx in active_greedy[sat]):
                continue
            if len(packet_queues_greedy[sat][gt]) > 0:
                packet = packet_queues_greedy[sat][gt][0]
                freshness_gain = max(0,receiver_aoi_greedy[gt] -(iter_step - packet.start_step))
                gt_aois[gt] = freshness_gain
            else:
                gt_aois[gt] = -np.inf

        # sort GTs based on freshness now, highest first
        gt_indices = np.argsort(gt_aois)[::-1]

        available_channels = [ch for ch in range(channels_per_satellite)
            if iter_step >= busy_until_greedy[sat][ch]]

        channel_index = 0

        for gt in gt_indices:
            if gt_aois[gt] == -np.inf:
                continue
            if channel_index >= len(available_channels):
                break

            if not packet_queues_greedy[sat][gt]:
                continue

            demand = gt_demand[gt]
            power_consumed = demand * power_per_mbps

            if total_power_sat + power_consumed > power_limit:
                continue

            ch = available_channels[channel_index]
            channel_index += 1

            packet = packet_queues_greedy[sat][gt][0]

            packet_aoi = packet.get_age(iter_step)

            total_aoi_sat += packet_aoi
            packets_served_greedy += 1
            aoi_ind_greedy.append(packet_aoi)

            total_power_sat += power_consumed
            total_throughput_sat += demand

            busy_until_greedy[sat][ch] = iter_step + packet.service_time

            active_greedy[sat][ch] = (gt, packet,power_consumed)
            packet_queues_greedy[sat][gt].pop(0)
            packets_served_sat += 1

        total_throughput_greedy += total_throughput_sat
        total_power_greedy += total_power_sat
        total_aoi_greedy += total_aoi_sat
        if sat == random_sat:
            aoi_one_sat_greedy[iteration] = (
                total_aoi_sat / packets_served_sat
                if packets_served_sat > 0 else 0
            )
    # total_packets_served_greedy +=packets_served_greedy
    throughput_greedy[iteration] = total_throughput_greedy
    power_greedy[iteration] = total_power_greedy
    latency_greedy[iteration] = num_gts / (total_throughput_greedy + 1e-5)
    return total_aoi_greedy, packets_served_greedy


def greedy_algorithm(
    iteration, iter_step, power_budget, gt_demand_all, packet_queues_greedy,
    busy_until_greedy, active_greedy, aoi_ind_greedy, aoi_one_sat_greedy,
    throughput_greedy, power_greedy, latency_greedy, random_sat,
    num_satellites, num_gts, channels_per_satellite, power_per_mbps,receiver_aoi_greedy
):
    # Greedy Algorithm

    total_throughput_greedy = 0
    total_power_greedy = 0
    total_aoi_greedy = 0
    packets_served_greedy = 0

    gt_demand_iter = gt_demand_all[iteration].copy()

    for sat in range(num_satellites):

        packets_served_sat = 0
        gt_demand = gt_demand_iter.copy()

        total_power_sat = sum(
            tx[2] for tx in active_greedy[sat] if tx is not None
        )
        total_throughput_sat = 0
        total_aoi_sat = 0

        power_limit = power_budget[sat]

        # find aoi of oldest packet for each GT
        gt_aois = np.full(num_gts, -np.inf)

        for gt in range(num_gts):
            
            # Skip this sat if its already in active transmission 
            if any(tx is not None and tx[0] == gt for tx in active_greedy[sat]):
                continue
            if len(packet_queues_greedy[sat][gt]) > 0:
                packet = packet_queues_greedy[sat][gt][0]
                gt_aois[gt] = receiver_aoi_greedy[gt]
            else:
                gt_aois[gt] = -np.inf

        # sort GTs based on AoI, highest first
        gt_indices = np.argsort(gt_aois)[::-1]

        available_channels = [ch for ch in range(channels_per_satellite)
            if iter_step >= busy_until_greedy[sat][ch]]

        channel_index = 0

        for gt in gt_indices:
            if gt_aois[gt] == -np.inf:
                continue
            if channel_index >= len(available_channels):
                break

            if not packet_queues_greedy[sat][gt]:
                continue

            demand = gt_demand[gt]
            power_consumed = demand * power_per_mbps

            if total_power_sat + power_consumed > power_limit:
                continue

            ch = available_channels[channel_index]
            channel_index += 1

            packet = packet_queues_greedy[sat][gt][0]

            packet_aoi = packet.get_age(iter_step)

            total_aoi_sat += packet_aoi
            packets_served_greedy += 1
            aoi_ind_greedy.append(packet_aoi)

            total_power_sat += power_consumed
            total_throughput_sat += demand

            busy_until_greedy[sat][ch] = iter_step + packet.service_time

            active_greedy[sat][ch] = (gt, packet,power_consumed)
            packet_queues_greedy[sat][gt].pop(0)
            packets_served_sat += 1

        total_throughput_greedy += total_throughput_sat
        total_power_greedy += total_power_sat
        total_aoi_greedy += total_aoi_sat
        if sat == random_sat:
            aoi_one_sat_greedy[iteration] = (
                total_aoi_sat / packets_served_sat
                if packets_served_sat > 0 else 0
            )
    # total_packets_served_greedy +=packets_served_greedy
    throughput_greedy[iteration] = total_throughput_greedy
    power_greedy[iteration] = total_power_greedy
    latency_greedy[iteration] = num_gts / (total_throughput_greedy + 1e-5)
    return total_aoi_greedy, packets_served_greedy


def marl_algorithm(
    iteration, iter_step, power_budget, gt_demand_all, packet_queues_marl,
    busy_until_marl, active_marl, q_tables, aoi_ind_marl, aoi_one_sat_marl,
    throughput_marl, power_marl, latency_marl, random_sat, epsilon, rewards,
    alpha, discount_factor, num_satellites, num_gts,
    channels_per_satellite, power_per_mbps,receiver_aoi_marl
):
    # MARL Simulation;

    total_throughput_marl = 0
    total_power_marl = 0
    total_aoi_marl = 0
    packets_served_marl = 0

    gt_demand_iter_marl = gt_demand_all[iteration].copy()

    for sat in range(num_satellites):
        packets_served_sat = 0

        gt_demand = gt_demand_iter_marl.copy()

        total_power_sat = sum(
            tx[2] for tx in active_marl[sat] if tx is not None
        )
        total_throughput_sat = 0
        total_aoi_sat = 0
        total_queue_lengths = 0 

        power_limit = power_budget[sat]

        q_table = q_tables[sat]

        served_gts = np.zeros(num_gts, dtype=bool)
        action = None

        for ch in range(channels_per_satellite):
            if iter_step < busy_until_marl[sat][ch]:
                continue

            # only consider unserved GTs that actually have packets
            active_gts = {
                l[0] for l in active_marl[sat]
                if l is not None
            }

            available_gts = np.array([
                gt for gt in range(num_gts)
                if not served_gts[gt]
                and len(packet_queues_marl[sat][gt]) > 0
                and total_power_sat + gt_demand[gt] * power_per_mbps <= power_limit
                and gt not in active_gts
            ])

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
                    aoi_bin = get_state(receiver_aoi_marl[gt])
                    state_index = state_to_index(aoi_bin, power_bin=power_bin)
                    q_values[gt] = q_table[state_index,gt,ch]

                action = np.argmax(q_values)
            # State of selected GT BEFORE action
            aoi_bin = get_state(receiver_aoi_marl[action])

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
                active_marl[sat][ch] = (action, packet,power_consumed)
                packet_queues_marl[sat][action].pop(0)
                packets_served_sat += 1
                

                # aaoi_after = system_aaoi(queues, iter_step)

                # Add power , ta
                freshness_gain = max(0,receiver_aoi_marl[action] - (iter_step - packet.start_step))
                reward = freshness_gain - 0.01 * power_consumed  # if we helped the system then before will be greaater than after and get pos reward
                # print(power_consumed , packet_aoi)
                aoi_ind_marl.append(packet_aoi)


            else:
                # reward = 0
                reward= -power_consumed #penalize
                # print("REWARD penalize:", reward)

            rewards.append(reward) # just out of curiousity track the reward, see if trending up?


            #update
            current_q = q_table[state_index,action,ch]

            next_channel = next(
                (c for c in range(ch + 1, channels_per_satellite)
                if iter_step >= busy_until_marl[sat][c]),
                None
            )


            next_available_gts = [
                gt for gt in range(num_gts)
                if not served_gts[gt]
                and packet_queues_marl[sat][gt]
                and total_power_sat + gt_demand[gt] * power_per_mbps <= power_limit
            ]
            if next_channel is None or len(next_available_gts) == 0:
                max_next_q = 0
            else:
                next_power_bin = get_power_bin(total_power_sat, power_limit)
                # next_aaoi = system_aaoi(packet_queues_marl[sat], iter_step)
                # next_aaoi_bin = bin_aaoi(next_aaoi)

                next_q_values = []

                for next_gt in next_available_gts:
                    next_aoi_bin = get_state(receiver_aoi_marl[next_gt])  
                    next_state_index = state_to_index(next_aoi_bin,next_power_bin)
                    next_q_values.append(q_table[next_state_index, next_gt, next_channel])

                max_next_q = np.max(next_q_values)

            if action is not None:
                target = reward + discount_factor * max_next_q
                q_table[state_index, action, ch] = (current_q + alpha * (target - current_q))

        q_tables[sat] = q_table

        total_throughput_marl += total_throughput_sat
        total_power_marl += total_power_sat
        total_aoi_marl += total_aoi_sat
        if sat == random_sat:
            aoi_one_sat_marl[iteration] = (
                total_aoi_sat / packets_served_sat
                if packets_served_sat > 0 else 0
            )

    # store the metrics for plots
    throughput_marl[iteration] = total_throughput_marl
    power_marl[iteration] = total_power_marl
    latency_marl[iteration] = num_gts / (total_throughput_marl + 1e-5)
    # total_packets_served_marl +=packets_served_marl
    
    return total_aoi_marl, packets_served_marl


def run_simulation(
    num_satellites=num_satellites,
    num_gts=num_gts,
    channels_per_satellite=channels_per_satellite,
    power_per_mbps=power_per_mbps,
    base_power_budget=base_power_budget,
    num_iterations=num_iterations,
    arrival_rate=arrival_rate,
    service_rate=service_rate,
    alpha=0.01,
    discount_factor=0.99,
    epsilon_start=1.0,
    epsilon_end=0.001,
    q_tables_init=None,
    evaluate=False,
    seed=50
):
    global rng
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    start_time = time.time()
    # variables init
    aoi_marl = np.zeros(num_iterations)
    aoi_greedy = np.zeros(num_iterations)
    aoi_greedy_fr = np.zeros(num_iterations)
    

    aoi_ind_marl = []
    aoi_ind_greedy = []
    aoi_ind_greedy_fr = []

    throughput_marl = np.zeros(num_iterations)
    throughput_greedy = np.zeros(num_iterations)
    throughput_greedy_fr = np.zeros(num_iterations)
    
    power_marl = np.zeros(num_iterations)
    power_greedy = np.zeros(num_iterations)
    power_greedy_fr = np.zeros(num_iterations)
    

    latency_marl = np.zeros(num_iterations)
    latency_greedy = np.zeros(num_iterations)
    latency_greedy_fr = np.zeros(num_iterations)
    

    receiver_aoi_marl = np.zeros(num_gts)
    receiver_aoi_greedy = np.zeros(num_gts)
    receiver_aoi_greedy_fr = np.zeros(num_gts)
    

    last_received_marl = np.full(num_gts, -np.inf)
    last_received_greedy = np.full(num_gts, -np.inf)
    last_received_greedy_fr = np.full(num_gts, -np.inf)
    


    # q_tables[sat][gt, ch]
    if q_tables_init is None:
        q_tables = [
            np.zeros((num_states, num_gts, channels_per_satellite))
            for _ in range(num_satellites)
        ]
    else:
        q_tables = [q.copy() for q in q_tables_init]
    # Q-learning / bandit parameters
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
    packet_queues_marl = [[[] for _ in range(num_gts)] for _ in range(num_satellites)]
    packet_queues_greedy = [[[] for _ in range(num_gts)] for _ in range(num_satellites)]  # capacity is 1
    packet_queues_greedy_fr = [[[] for _ in range(num_gts)] for _ in range(num_satellites)]  # capacity is 1
    

    active_marl = [[None] * channels_per_satellite
                   for _ in range(num_satellites)]

    active_greedy = [[None] * channels_per_satellite
                     for _ in range(num_satellites)]
    
    active_greedy_fr = [[None] * channels_per_satellite
                     for _ in range(num_satellites)]
    

    #Removed this so that the initial queues are empty. 
    # Adds initial packets with arrival times -5 to -1
    # for sat in range(num_satellites):
    #     for gt in range(num_gts):
    #         for packet_time in range(-5, 0):
    #             packet_queues_marl[sat][gt].append(AOIPacket(packet_time))
    #             packet_queues_greedy[sat][gt].append(AOIPacket(packet_time))


    # Main sim loop
    epsilon = 0.0 if evaluate else epsilon_start
    rewards=[]

    total_packets_served_marl = 0
    total_packets_served_greedy = 0
    total_packets_served_greedy_fr = 0
    total_packets = 0

    busy_until_marl = np.zeros((num_satellites, channels_per_satellite))
    busy_until_greedy = np.zeros((num_satellites, channels_per_satellite))
    busy_until_greedy_fr = np.zeros((num_satellites, channels_per_satellite))
    

    aoi_one_sat_marl = np.zeros(num_iterations)
    aoi_one_sat_greedy = np.zeros(num_iterations)
    aoi_one_sat_greedy_fr = np.zeros(num_iterations)
    

    random_sat = np.random.randint(num_satellites)

    dropped_marl = 0
    dropped_greedy = 0
    dropped_greedy_fr = 0
    
    for iteration in range(num_iterations):
        completed_marl = 0
        completed_greedy = 0
        completed_greedy_fr = 0
        iter_step = iteration + 1

        # age all receivers by 1 
        receiver_aoi_marl += 1
        receiver_aoi_greedy += 1
        receiver_aoi_greedy_fr += 1

        for sat in range(num_satellites):
            for ch in range(channels_per_satellite):

                if active_marl[sat][ch] is not None:
                    gt, packet, power_consumed = active_marl[sat][ch]

                    if iter_step >= busy_until_marl[sat][ch]:
                        total_packets_served_marl += 1
                        completed_marl+=1

                        if packet.start_step > last_received_marl[gt]:
                            last_received_marl[gt] = packet.start_step
                            receiver_aoi_marl[gt] = iter_step - packet.start_step   # time since this packet was generated  

                        active_marl[sat][ch] = None

                if active_greedy[sat][ch] is not None:
                    gt, packet,power_consumed = active_greedy[sat][ch]

                    if iter_step >= busy_until_greedy[sat][ch]:
                        total_packets_served_greedy += 1
                        completed_greedy+=1

                        if packet.start_step > last_received_greedy[gt]:
                            last_received_greedy[gt] = packet.start_step
                            receiver_aoi_greedy[gt] = iter_step - packet.start_step

                        active_greedy[sat][ch] = None
                if active_greedy_fr[sat][ch] is not None:
                    gt, packet,power_consumed = active_greedy_fr[sat][ch]

                    if iter_step >= busy_until_greedy_fr[sat][ch]:
                        total_packets_served_greedy_fr += 1
                        completed_greedy_fr+=1

                        if packet.start_step > last_received_greedy_fr[gt]:
                            last_received_greedy_fr[gt] = packet.start_step
                            receiver_aoi_greedy_fr[gt] = iter_step - packet.start_step

                        active_greedy_fr[sat][ch] = None

        # Add new packets
        for sat in range(num_satellites):
            for gt in range(num_gts):
                rate = rng.uniform(arrival_rate - 0.5, arrival_rate + 0.5)
                num_arrivals = np.random.poisson(rate)

                for _ in range(num_arrivals):
                    packet = AOIPacket(iter_step, service_rate)
                    total_packets += 1

                    if len(packet_queues_marl[sat][gt]) < max_queue_length:
                        packet_queues_marl[sat][gt].append(packet)
                    else:
                        dropped_marl += 1

                    if len(packet_queues_greedy[sat][gt]) < max_queue_length:
                        packet_queues_greedy[sat][gt].append(packet)
                    else:
                        dropped_greedy += 1
                        
                    if len(packet_queues_greedy_fr[sat][gt]) < max_queue_length:
                        packet_queues_greedy_fr[sat][gt].append(packet)
                    else:
                        dropped_greedy_fr += 1


        # dynamically change power budgets
        power_budget = base_power_budget + np.random.randint(50, 101, size=num_satellites)

        total_aoi_greedy, packets_served_greedy = greedy_algorithm(
            iteration, iter_step, power_budget, gt_demand_all, packet_queues_greedy,
            busy_until_greedy, active_greedy, aoi_ind_greedy, aoi_one_sat_greedy,
            throughput_greedy, power_greedy, latency_greedy, random_sat,
            num_satellites, num_gts, channels_per_satellite, power_per_mbps,receiver_aoi_greedy
        )


        total_aoi_marl, packets_served_marl = marl_algorithm(
            iteration, iter_step, power_budget, gt_demand_all, packet_queues_marl,
            busy_until_marl, active_marl, q_tables, aoi_ind_marl, aoi_one_sat_marl,
            throughput_marl, power_marl, latency_marl, random_sat, epsilon, rewards,
            0.0 if evaluate else alpha, discount_factor, num_satellites, num_gts,
            channels_per_satellite, power_per_mbps,receiver_aoi_marl
        )
        
        total_aoi_greedy_2, packets_served_greedy_2 = greedy_algorithm_freshness(
                    iteration, iter_step, power_budget, gt_demand_all, packet_queues_greedy_fr,
                    busy_until_greedy_fr, active_greedy_fr, aoi_ind_greedy_fr, aoi_one_sat_greedy_fr,
                    throughput_greedy_fr, power_greedy_fr, latency_greedy_fr, random_sat,
                    num_satellites, num_gts, channels_per_satellite, power_per_mbps,receiver_aoi_greedy_fr
                )
        
        

        throughput_marl[iteration] = completed_marl
        throughput_greedy[iteration] = completed_greedy
        throughput_greedy_fr[iteration] = completed_greedy_fr
        # Calculate average AoI
        aoi_marl[iteration] = np.mean(receiver_aoi_marl)
        aoi_greedy[iteration] = np.mean(receiver_aoi_greedy)
        aoi_greedy_fr[iteration] = np.mean(receiver_aoi_greedy_fr)

        # epsilon decay
        epsilon *= epsilon_decay
        if (iteration + 1) % print_interval == 0:
            start = iteration + 1 - print_interval
            print(
        f"Iteration {iteration + 1}/{num_iterations} "
        f"Epsilon: {epsilon:.6f} "
        f"MARL AoI: {np.mean(aoi_marl[start:iteration+1]):.2f} "
        f"Greedy AoI: {np.mean(aoi_greedy[start:iteration+1]):.2f}"
        f"Greedy AoI: {np.mean(aoi_greedy_fr[start:iteration+1]):.2f}"
        
    )

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
    plt.plot(iterations, aoi_greedy_fr, linewidth=2, label="Greedy AAoI")
    

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
    plt.plot(iterations, power_greedy_fr, linewidth=2, label="Greedy Power Usage")
    

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
    plt.plot(iterations, throughput_greedy_fr, linewidth=2, label="Greedy Throughput")
    

    plt.xlabel("Iterations")
    plt.ylabel("Packets delivered per timestep")
    plt.title("Throughput Comparison: MARL vs Greedy")

    plt.legend()
    plt.grid(True)
    plt.savefig("plots/" + str(int(time.time())) + "-Throughput.png", dpi=300, bbox_inches="tight")



    # One sat AOI
    window = 200

    marl_smooth = np.convolve(
        aoi_one_sat_marl, np.ones(window) / window, mode="valid"
    )

    greedy_smooth = np.convolve(
        aoi_one_sat_greedy, np.ones(window) / window, mode="valid"
    )
    
    greedy_smooth_fr = np.convolve(
        aoi_one_sat_greedy_fr, np.ones(window) / window, mode="valid"
    )

    plt.figure(figsize=(12, 5))
    plt.plot(marl_smooth, label="MARL")
    plt.plot(greedy_smooth, label="Greedy")
    plt.plot(greedy_smooth_fr, label="Greedy")
    

    plt.xlabel("Iterations")
    plt.ylabel("Average Packet AoI")
    plt.title(f"Satellite {random_sat}: AoI Moving Average")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/" + str(int(time.time())) + "-one_sat.png", dpi=300, bbox_inches="tight")


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
    # print("Packet loss Greedy: ",total_packets_served_greedy/total_packets , "MARL :",total_packets_served_marl/total_packets)
    loss_greedy = dropped_greedy / total_packets if total_packets else 0
    loss_greedy_fr = dropped_greedy_fr / total_packets if total_packets else 0
    loss_marl = dropped_marl / total_packets if total_packets else 0

    print("Packet loss Greedy:", loss_greedy)
    print("Packet loss MARL:", loss_marl)

    print("MARL Average Receiver AoI:", np.mean(aoi_marl))
    print("Greedy Average Receiver AoI:", np.mean(aoi_greedy))
    print("Greedy Average Receiver AoI:", np.mean(aoi_greedy_fr))
    
    return {
        "aoi_marl": aoi_marl,
        "aoi_greedy": aoi_greedy,
        "aoi_greedy_fr": aoi_greedy_fr,
        "throughput_marl": throughput_marl,
        "throughput_greedy": throughput_greedy,
        "throughput_greedy_fr": throughput_greedy_fr,
        "power_marl": power_marl,
        "power_greedy": power_greedy,
        "latency_marl": latency_marl,
        "latency_greedy": latency_greedy,
        "loss_marl": loss_marl,
        "loss_greedy": loss_greedy,
        "power_greedy_fr": power_greedy_fr,
        "loss_greedy_fr": loss_greedy_fr,
        "q_tables": q_tables,
    }


if __name__ == "__main__":
    # training 
    train = run_simulation(
        num_iterations=40000,
        seed=50,
        num_satellites=60,
        num_gts=200
    )

    print("\n\n-----------------Evaluation------------ \n\n")

    # Eval on random traffic
    times =0
    if times < 2:
        for seed in ([20,78]):
            test = run_simulation(
                num_iterations=5000,
                q_tables_init=train["q_tables"],
                evaluate=True,
                seed=seed,
                num_satellites=60,
                num_gts=200,
            )

            print(
                seed,
                "MARL:", np.mean(test["aoi_marl"]),
                "Greedy:", np.mean(test["aoi_greedy"])
            )
            times+=1
    q_tables = np.array(train["q_tables"])

    np.savetxt("q_tables.txt",q_tables.reshape(-1, q_tables.shape[-1]))

# Average only 1 satellite  
# keep a single agent baseline
# also change around how much state each satellite sees, also local observation
# SARL , MARL incomplete state, MARL full state, greedy
# Try and extend to other types of queues
# NS tree for satellite comm
