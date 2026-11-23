import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from cycler import cycler

plt.rcParams["axes.prop_cycle"] = (
    cycler(color=[
        "blue",
        "green",
        "black",
        "orange",
    ])
    + cycler(linewidth=[2.5, 2.5, 2.5, 2.5])
)

plt.rcParams["axes.prop_cycle"] += cycler(alpha=[0.9, 0.85, 0.9, 0.85])

class AOIPacket:
    def __init__(self, start_step, service_rate=None):
        self.start_step = start_step
        self.delivered = True
        if service_rate is None:
            service_rate = globals()["service_rate"]
        self.service_time = np.random.exponential(
            1 / rng.uniform(service_rate - 0.2, service_rate + 0.2)
        )

    def get_age(self, current_step):
        return (current_step - self.start_step) + 1


LOG = 1.5
rng = np.random.default_rng(50)
np.random.seed(50)


def get_server_state(queue, current_step):
    """
    instead of get AOI of only one packet in queue, get AOI of ALL the packets in the queue
    return the aoi in the queue and the queue legnth

    later use this to calculate the avergae AOI of system (use summation of queu_len , all over summation of queruelen)
    """
    queue_aoi = 0
    for i in range(len(queue)):
        try:
            i_aoi = queue[i].get_age(current_step)
        except:
            i_aoi = 0
        queue_aoi += i_aoi
    # Queue length bins

    queue_len = len(queue)

    return queue_aoi, queue_len


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

    return min(int(np.log(max(1, aoi)) / np.log(LOG)), num_gt_aoi_bins - 1)


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


def get_packet_age_bin(packet, iter_step):
    age = iter_step - packet.start_step

    if age <= 1:
        return 0
    elif age <= 3:
        return 1
    elif age <= 7:
        return 2
    else:
        return 3


# the below is new code to add server aware AAOI
num_gt_aoi_bins = 12
num_power_bins = 5
num_packet_age_bins = 1
num_demand_bins = 3
num_states = num_gt_aoi_bins * num_packet_age_bins * num_power_bins * num_demand_bins
print_interval = 1000


def state_to_index_q(aoi_bin, power_bin, demand_bin, link_bin):
    return (
        (aoi_bin * num_power_bins + power_bin) * num_demand_bins + demand_bin
    ) * num_link_bins + link_bin


# def state_to_index(gt_aoi_bin, packet_age_bin, power_bin):
#     return (
#         gt_aoi_bin * num_packet_age_bins * num_power_bins
#         + packet_age_bin * num_power_bins
#         + power_bin
#     )


# params
num_satellites = 60
num_gts = 20
channels_per_satellite = 10
power_per_mbps = 5
base_power_budget = 500
num_iterations = 20_000

#############################
MAX_QUEUE_LEN = 1  # CHANGE THIS TO MAKE IT M/M/1/K
#############################

arrival_rate = 1.5  # lambda
service_rate = 0.5

AOI_FACTOR = 0.4
POWER_FACTOR = 1 - AOI_FACTOR

LAMBDA = 0.05

LINK_BAD_FRAC = 0.2  # fraction of links that are poor
HIST_NOISE = 0.05  # error in the values
LINK_TRANSITION_P = 0.2 # used for the 2 state Markov chain 

num_link_bins = 3


num_states_q = num_gt_aoi_bins * num_power_bins * num_demand_bins * num_link_bins

# these are used for the stateful MARL agent
mrng = np.random.default_rng(
    1
)  # exploration RNG, keeps traffic identical across variants

# num_states_q = (
#     4 * num_packet_age_bins * num_power_bins
# )  # original: 4 AoI bins -> 80 states


def get_state_q(aoi):
    return min(int(np.log(max(1, aoi)) / np.log(LOG)), num_gt_aoi_bins - 1)


#######################


# these are for contextual bandit
def get_state(aoi):
    return min(int(np.log(max(1, aoi)) / np.log(LOG)), num_gt_aoi_bins - 1)


def get_demand_bin(demand):
    return min(int(demand // 6), num_demand_bins - 1)


def get_link_bin(p):
    return 0 if p < 0.4 else (1 if p < 0.75 else 2)


def state_to_index(aoi_bin, age_bin, power_bin, demand_bin):
    return (
        (aoi_bin * num_packet_age_bins + age_bin) * num_power_bins + power_bin
    ) * num_demand_bins + demand_bin


def greedy_algorithm_freshness(
    iteration,
    iter_step,
    power_budget,
    gt_demand_all,
    packet_queues_greedy,
    busy_until_greedy,
    active_greedy,
    aoi_ind_greedy,
    aoi_one_sat_greedy,
    throughput_greedy,
    power_greedy,
    latency_greedy,
    random_sat,
    num_satellites,
    num_gts,
    channels_per_satellite,
    power_per_mbps,
    receiver_aoi_greedy,
    last_received_greedy_fr,
    link_hist,
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

        total_power_sat = sum(tx[2] for tx in active_greedy[sat] if tx is not None)
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
                # freshness_gain = max(0,receiver_aoi_greedy[gt] -(iter_step - packet.start_step))

                local_aoi = iter_step - last_received_greedy_fr[sat][gt]

                freshness_gain = max(0, local_aoi - (iter_step - packet.start_step))

                # if no freshness gained, dont transmit
                # if freshness_gain <= 0 :
                #     continue
                gt_aois[gt] = link_hist[sat][gt] * freshness_gain
            else:
                gt_aois[gt] = -np.inf

        # sort GTs based on freshness now, highest first
        gt_indices = np.argsort(gt_aois)[::-1]

        available_channels = [
            ch
            for ch in range(channels_per_satellite)
            if iter_step >= busy_until_greedy[sat][ch]
        ]

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

            active_greedy[sat][ch] = (gt, packet, power_consumed)
            packet_queues_greedy[sat][gt].pop(0)
            packets_served_sat += 1

        total_throughput_greedy += total_throughput_sat
        total_power_greedy += total_power_sat
        total_aoi_greedy += total_aoi_sat
    # total_packets_served_greedy +=packets_served_greedy
    throughput_greedy[iteration] = total_throughput_greedy
    power_greedy[iteration] = total_power_greedy
    latency_greedy[iteration] = num_gts / (total_throughput_greedy + 1e-5)
    return total_aoi_greedy, packets_served_greedy


def greedy_algorithm(
    iteration,
    iter_step,
    power_budget,
    gt_demand_all,
    packet_queues_greedy,
    busy_until_greedy,
    active_greedy,
    aoi_ind_greedy,
    aoi_one_sat_greedy,
    throughput_greedy,
    power_greedy,
    latency_greedy,
    random_sat,
    num_satellites,
    num_gts,
    channels_per_satellite,
    power_per_mbps,
    receiver_aoi_greedy,
    last_received_greedy,
    link_hist,
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

        total_power_sat = sum(tx[2] for tx in active_greedy[sat] if tx is not None)
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
                gt_aois[gt] = receiver_aoi_greedy[sat][gt]
            else:
                gt_aois[gt] = -np.inf

        # sort GTs based on AoI, highest first
        gt_indices = np.argsort(gt_aois)[::-1]

        available_channels = [
            ch
            for ch in range(channels_per_satellite)
            if iter_step >= busy_until_greedy[sat][ch]
        ]

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

            active_greedy[sat][ch] = (gt, packet, power_consumed)
            packet_queues_greedy[sat][gt].pop(0)
            packets_served_sat += 1

        total_throughput_greedy += total_throughput_sat
        total_power_greedy += total_power_sat
        total_aoi_greedy += total_aoi_sat
    # total_packets_served_greedy +=packets_served_greedy
    throughput_greedy[iteration] = total_throughput_greedy
    power_greedy[iteration] = total_power_greedy
    latency_greedy[iteration] = num_gts / (total_throughput_greedy + 1e-5)
    return total_aoi_greedy, packets_served_greedy


# changed now to a bandit approach, no future look ahead. So all learaning is done with the current available information
def marl_algorithm(
    iteration,
    iter_step,
    power_budget,
    gt_demand_all,
    packet_queues_marl,
    busy_until_marl,
    active_marl,
    q_table,
    aoi_ind_marl,
    aoi_one_sat_marl,
    throughput_marl,
    power_marl,
    latency_marl,
    random_sat,
    epsilon,
    rewards,
    alpha,
    num_satellites,
    num_gts,
    channels_per_satellite,
    power_per_mbps,
    receiver_aoi_marl,
    last_received_marl,
    link_hist
):
    total_throughput_marl = 0
    total_power_marl = 0
    total_aoi_marl = 0
    packets_served_marl = 0
    gt_demand = gt_demand_all[iteration]

    for sat in range(num_satellites):
        queues = packet_queues_marl[sat]
        power_limit = power_budget[sat]
        total_power_sat = sum(tx[2] for tx in active_marl[sat] if tx is not None)
        total_throughput_sat = 0
        total_aoi_sat = 0

        # Gts that are already in transmission are not considered
        served_gts = np.zeros(num_gts, dtype=bool)
        for tx in active_marl[sat]:
            if tx is not None:
                served_gts[tx[0]] = True

        for ch in range(channels_per_satellite):
            if iter_step < busy_until_marl[sat][ch]:
                continue

            candidates = [
                gt
                for gt in range(num_gts)
                if not served_gts[gt]
                and queues[gt]
                and total_power_sat + gt_demand[gt] * power_per_mbps <= power_limit
            ]
            if not candidates:
                break

            # power_bin = get_power_bin(total_power_sat, power_limit)
            # idx = [
            #     state_to_index(
            #         get_state(iter_step - last_received_marl[sat][gt]),
            #         get_packet_age_bin(queues[gt][0], iter_step),
            #         power_bin,
            #         get_demand_bin(gt_demand[gt]),
            #     )
            #     for gt in candidates
            # ]

            idx = [
                state_to_index_q(
                    get_state(iter_step - last_received_marl[sat][gt]),
                    get_power_bin(total_power_sat, power_limit),
                    get_demand_bin(gt_demand[gt]),
                    get_link_bin(link_hist[sat][gt])
                    # 0
                )
                for gt in candidates
            ]

            if mrng.random() < epsilon:
                k = int(mrng.integers(len(candidates)))
            else:
                qv = q_table[idx]
                best = np.flatnonzero(np.isclose(qv, qv.max()))
                # break ties by true AoI so bins do not lose ordering
                k = max(
                    best,
                    key=lambda i: iter_step - last_received_marl[sat][candidates[i]],
                )

            action, s = candidates[k], idx[k]
            demand = gt_demand[action]
            power_consumed = demand * power_per_mbps
            packet = queues[action].pop(0)

            local_aoi = iter_step - last_received_marl[sat][action]
            freshness_gain = max(0, local_aoi - (iter_step - packet.start_step))
            
            # reward = ( freshness_gain / (demand)) if packet.delivered else (-power_consumed) # so the reward is aoi gain per mbps
            reward = ( freshness_gain / (demand))

            q_table[s] += alpha * (reward - q_table[s])
            rewards.append(reward)

            packet_aoi = packet.get_age(iter_step)
            served_gts[action] = True
            packets_served_marl += 1
            total_power_sat += power_consumed
            total_throughput_sat += demand
            total_aoi_sat += packet_aoi
            busy_until_marl[sat][ch] = iter_step + packet.service_time
            active_marl[sat][ch] = (action, packet, power_consumed)
            aoi_ind_marl.append(packet_aoi)

        total_throughput_marl += total_throughput_sat
        total_power_marl += total_power_sat
        total_aoi_marl += total_aoi_sat

    throughput_marl[iteration] = total_throughput_marl
    power_marl[iteration] = total_power_marl
    latency_marl[iteration] = num_gts / (total_throughput_marl + 1e-5)
    return total_aoi_marl, packets_served_marl


def marl_algorithm_qlearning(
    iteration,
    iter_step,
    power_budget,
    gt_demand_all,
    packet_queues_marl,
    busy_until_marl,
    active_marl,
    q_tables,
    aoi_ind_marl,
    aoi_one_sat_marl,
    throughput_marl,
    power_marl,
    latency_marl,
    random_sat,
    epsilon,
    rewards,
    alpha,
    discount_factor,
    num_satellites,
    num_gts,
    channels_per_satellite,
    power_per_mbps,
    receiver_aoi_marl,
    last_received_marl,
    link_hist,
    pend,
):
    total_throughput_marl = 0
    total_power_marl = 0
    total_aoi_marl = 0
    packets_served_marl = 0
    gt_demand = gt_demand_all[iteration]
    
    # pend_a == 1 if GT is served that iter, 0 if not 
    pend_s, pend_a, pend_r, pend_t = pend["s"], pend["a"], pend["r"], pend["t"]
    total_throughput_marl = 0
    for sat in range(num_satellites):
        power_limit = power_budget[sat]
        total_power_sat = sum(tx[2] for tx in active_marl[sat] if tx is not None)
        total_throughput_sat = 0
        total_aoi_sat = 0
        q_table = q_tables[sat]
        served_gts = np.zeros(num_gts, dtype=bool)
        action = None
        
        # Basically save state across iterations so that reward can have future 
        active_now = {tx[0] for tx in active_marl[sat] if tx is not None}
        has_free_channel = any(iter_step >= busy_until_marl[sat][ch] for ch in range(channels_per_satellite))

        decision_gts = [
            gt for gt in range(num_gts)
            if has_free_channel
            and packet_queues_marl[sat][gt]
            and gt not in active_now
            and total_power_sat + gt_demand[gt] * power_per_mbps <= power_limit
        ]
        
        
        power_bin = get_power_bin(total_power_sat, power_limit)
        s_of = {gt: state_to_index_q(
                 get_state_q(iter_step - last_received_marl[sat][gt]),
                 power_bin,
                 get_demand_bin(gt_demand[gt]),
                 get_link_bin(link_hist[sat][gt])
                # 0
                 )
             for gt in decision_gts}
        
        for gt in decision_gts:
            ps = pend_s[sat, gt]
            if ps >= 0:
                pa = pend_a[sat, gt]
                dt = iter_step - pend_t[sat, gt]
                target = pend_r[sat, gt] + discount_factor ** dt * q_table[s_of[gt]].max()
                q_table[ps, pa] += alpha * (target - q_table[ps, pa])

        for ch in range(channels_per_satellite):
            if iter_step < busy_until_marl[sat][ch]:
                continue

            active_gts = {tx[0] for tx in active_marl[sat] if tx is not None}
            available_gts = np.array(
                [
                    gt
                    for gt in range(num_gts)
                    if not served_gts[gt]
                    and len(packet_queues_marl[sat][gt]) > 0
                    and total_power_sat + gt_demand[gt] * power_per_mbps <= power_limit
                    and gt not in active_gts
                ]
            )
            if len(available_gts) == 0:
                break

            power_bin = get_power_bin(total_power_sat, power_limit)

            if mrng.random() < epsilon:
                action = mrng.choice(available_gts)
            else:
                q_values = np.full(num_gts, -np.inf)
                for gt in available_gts:
                    s_gt = state_to_index_q(
                    get_state_q(iter_step - last_received_marl[sat][gt]),
                    power_bin,
                    get_demand_bin(gt_demand[gt]),
                    get_link_bin(link_hist[sat][gt])
                    # 0
                )
                    # This is basically the difference between if we were to transmit vs not (1 vs 0 respectively)
                    # q_values[gt] = q_table[s_gt, 1] - q_table[s_gt, 0]
                    
                    # THis chooses best imemdiate reward from the q_table
                    q_values[gt] =  max(q_table[s_gt,0],q_table[s_gt,1])
                    
                best_gts = available_gts[
                    np.isclose(q_values[available_gts], np.max(q_values[available_gts]))
                ]
                action = max(
                    best_gts, key=lambda g: iter_step - last_received_marl[sat][g]
                )  # tie-break by true AoI

            # state of selected GT before the action
            aoi_bin = get_state_q(iter_step - last_received_marl[sat][action])
            packet = packet_queues_marl[sat][action][0]
            # age_bin = get_packet_age_bin(packet, iter_step)
            age_bin = 0
            demand = gt_demand[action]
            
            state_index = state_to_index_q(
                get_state_q(iter_step - last_received_marl[sat][action]),
                power_bin,
                get_demand_bin(demand),
                get_link_bin(link_hist[sat][action])
                # 0
            )
            power_consumed = demand * power_per_mbps
            served_gts[action] = True

            q_len = len(packet_queues_marl[sat][action])
            if q_len > 0 and total_power_sat + power_consumed <= power_limit:
                packet_aoi = packet.get_age(iter_step)
                local_aoi = iter_step - last_received_marl[sat][action]
                freshness_gain = max(0, local_aoi - (iter_step - packet.start_step))

                packets_served_marl += 1
                total_power_sat += power_consumed
                total_throughput_sat += demand
                total_aoi_sat += packet_aoi

                busy_until_marl[sat][ch] = iter_step + packet.service_time
                active_marl[sat][ch] = (action, packet, power_consumed)
                packet_queues_marl[sat][action].pop(0)

                # reward = AOI_FACTOR * freshness_gain - (
                #     (0.01 * power_consumed) * POWER_FACTOR
                # )

                # reward = ( freshness_gain / (demand)) if packet.delivered else (-power_consumed) # this gives much worse aoi but better power consumoption
                reward = freshness_gain / demand
                reward = reward if packet.delivered else 0.0
                
                # gain = freshness_gain if packet.delivered else 0
                # reward = gain / demand

                # reward = freshness_gain - LAMBDA * power_consumed
                aoi_ind_marl.append(packet_aoi)
            else:
                reward = -power_consumed
                
            pend_s[sat, action] = state_index
            pend_a[sat, action] = 1
            pend_r[sat, action] = reward
            pend_t[sat, action] = iter_step
        
        for gt in decision_gts:
            if not served_gts[gt]:
                pend_s[sat, gt] = s_of[gt]
                pend_a[sat, gt] = 0
                pend_r[sat, gt] = 0.0
                pend_t[sat, gt] = iter_step
    
        q_tables[sat] = q_table
        total_throughput_marl += total_throughput_sat
        total_power_marl += total_power_sat
        total_aoi_marl += total_aoi_sat

    throughput_marl[iteration] = total_throughput_marl
    power_marl[iteration] = total_power_marl
    latency_marl[iteration] = num_gts / (total_throughput_marl + 1e-5)
    return total_aoi_marl, packets_served_marl, q_tables


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
    seed=50,
    use_state=True,
    run_dir="",
):
    global rng, mrng
    rng = np.random.default_rng(seed)
    mrng = np.random.default_rng(seed + 1)
    np.random.seed(seed)
    start_time = time.time()

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # variables init
    aoi_marl = np.zeros(num_iterations)
    aoi_greedy = np.zeros(num_iterations)
    aoi_greedy_fr = np.zeros(num_iterations)

    aoi_ind_marl = []
    aoi_ind_greedy = []
    aoi_ind_greedy_fr = []

    weighted_aoi_marl = np.zeros(num_iterations)
    weighted_aoi_greedy = np.zeros(num_iterations)
    weighted_aoi_greedy_fr = np.zeros(num_iterations)

    throughput_marl = np.zeros(num_iterations)
    throughput_greedy = np.zeros(num_iterations)
    throughput_greedy_fr = np.zeros(num_iterations)

    power_marl = np.zeros(num_iterations)
    power_greedy = np.zeros(num_iterations)
    power_greedy_fr = np.zeros(num_iterations)

    latency_marl = np.zeros(num_iterations)
    latency_greedy = np.zeros(num_iterations)
    latency_greedy_fr = np.zeros(num_iterations)

    shape = (num_satellites, num_gts)
    
    pend = {"s": np.full(shape, -1, dtype=int), "a": np.zeros(shape, dtype=int), "r": np.zeros(shape), "t": np.zeros(shape)}

    receiver_aoi_marl = np.zeros(shape)
    receiver_aoi_greedy = np.zeros(shape)
    receiver_aoi_greedy_fr = np.zeros(shape)

    last_received_marl = np.zeros(shape)
    last_received_greedy = np.zeros(shape)
    last_received_greedy_fr = np.zeros(shape)

    link_rng = np.random.default_rng(seed + 2)

    # Initial link states: True = Bad, False = Good
    is_bad = link_rng.random(shape) < LINK_BAD_FRAC

    # Fixed success probabilities for each link in each state
    good_link_p = link_rng.uniform(0.8, 1.0, shape)
    bad_link_p = link_rng.uniform(0.1, 0.4, shape)

    # Current link success probabilities
    link_p = np.where(is_bad, bad_link_p, good_link_p)

    # Noisy link estimates observed by schedulers
    link_hist = np.clip(link_p + link_rng.normal(0, HIST_NOISE, shape), 0, 1)

    # q_tables[sat][gt, ch]
    if q_tables_init is None:
        q_table = np.zeros(num_states)
    else:
        q_table = q_tables_init.copy()
    # Q-learning / bandit parameters
    # epsilon_decay = (epsilon_end / epsilon_start) ** (1 / num_iterations)
    epsilon_decay = 2 * (epsilon_start - epsilon_end) / num_iterations

    def copy_q(q):
        return q.copy() if isinstance(q, np.ndarray) else [x.copy() for x in q]

    if q_tables_init is not None:
        q_table = copy_q(q_tables_init)
    elif use_state:
        q_table = [np.zeros((num_states_q, 2)) for _ in range(num_satellites)]
    else:
        q_table = np.zeros(num_states_q)

    # Generate consistent GT demands
    half = num_gts // 2

    high_demand_gts = np.arange(0, half)
    low_demand_gts = np.arange(half, num_gts)

    high_demand_weight = 2
    low_demand_weight = 1

    weights_AOI = np.where(
        np.arange(num_gts) < half, high_demand_weight, low_demand_weight
    )

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
    packet_queues_greedy = [
        [[] for _ in range(num_gts)] for _ in range(num_satellites)
    ]  # capacity is 1
    packet_queues_greedy_fr = [
        [[] for _ in range(num_gts)] for _ in range(num_satellites)
    ]  # capacity is 1

    active_marl = [[None] * channels_per_satellite for _ in range(num_satellites)]

    active_greedy = [[None] * channels_per_satellite for _ in range(num_satellites)]

    active_greedy_fr = [[None] * channels_per_satellite for _ in range(num_satellites)]

    # Removed this so that the initial queues are empty.
    # Adds initial packets with arrival times -5 to -1
    # for sat in range(num_satellites):
    #     for gt in range(num_gts):
    #         for packet_time in range(-5, 0):
    #             packet_queues_marl[sat][gt].append(AOIPacket(packet_time))
    #             packet_queues_greedy[sat][gt].append(AOIPacket(packet_time))

    # Main sim loop
    epsilon = 0.0 if evaluate else epsilon_start
    rewards = []

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

    previous_aoi = float("inf")
    bad_intervals = 0

    best_q_table = None
    best_aoi = float("inf")

    for iteration in range(num_iterations):
        completed_marl = 0
        completed_greedy = 0
        completed_greedy_fr = 0
        iter_step = iteration + 1
        
        # Markov transition for every satellite-GT link
        transition = link_rng.random(shape)

        # good -> bad with probability p
        became_bad = (~is_bad) & (transition < LINK_TRANSITION_P)

        # bad -> good with probability 1-p
        became_good = is_bad & (transition < 1 - LINK_TRANSITION_P)

        is_bad[became_bad] = True
        is_bad[became_good] = False

        # update success probabilities
        link_p[:] = np.where(is_bad, bad_link_p, good_link_p)

        # update the noisy link estimates
        link_hist[:] = np.clip(link_p + link_rng.normal(0, HIST_NOISE, shape), 0, 1)

        # age all receivers by 1
        receiver_aoi_marl += 1
        receiver_aoi_greedy += 1
        receiver_aoi_greedy_fr += 1

        for sat in range(num_satellites):
            for ch in range(channels_per_satellite):

                if active_marl[sat][ch] is not None:
                    gt, packet, power_consumed = active_marl[sat][ch]

                    if iter_step >= busy_until_marl[sat][ch]:

                        # delivered check
                        if (
                            packet.start_step > last_received_marl[sat][gt]
                            and packet.delivered
                        ):
                            last_received_marl[sat][gt] = packet.start_step
                            receiver_aoi_marl[sat][gt] = (
                                iter_step - packet.start_step
                            )  # time since this packet was generated
                            total_packets_served_marl += 1

                            completed_marl += 1
                        active_marl[sat][ch] = None

                if active_greedy[sat][ch] is not None:
                    gt, packet, power_consumed = active_greedy[sat][ch]

                    if iter_step >= busy_until_greedy[sat][ch]:

                        if (
                            packet.start_step > last_received_greedy[sat][gt]
                            and packet.delivered
                        ):
                            last_received_greedy[sat][gt] = packet.start_step
                            receiver_aoi_greedy[sat][gt] = iter_step - packet.start_step
                            total_packets_served_greedy += 1
                            completed_greedy += 1

                        active_greedy[sat][ch] = None
                if active_greedy_fr[sat][ch] is not None:
                    gt, packet, power_consumed = active_greedy_fr[sat][ch]

                    if iter_step >= busy_until_greedy_fr[sat][ch]:

                        if (
                            packet.start_step > last_received_greedy_fr[sat][gt]
                            and packet.delivered
                        ):
                            last_received_greedy_fr[sat][gt] = packet.start_step
                            receiver_aoi_greedy_fr[sat][gt] = (
                                iter_step - packet.start_step
                            )
                            total_packets_served_greedy_fr += 1
                            completed_greedy_fr += 1

                        active_greedy_fr[sat][ch] = None

        # Add new packets
        for sat in range(num_satellites):
            for gt in range(num_gts):
                rate = rng.uniform(arrival_rate - 0.5, arrival_rate + 0.5)
                num_arrivals = np.random.poisson(rate)

                for _ in range(num_arrivals):
                    packet = AOIPacket(iter_step, service_rate)
                    # link_p is the link quality. Good links stay 0.9 ish and bad ones are at 0.1-0.4 ish. SO higher success for good links
                    packet.delivered = bool(link_rng.random() < link_p[sat, gt])
                    total_packets += 1

                    # below is for M/M/1/1
                    if not packet_queues_marl[sat][gt] and not any(
                        tx is not None and tx[0] == gt for tx in active_marl[sat]
                    ):
                        # if len(packet_queues_marl[sat][gt]) < MAX_QUEUE_LEN:
                        packet_queues_marl[sat][gt].append(packet)
                    else:
                        # comment below 2 lines for M/M/1/1
                        # packet_queues_marl[sat][gt].pop(0)  # drop oldest, keep freshest
                        # packet_queues_marl[sat][gt].append(packet)
                        dropped_marl += 1

                    if not packet_queues_greedy[sat][gt] and not any(
                        tx is not None and tx[0] == gt for tx in active_greedy[sat]
                    ):
                        # if len(packet_queues_greedy[sat][gt]) < MAX_QUEUE_LEN:
                        packet_queues_greedy[sat][gt].append(packet)
                    else:
                        # packet_queues_greedy[sat][gt].pop(
                        #     0
                        # )  # drop oldest, keep freshest
                        # packet_queues_greedy[sat][gt].append(packet)
                        dropped_greedy += 1

                    if not packet_queues_greedy_fr[sat][gt] and not any(
                        tx is not None and tx[0] == gt for tx in active_greedy_fr[sat]
                    ):
                        # if len(packet_queues_greedy_fr[sat][gt]) < MAX_QUEUE_LEN:
                        packet_queues_greedy_fr[sat][gt].append(packet)
                    else:
                        # packet_queues_greedy_fr[sat][gt].pop(
                        #     0
                        # )  # drop oldest, keep freshest
                        # packet_queues_greedy_fr[sat][gt].append(packet)
                        dropped_greedy_fr += 1

        # dynamically change power budgets
        power_budget = base_power_budget + np.random.randint(
            50, 101, size=num_satellites
        )

        total_aoi_greedy, packets_served_greedy = greedy_algorithm(
            iteration,
            iter_step,
            power_budget,
            gt_demand_all,
            packet_queues_greedy,
            busy_until_greedy,
            active_greedy,
            aoi_ind_greedy,
            aoi_one_sat_greedy,
            throughput_greedy,
            power_greedy,
            latency_greedy,
            random_sat,
            num_satellites,
            num_gts,
            channels_per_satellite,
            power_per_mbps,
            receiver_aoi_greedy,
            last_received_greedy,
            link_hist,
        )

        total_aoi_greedy_2, packets_served_greedy_2 = greedy_algorithm_freshness(
            iteration,
            iter_step,
            power_budget,
            gt_demand_all,
            packet_queues_greedy_fr,
            busy_until_greedy_fr,
            active_greedy_fr,
            aoi_ind_greedy_fr,
            aoi_one_sat_greedy_fr,
            throughput_greedy_fr,
            power_greedy_fr,
            latency_greedy_fr,
            random_sat,
            num_satellites,
            num_gts,
            channels_per_satellite,
            power_per_mbps,
            receiver_aoi_greedy_fr,
            last_received_greedy_fr,
            link_hist,
        )

        if use_state:
            total_aoi_marl, packets_served_marl, _ = marl_algorithm_qlearning(
                iteration,
                iter_step,
                power_budget,
                gt_demand_all,
                packet_queues_marl,
                busy_until_marl,
                active_marl,
                q_table,
                aoi_ind_marl,
                aoi_one_sat_marl,
                throughput_marl,
                power_marl,
                latency_marl,
                random_sat,
                epsilon,
                rewards,
                0.0 if evaluate else alpha,
                discount_factor,
                num_satellites,
                num_gts,
                channels_per_satellite,
                power_per_mbps,
                receiver_aoi_marl,
                last_received_marl,
                link_hist,
                pend,
            )
        else:
            total_aoi_marl, packets_served_marl = marl_algorithm(
                iteration,
                iter_step,
                power_budget,
                gt_demand_all,
                packet_queues_marl,
                busy_until_marl,
                active_marl,
                q_table,
                aoi_ind_marl,
                aoi_one_sat_marl,
                throughput_marl,
                power_marl,
                latency_marl,
                random_sat,
                epsilon,
                rewards,
                0.0 if evaluate else alpha,
                num_satellites,
                num_gts,
                channels_per_satellite,
                power_per_mbps,
                receiver_aoi_marl,
                last_received_marl,
                link_hist,
            )

        # Drop packets that could not begin transmission this timestep
        for sat in range(num_satellites):
            for gt in range(num_gts):
                dropped_marl += len(packet_queues_marl[sat][gt])
                dropped_greedy += len(packet_queues_greedy[sat][gt])
                dropped_greedy_fr += len(packet_queues_greedy_fr[sat][gt])

                packet_queues_marl[sat][gt].clear()
                packet_queues_greedy[sat][gt].clear()
                packet_queues_greedy_fr[sat][gt].clear()

        throughput_marl[iteration] = completed_marl
        throughput_greedy[iteration] = completed_greedy
        throughput_greedy_fr[iteration] = completed_greedy_fr
        # Calculate average AoI
        aoi_marl[iteration] = np.mean(receiver_aoi_marl)
        aoi_greedy[iteration] = np.mean(receiver_aoi_greedy)
        aoi_greedy_fr[iteration] = np.mean(receiver_aoi_greedy_fr)

        weighted_aoi_marl[iteration] = np.sum(
            receiver_aoi_marl * weights_AOI[None, :]
        ) / (num_satellites * np.sum(weights_AOI))
        weighted_aoi_greedy[iteration] = np.sum(
            receiver_aoi_greedy * weights_AOI[None, :]
        ) / (num_satellites * np.sum(weights_AOI))
        weighted_aoi_greedy_fr[iteration] = np.sum(
            receiver_aoi_greedy_fr * weights_AOI[None, :]
        ) / (num_satellites * np.sum(weights_AOI))

        aoi_one_sat_marl[iteration] = np.mean(receiver_aoi_marl[random_sat])
        aoi_one_sat_greedy[iteration] = np.mean(receiver_aoi_greedy[random_sat])
        aoi_one_sat_greedy_fr[iteration] = np.mean(receiver_aoi_greedy_fr[random_sat])

        # freezing training if 2 increases in AOI
        # if not evaluate and (iteration + 1) % print_interval == 0:
        #     start = iteration + 1 - print_interval
        #     current_aoi = np.mean(aoi_marl[start:iteration + 1])

        #     if current_aoi < best_aoi:
        #         best_aoi = current_aoi
        #         best_q_table = q_table.copy()
        #         best_q_table = copy_q(q_table)

        #     if current_aoi > previous_aoi:
        #         bad_intervals += 1
        #     else:
        #         bad_intervals = 0

        #     previous_aoi = current_aoi

        #     if bad_intervals >= 4 and alpha != 0 and iteration >= 15000:
        #         epsilon = 0
        #         alpha = 0
        #         print(f"Training frozen at iteration {iteration + 1}, AoI: {current_aoi:.2f}")
        #         q_table = best_q_table.copy()
        #         q_table = copy_q(best_q_table)

        # epsilon decay
        # epsilon *= epsilon_decay
        epsilon = max((epsilon - epsilon_decay),0)
        
        if (iteration + 1) % print_interval == 0:
            start = iteration + 1 - print_interval
            temp = "MARL" if use_state else "Bandit"
            print(
                f"Iteration {iteration + 1}/{num_iterations} "
                f"Epsilon: {epsilon:.6f} "
                f"Power MARL: {np.mean(power_marl[start:iteration+1]):.3f} "
                f"Power Greedy Link Aware: {np.mean(power_greedy_fr[start:iteration+1]):.3f} "
                f"{temp} AoI: {np.mean(aoi_marl[start:iteration+1]):.2f} "
                f"Greedy Link Blind AoI: {np.mean(aoi_greedy[start:iteration+1]):.2f} "
                f"Greedy Link Aware AoI: {np.mean(aoi_greedy_fr[start:iteration+1]):.2f} "
                f"AoI mean/max-GT: {receiver_aoi_marl.mean():.1f}/{receiver_aoi_marl.max(axis=1).mean():.1f} "
            )
    if use_state and not evaluate:
        for sat in range(num_satellites):
            for gt in range(num_gts):
                ps = pend["s"][sat, gt]
                if ps < 0:
                    continue

                pa = pend["a"][sat, gt]

                if pa == 1 and any(tx is not None and tx[0] == gt for tx in active_marl[sat]):
                    continue

                reward = pend["r"][sat, gt]
                q_table[sat][ps, pa] += alpha * (reward - q_table[sat][ps, pa])

    # window = 10000

    # smoothed_rewards = np.convolve(rewards,np.ones(window) / window,mode="valid")

    # plt.figure()
    # plt.plot(smoothed_rewards)
    # plt.xlabel("Decision")
    # plt.ylabel("Average Reward")
    # plt.title("MARL Reward Moving Average")
    # plt.grid(True)
    # plt.show()
    suffix = ""

    # suffix = f"train-{seed}" if not evaluate else f"eval-{seed}"
    suffix = f"{'qlearn' if use_state else 'bandit'}-{'eval-' + str(seed) if evaluate else 'train'}"

    # Plots

    iterations = np.arange(1, num_iterations + 1)

    # AAoI
    plt.figure()

    eval_model = "MARL " if use_state else "Bandit "
    greedy_1 = "Greedy Link Blind "
    greedy_2 = "Greedy Link Aware "
    plt.plot(iterations, aoi_marl, linewidth=2, label=eval_model + "AAoI")
    plt.plot(iterations, aoi_greedy, linewidth=2, label=greedy_1 + "AAoI")
    plt.plot(iterations, aoi_greedy_fr, linewidth=2, label=greedy_2 + "AAoI")

    plt.xlabel("Iterations")
    plt.ylabel("Average AOI")
    plt.title(f"AoI Comparison: {eval_model} vs Greedy")

    plt.legend()
    plt.grid(True)
    plt.savefig(run_dir / f"AAOI-{suffix}.png", dpi=300, bbox_inches="tight")
    # plt.show()

    # Weighted sum aoi
    plt.figure()

    plt.plot(iterations, weighted_aoi_marl, linewidth=2, label=eval_model + "WS AoI")
    plt.plot(iterations, weighted_aoi_greedy, linewidth=2,  label=greedy_1 +  "WS AoI")
    plt.plot(iterations, weighted_aoi_greedy_fr, linewidth=2,  label=greedy_2 + "WS AoI")

    plt.xlabel("Iterations")
    plt.ylabel("Weighted Sum AOI")
    plt.title(f"WS AoI Comparison: {eval_model} vs Greedy")

    plt.legend()
    plt.grid(True)
    plt.savefig(run_dir / f"WS_AOI-{suffix}.png", dpi=300, bbox_inches="tight")
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

    plt.plot(iterations, power_marl, linewidth=2, label=eval_model + "Usage")
    plt.plot(iterations, power_greedy, linewidth=2,  label=greedy_1 +  "Power Usage")
    plt.plot(iterations, power_greedy_fr, linewidth=2,  label=greedy_2 + "Power Usage")

    plt.xlabel("Iterations")
    plt.ylabel("Total Power Usage (Watts)")
    plt.title(f"Power Usage Comparison: {eval_model} vs Greedy")

    plt.legend()
    plt.grid(True)
    plt.savefig(run_dir / f"Power-{suffix}.png", dpi=300, bbox_inches="tight")

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

    # Throughput
    plt.figure()
    plt.plot(iterations, throughput_marl, linewidth=2, label=eval_model + "Throughput")
    plt.plot(iterations, throughput_greedy, linewidth=2,  label=greedy_1 +  "Throughput")
    plt.plot(iterations, throughput_greedy_fr, linewidth=2,  label=greedy_2 + "Throughput")
    
    plt.xlabel("Iterations")
    plt.ylabel("Packets delivered per timestep")
    plt.title(f"Throughput Comparison: {eval_model} vs Greedy")

    plt.legend()
    plt.grid(True)
    plt.savefig(run_dir / f"Throughput-{suffix}.png", dpi=300, bbox_inches="tight")

    # One sat AOI
    window = 200

    marl_smooth = np.convolve(aoi_one_sat_marl, np.ones(window) / window, mode="valid")

    greedy_smooth = np.convolve(
        aoi_one_sat_greedy, np.ones(window) / window, mode="valid"
    )

    greedy_smooth_fr = np.convolve(
        aoi_one_sat_greedy_fr, np.ones(window) / window, mode="valid"
    )

    plt.figure(figsize=(12, 5))
    plt.plot(marl_smooth, label=eval_model)
    plt.plot(greedy_smooth, label=greedy_1)
    plt.plot(greedy_smooth_fr, label=greedy_2)

    plt.xlabel("Iterations")
    plt.ylabel("Average Packet AoI")
    plt.title(f"Satellite {random_sat}: AoI Moving Average")
    plt.legend()
    plt.grid(True)
    plt.savefig(run_dir / f"one_sat-{suffix}.png", dpi=300, bbox_inches="tight")

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
    print(f"Packet loss {eval_model}:", loss_marl)
    
    eval_model = "MARL " if use_state else "Bandit"
    print(eval_model + "Average Receiver AoI:", np.mean(aoi_marl))
    print("Greedy Link Blind Average Receiver AoI:", np.mean(aoi_greedy))
    print("Greedy Link Aware Average Receiver AoI:", np.mean(aoi_greedy_fr))

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
        "q_table": q_table,
        "weighted_aoi_marl": weighted_aoi_marl,
        "weighted_aoi_greedy": weighted_aoi_greedy,
        "weighted_aoi_greedy_fr": weighted_aoi_greedy_fr,
    }


USE_STATE = False
if __name__ == "__main__":

    run_dir = Path("plots") / time.strftime("run_%Y%m%d-%H%M%S")
    for use_state in [True, False]:
        train = run_simulation(
            num_iterations=200_000,
            seed=50,
            num_satellites=30,
            num_gts=100,
            use_state=use_state,
            run_dir=run_dir,
            discount_factor=0.95 ,
        )

        # for seed in [20, 78]:
        #     test = run_simulation(
        #         num_iterations=5000,
        #         q_tables_init=train["q_table"],
        #         evaluate=True,
        #         seed=seed,
        #         num_satellites=30,
        #         num_gts=100,
        #         use_state=use_state
        #     )

        if use_state:
            np.save(run_dir / "q_tables_qlearn.npy", np.array(train["q_table"]))
        else:
            np.savetxt(run_dir / "q_table_bandit.txt", train["q_table"])
# Average only 1 satellite
# keep a single agent baseline
# also change around how much state each satellite sees, also local observation
# SARL , MARL incomplete state, MARL full state, greedy
# Try and extend to other types of queues
# NS tree for satellite comm
