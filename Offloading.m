% Parameters
num_agents = 60; % Number of agents (LEO satellites)
episodes = 1000; % Number of training episodes
max_steps = 50; % Steps per episode
state_size = 11; % State space size (queue lengths from 0 to 10)
action_size = 3; % Actions: 1=Process locally, 2=Offload to neighbor, 3=Offload to ground
gamma = 0.99; % Discount factor
alpha = 0.1; % Learning rate
lambda = 0.9; %Factor for proc time in reward function 
epsilon = 1.0; % Initial epsilon for epsilon-greedy policy
epsilon_min = 0.01;
epsilon_decay = 0.995;

% Environment Parameters
parameters.task_arrival_rate = 0.5; % Probability of new task arrival
parameters.max_queue_length = 10;
parameters.local_processing_time = 1;
parameters.local_energy_consumption = 2;
parameters.offload_processing_time = 1;
parameters.offload_energy_consumption = 1;
parameters.ground_processing_time = 1;
parameters.ground_energy_consumption = 0.5;
parameters.offload_delay = 2; % Time delay when offloading to neighbor
parameters.ground_delay = 3; % Time delay when offloading to ground
parameters.energy_weight = 0.1; % Weight of energy consumption in reward
parameters.communication_cost = -0.5; % Communication cost for CDRL

% Initialize Q-tables for MARL and CDRL
Q_marl = zeros(state_size, action_size, num_agents);
Q_cdrl = zeros(state_size, action_size);

% Train MARL and CDRL frameworks
rewards_marl = train_marl(Q_marl, parameters, episodes, max_steps, gamma, alpha, epsilon, epsilon_min, epsilon_decay, num_agents,lambda);
rewards_cdrl = train_cdrl(Q_cdrl, parameters, episodes, max_steps, gamma, alpha, epsilon, epsilon_min, epsilon_decay, num_agents,lambda);

% Plot the results
figure;
plot(1:episodes, rewards_marl, 'LineWidth', 1.5);
hold on;
plot(1:episodes, rewards_cdrl, 'LineWidth', 1.5);
xlabel('Episodes');
ylabel('Total Reward');
title('Comparison of MARL vs CDRL in LEO Satellite Offloading');
legend('MARL', 'CDRL');
grid on;

% LEO Environment function
function [next_state, reward,packet_queue] = leo_environment(packet_queue, action, step,parameters,lambda)
    % Simulate task arrival
    if rand < parameters.task_arrival_rate
        if length(packet_queue) < parameters.max_queue_length
            packet_queue{ end+1 } = AOIPacket(step);
        end
    end
    % Process Packet and reward
    if ~isempty(packet_queue)
        packet = packet_queue{1};
        processing_time = 0;
      
        if action == 1
            %Local Processing 
            processing_time = parameters.local_processing_time;
        
        elseif action == 2
            %Offload to satellite Processing 
            processing_time = parameters.offload_processing_time;
        elseif action == 3
            %Ground Station time
            processing_time = parameters.ground_processing_time;
        else
            error('Invalid Action');
        end

        aoi = packet.get_age(step);
        reward = -(aoi + lambda * processing_time);

        packet_queue(1) = [];

    else
        reward =0;
        end 
  
    next_state = length(packet_queue);
end

% MARL training function
function rewards = train_marl(Q_marl, parameters, episodes, max_steps, gamma, alpha, epsilon, epsilon_min, epsilon_decay, num_agents,lambda)
    rewards = zeros(1, episodes);
    for episode = 1:episodes
        % Initialize all packets in the queue
        packet_queue = cell(1,num_agents);
        for agent = 1:num_agents
            packet_queue{agent} = {};
        end 

        total_reward = 0;
        % Initialize state for each agent
        states = zeros(1, num_agents); % Queue lengths, start at 0
        for step = 1:max_steps
            actions = zeros(1, num_agents);
            for agent = 1:num_agents
                state = states(agent) + 1; % Add 1 for 1-based indexing
                % Epsilon-greedy action selection
                if rand < epsilon
                    action = randi([1, 3]);
                else
                    [~, action] = max(Q_marl(state, :, agent));
                end
                actions(agent) = action;
            end
            % Simulate environment and update Q-tables
            rewards_step = zeros(1, num_agents);
            next_states = zeros(1, num_agents);
            for agent = 1:num_agents
                state = length(packet_queue{agent});
                action = actions(agent);
                [next_state, reward,packet_queue{agent}] = leo_environment(packet_queue{agent}, action,step,parameters,lambda);
                rewards_step(agent) = reward;
                next_state_idx = next_state + 1;
                state_idx = state + 1;
                % Update Q-table
                Q_marl(state_idx, action, agent) = Q_marl(state_idx, action, agent) + alpha * ...
                    (reward + gamma * max(Q_marl(next_state_idx, :, agent)) - Q_marl(state_idx, action, agent));
                next_states(agent) = next_state;
            end
            total_reward = total_reward + sum(rewards_step);
            states = next_states;
        end
        rewards(episode) = total_reward;
        % Decay epsilon
        if epsilon > epsilon_min
            epsilon = epsilon * epsilon_decay;
        end
    end
end

% CDRL training function
function rewards = train_cdrl(Q_cdrl, parameters, episodes, max_steps, gamma, alpha, epsilon, epsilon_min, epsilon_decay, num_agents,lambda)
    rewards = zeros(1, episodes);
    for episode = 1:episodes
        packet_queue = cell(1,num_agents);
        for agent = 1:num_agents
            packet_queue{agent} = {};
        end
        

        total_reward = 0;
        % Initialize state for each agent
        states = zeros(1, num_agents); % Queue lengths, start at 0
        for step = 1:max_steps
            actions = zeros(1, num_agents);
            for agent = 1:num_agents
                state = states(agent) + 1; % Add 1 for 1-based indexing
                % Epsilon-greedy action selection
                if rand < epsilon
                    action = randi([1, 3]);
                else
                    [~, action] = max(Q_cdrl(state, :));
                end
                actions(agent) = action;
            end
            % Simulate environment and update Q-table
            rewards_step = zeros(1, num_agents);
            next_states = zeros(1, num_agents);
            for agent = 1:num_agents
                state = length(packet_queue{agent});
                action = actions(agent);
                [next_state, reward,packet_queue{agent}] = leo_environment(packet_queue{agent}, action, step, parameters,lambda);
                % Add communication cost for centralized control
                reward = reward + parameters.communication_cost;
                rewards_step(agent) = reward;
                next_state_idx = next_state + 1;
                state_idx = state + 1;
                % Update centralized Q-table
                Q_cdrl(state_idx, action) = Q_cdrl(state_idx, action) + alpha * ...
                    (reward + gamma * max(Q_cdrl(next_state_idx, :)) - Q_cdrl(state_idx, action));
                next_states(agent) = next_state;
            end
            total_reward = total_reward + sum(rewards_step);
            states = next_states;
        end
        rewards(episode) = total_reward;
        % Decay epsilon
        if epsilon > epsilon_min
            epsilon = epsilon * epsilon_decay;
        end
    end
end
