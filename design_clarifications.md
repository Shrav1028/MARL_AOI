### Design choices

- Greedy sorts all GTs by aoi of the packet in their packet queues, and then chooses teh one with the highest one. Taken from survey paper
- MARL chooses which GT to actually serve for each satellite
- Changed aoi calculations, old one was only checking packet aoi when it was served. Now i calculate aoi for EACH satellite, increasing the aoi for each satellite by 1 each iteration. 

- Should now be a M/M/1/k if we consider each GT as a server, since each queue has capacity k.
 
- We also prevent transmission if another satelite is already transmitting to this gt. So two satellites cannot serve the same GT at the same time. This is to make the server queue have capacity 1, when we consider each GT as a server. 
- There is also another greedy algorithm baseline. This one chooses maximum freshness gain
    - `Freshness gain = last received packet timestamp for the given SAT,GT pair - (currrent timestamp - packet generation time)`
    - This is how much the AOI of the receiver would go down by if we were to transmit this packet 

- MALR seems to perform best when there is heavy contention, like when \#satellites << \#GTs
<!-- ### Packet Queues

- Each sat,gt pair has a queue. Each packet in this queue represents that the packet is awaiting transmission from this sat to this gt.
- `Change : Instead, only each gt will have a queue and each packet would represent that that packet is to be sent there irrepsective of which sat its in. Also change it from FIFo to LIFO and remove each packet and all that`
- If a gt is waiting for transmission from one satellite, but that specific satellite has multiple packets to be sent, only send the most recent one. This design assumes that each satellite sends data from only one sensor. 
- So each sat,gt queue should have capacity one to make this happen.
 -->

### Constants' Sources

- \# GT's -> MARL survey paper
- \# Comm channels per satellite
- Discount factor, epsilon
-  Max queue length 
- Task arrival rate
- 
<!-- #### Connectivity 
- \# neighboring satellites
- Update interval 
- Loop penalty, positive reward
- Replay buffer size, batch size -->


### Problems / Clarifications 

- There is also now a training and evaluation. In evaluation the epsilon and q_table updates are frozen.
    - This is because alpha is set to 0 is evaluate is false
    - And epsilon is also set to 0, so it always chooses the max_q

- `If we were to deploy freshness gain as a real metric for the greedy and the MARL model, then we'd need some sort of ACK mechanism to know when the last delivered packet was from that sensor to that GT. `
    - `An alternative could be to log when the packet was sent from the given satellite and use that instead of last received assuming (VERY optimistically) that the transmission was successful.`
    - WHat other metrics would be restricted if this was to be considered?

- The comparison plots donot currently have the frshness based greedy baseline, I will make sure to update it when the simulation is done running. 
- Currently changing around `AOI_FACTOR` and `POWER_FACTOR` to see whats the best?
- An issue that I'm facing is that the MARL agent is overfitting (I think?), the AAOI goes down to a minimum if 4.44 but then rises again to over the greedy baseline before eventually oscillating. 
- If both are set to 0.5, with the power normalizing factor also set to 0.01, the AOI is minimized but the power consumption is really high. 
