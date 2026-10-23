### Design choices

- Greedy sorts all GTs by aoi of the packet in their packet queues, and then chooses teh one with the highest one. Taken from survey paper
- MARL chooses which GT to actually serve for each satellite
- Changed aoi calculations, old one was only checking packet aoi when it was served. Now i calculate aoi for EACH satellite, increasing the aoi for each satellite by 1 each iteration. 

- Should now be a M/M/1/k if we consider each GT as a server, since each queue has capacity k.
 
- We also prevent transmission if another satelite is already transmitting to this gt. So two satellites cannot serve the same GT at the same time. This is to make the server queue have capacity 1, when we consider each GT as a server. 

### Packet Queues

- Each sat,gt pair has a queue. Each packet in this queue represents that the packet is awaiting transmission from this sat to this gt.
- `Change : Instead, only each gt will have a queue and each packet would represent that that packet is to be sent there irrepsective of which sat its in. Also change it from FIFo to LIFO and remove each packet and all that`
- If a gt is waiting for transmission from one satellite, but that specific satellite has multiple packets to be sent, only send the most recent one. This design assumes that each satellite sends data from only one sensor. 
- So each sat,gt queue should have capacity one to make this happen.


### Constants' Sources

- \# GT's -> MARL survey paper
- \# Comm channels per satellite
- Discount factor, epsilon
-  Max queue length 
- Task arrival rate
- 
#### Connectivity 
- \# neighboring satellites
- Update interval 
- Loop penalty, positive reward
- Replay buffer size, batch size


### Problems / Clarifications 

- To make this an M/M/1 quque, the service time would also have to be a Poisson process. The arrival times are Poisson distributed and the capacity of each service quque is 1. 
- `But, there is currently no service times in place, since once a packet is chosen to be served it gets transmitted immediately. Theere is no transmission time or processing time for the satellite in sending the packet. The only closest thing would be waiting time, so should we rather calculate system time since that includes waiting time and service time. `


- There is also now a training and evaluation. In evaluation the epsilon and q_table updates are frozen.
    - This is because alpha is set to 0 is evaluate is false
    - And epsilon is also set to 0, so it always chooses the max_q


m -> # sats
n-> # gts 
if m < n 60 100
m = n   60 60
m > n : What we have right now. 60 20
Only changing the no sats
Try and introduce heterogeniety within service times and arrival rates 
Maybe try and add a z axis for the satellties during routing if possible? Maybe try and add it to the satellites randomness 

Take avg over one sat vs global 
    
talk with omid about power in reward func 

MM12 


Add some constants for the rward
penalty can be -inf
case where n packets in queue
- state could have some aggr statistic about the queue 

Consider that sats are not disjoint. Multiple sensors are connected to one and only one satellite. 

could be a masters thsis, but not for now 
if solar storm
- no packets can be sent
- consider how much disruption is caused
- look at tail expectiations as well
instead of max expected reward value , max CVAR of reward. considers worse case as well
