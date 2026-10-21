### Design choices

- Greedy sorts all GTs by aoi of the packet in their packet queues, and then chooses teh one with the highest one. Taken from survey paper
- MARL chooses which GT to actually serve for each satellite


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

