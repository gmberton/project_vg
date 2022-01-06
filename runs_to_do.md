### Standard

| Max epochs | LR      | Stopped Early | Link                                 |
| ---------- | ------- | ------------- | ------------------------------------ |
| 10         | 0.00001 |               | [example](Runs/netvlad_5_0.00001_64) |
| 10         | 0.0001  |               |                                      |
| 10         | 0.001   |               |                                      |
| 10         | 0.01    |               |                                      |
| 10         | 0.1     |               |                                      |

### NetVLAD

| Max epochs | Num Clusters | LR              | Stopped Early | Link |
| ---------- | ------------ | --------------- | ------------- | ---- |
| 10         | 64           | 0.00001         |               |      |
| 10         | 64           | 0.0001          |               |      |
| 10         | 64           | 0.001           |               |      |
| 10         | 8            | best NetVLAD LR |               |      |
| 10         | 16           | best NetVLAD LR |               |      |
| 10         | 32           | best NetVLAD LR |               |      |
| 10         | 128          | best NetVLAD LR |               |      |

LRs to test are the best three from the standard nn

### Assignments

FRA: [std_10_0.00001, std_10_0.0001]

DESI: [std_10_0.001, std_10_0.01]

MARCO: [std_10_0.1, :(]
