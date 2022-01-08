### Standard

| Max epochs | LR       | Stopped Early | Best R@5 Val | Best R@5 Test | Link                            |
| ---------- | -------- | ------------- | ------------ | ------------- | ------------------------------- |
| 10         | 0.000001 | N (10)        | 79.6         | 77.9          | [example](Runs/std_10_0.000001) |
| 10         | 0.00001  | Y (8)         | 81.0         | 80.6          | [example](Runs/std_10_0.00001)  |
| 10         | 0.0001   | Y (6)         | 79.6         | 78.3          | [example](Runs/std_10_0.0001)   |
| 10         | 0.001    | Y (8)         | 76.9         | 77.8          | [example](Runs/std_10_0.001)    |
| 10         | 0.01     | Y (7)         | 71.6         | 72.3          | [example](Runs/std_10_0.01)     |
| 10         | 0.1      | Y (10)        | 71.7         | 70.2          | [example](Runs/std_10_0.1)      |

### NetVLAD

| Max epochs | Num Clusters | LR              | Stopped Early | Best R@5 Val | Best R@5 Test | Link |
| ---------- | ------------ | --------------- | ------------- | ------------ | ------------- | ---- |
| 10         | 64           | 0.000001        |               |              |               |      |
| 10         | 64           | 0.00001         |               |              |               |      |
| 10         | 64           | 0.0001          |               |              |               |      |
| 10         | 16           | best NetVLAD LR |               |              |               |      |
| 10         | 32           | best NetVLAD LR |               |              |               |      |
| 10         | 128          | best NetVLAD LR |               |              |               |      |

LRs to test are the best three from the standard nn

### Assignments

FRA: [std_15_0.000001, netvlad_10_0.000001_64]

DESI: [netvlad_10_0.00001_64]

MARCO: [netvlad_10_0.0001_64]
