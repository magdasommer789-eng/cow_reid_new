# Cow Re-Identification — Final Results

## Test Protocol
- **Query**: first 10-second clip per test cow
- **Gallery**: all remaining clips from all test cows (non-overlapping with query)
- **Test cows**: 10  |  **Train cows**: 15  |  **Val cows**: 6
- **Transfer learning**: ImageNet-pretrained backbones

## Results

| Model   | mAP    | Rank-1   | Rank-5   | Rank-10   |
|:--------|:-------|:---------|:---------|:----------|
| C3D     | 48.47% | 30.00%   | 70.00%   | 100.00%   |
| X3D     | 66.83% | 50.00%   | 90.00%   | 100.00%   |
| SWIN    | 65.33% | 50.00%   | 100.00%  | 100.00%   |
| VIVIT   | 71.83% | 60.00%   | 90.00%   | 100.00%   |

*mAP = mean Average Precision; Rank-k = CMC@k (fraction of queries with correct ID in top-k)*
