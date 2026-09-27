# Benchmark rr_f59ac95_test

Jobs: 4; scored: 4. Errors (no score): 0.

## By scenario (median / p90)

| scenario | runs | no score | runs with failed checks | crashed | vel_rmse | vel_rmse_clean_ref | focus_rmse | final_drift_xyz_pct | xy_rmse | xy_rmse_rtk | xy_med | xyz_rmse | lat_p95_ms | lat_max_ms | cpu_p95 | rss_max_mib |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 4 | 0 | 0 | 0 | 0.044 / 0.166 | 0.044 / 0.056 | — / — | 0.036 / 0.107 | 5.587 / 9.595 | 2.393 / 3.142 | 0.988 / 1.323 | 5.962 / 9.974 | 1.207 / 1.409 | 51.242 / 58.767 | 0.100 / 0.140 | 105.447 / 105.637 |

## Jobs

| job | vel_rmse | vel_rmse_post_init | vel_rmse_clean_ref | vel_bias | vel_matched | focus_rmse | final_drift_xyz_pct | xy_rmse | xy_rmse_rtk | xy_med | xyz_rmse | rate_vel_hz | rate_pos_hz | lat_p95_ms | lat_max_ms | cpu_p95 | rss_max_mib | failed_checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30618_0686195f__clean__s0 | 0.046 | 0.046 | 0.046 | -0.004 | 0.992 | — | 0.008 | 7.755 | — | 0.868 | 8.059 | 29.029 | 29.028 | 1.024 | 50.920 | 0.100 | 105.387 | 0 |
| 30618_27e994fc__clean__s0 | 0.166 | 0.166 | 0.056 | -0.003 | 0.980 | — | 0.107 | 3.420 | 3.142 | 1.323 | 3.866 | 29.486 | 29.485 | 1.409 | 51.564 | 0.140 | 105.508 | 0 |
| 30618_88548b02__clean__s0 | 0.036 | 0.037 | 0.037 | -0.004 | 0.991 | — | 0.026 | 1.537 | 1.644 | 0.936 | 1.937 | 29.293 | 29.292 | 1.389 | 58.767 | 0.100 | 105.367 | 0 |
| 30618_defd0170__clean__s0 | 0.041 | 0.041 | 0.041 | -0.008 | 0.992 | — | 0.046 | 9.595 | — | 1.040 | 9.974 | 29.222 | 29.220 | 0.920 | 35.440 | 0.100 | 105.637 | 0 |
