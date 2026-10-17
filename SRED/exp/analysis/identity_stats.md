# Identity / sample distribution

P-dev ∩ P-hold identities: 0 (must be 0). Train and public identity namespaces are independent (folder IDs are reused for different people).

| metric | train raw folders | train cleaned identities | public raw folders | public (full) | P-dev (cleaned) | P-hold (cleaned) |
|---|---|---|---|---|---|---|
| identities | 1,900 | 1,823 | 902 | 868 | 424 | 439 |
| files | 58,404 | 57,010 | 17,850 | 17,850 | 5,974 | 7,421 |
| hours | 145.91 | 143.74 | 37.32 | 37.32 | 11.71 | 15.50 |
| files_per_id_min | 1 | 1 | 1 | 1 | 1 | 1 |
| files_per_id_p25 | 6 | 6 | 5 | 5 | 3 | 3 |
| files_per_id_median | 11 | 11 | 12 | 12 | 9 | 11 |
| files_per_id_mean | 30.74 | 31.27 | 19.79 | 20.56 | 14.09 | 16.90 |
| files_per_id_p75 | 41 | 41 | 24 | 25 | 18 | 21 |
| files_per_id_p95 | 91.05 | 92 | 54 | 54.65 | 35.85 | 41.10 |
| files_per_id_max | 577 | 894 | 603 | 603 | 510 | 548 |
| ids_with_1_file | 14 | 16 | 73 | 71 | 61 | 42 |
| ids_with_le3 | 122 | 143 | 164 | 151 | 138 | 114 |
| min_per_id_median | 1.43 | 1.39 | 1.50 | 1.56 | 1.11 | 1.39 |
| top1_id_share_files | 1.0% | 1.6% | 3.4% | 3.4% | 8.5% | 7.4% |
| top10_ids_share_files | 7.4% | 8.9% | 14.2% | 15.0% | 23.9% | 22.3% |
| file_dur_median | 7.36 | 7.46 | 5.37 | 5.37 | 4.98 | 5.33 |
| trials | — | — | 49,993 | 49,993 | 13,489 | 22,776 |
| targets | — | — | 25,000 | 25,000 | 8,070 | 15,955 |
| nontargets | — | — | 24,993 | 24,993 | 5,419 | 6,821 |
| ids_with_target_trials | — | — | 387 | 387 | 185 | 201 |
| top1_id_share_targets | — | — | 29.5% | 29.5% | 4.3% | 45.1% |

## Identities by number of files

| files per identity | train raw folders | train cleaned identities | public raw folders | public (full) | P-dev (cleaned) | P-hold (cleaned) |
|---|---|---|---|---|---|---|
| 1 | 14 | 16 | 73 | 71 | 61 | 42 |
| 2–3 | 108 | 127 | 91 | 80 | 77 | 72 |
| 4–5 | 251 | 241 | 109 | 100 | 43 | 53 |
| 6–10 | 534 | 509 | 143 | 139 | 45 | 49 |
| 11–20 | 271 | 230 | 212 | 206 | 128 | 104 |
| 21–50 | 373 | 361 | 214 | 210 | 59 | 103 |
| 51–100 | 267 | 258 | 50 | 49 | 6 | 11 |
| >100 | 82 | 81 | 10 | 13 | 5 | 5 |
