# turnscope: real-data check

6 labelled conversations.

| method | speech F1 | precision | recall | labelled hand-overs | not found in audio | median abs error (s) | p90 abs error (s) |
|---|---|---|---|---|---|---|---|
| turnscope | 0.897 | 0.944 | 0.856 | 449 | 126 | 0.277 | 0.440 |
| simple threshold | 0.700 | 0.965 | 0.600 | 449 | 207 | 0.410 | 1.339 |

How to read it: labelled speech times are usually coarse (they include short pauses), so F1 will not be near 1.0 for either method; compare the two rows. The response-delay error is the number that matters. turnscope is only worth keeping if it is clearly better than the simple threshold here (lower error AND fewer hand-overs missed).
