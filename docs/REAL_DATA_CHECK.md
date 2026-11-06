# Real-data check for turnscope

Goal: find out if turnscope is really better than the simple threshold on real two-channel calls.

1. **Get data.** Open dual-track conversations with timestamped transcripts: MagicHub full-duplex set (arXiv 2509.04093, CC BY 4.0; 27 Chinese + 8 English conversations, dual-track 16 kHz WAV). Put the WAV files in `recordings/`. Check each file has one speaker per channel.
2. **Make `labels.csv`** from the timestamps, one row per speech segment:

       file,channel,start_s,end_s
       conv01.wav,0,1.25,3.80
       conv01.wav,1,4.10,6.00

3. **Run:**

       pip install -e .[dev]
       python scripts/real_data_check.py --audio-dir recordings/ --labels labels.csv --out real_check

4. **Read `real_check/results.md`.** It compares turnscope and the fixed threshold on speech-detection F1 and on response-delay error (turns found in both labels and audio).
5. **Decide.**
   - turnscope clearly better on both F1 and delay error: good, send me the table.
   - About equal or worse: send the table, I make one fix round (two at most).
   - Still not clearly better after round two: stop. Do not publish.

Do not change thresholds and re-test on the same files without telling me; that gives false good numbers. Keep a few conversations unseen for the final check.
