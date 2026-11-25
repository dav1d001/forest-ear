# Demo video script (about 4 minutes, camera ON)

Keep your face visible in a corner (Zoom / Meet / Loom camera bubble). Speak while you click. Rubric items in **bold**.

| Time | Show | Say |
|---|---|---|
| 0:00 | Camera on, README top | Who you are, the problem (illegal logging detection), what the system does. |
| 0:30 | Notebook, scroll quickly: preprocessing, hyper-parameter table, metrics table, confusion matrix | **Evaluation:** accuracy, precision, recall, F1, ROC-AUC; baseline vs optimised; the optimisation techniques used. |
| 1:15 | UI **Listen** tab | **Prediction:** click the *chainsaw* sample; point at the spectrogram, the orange alert and the confidence. Then drop your own file and show a second correct prediction. Show one clip with low confidence and explain the warning. |
| 2:00 | **Insights** tab | Read one of the three stories aloud (e.g. brightness / duration / texture). |
| 2:20 | **Teach & retrain** tab | **Retraining:** (1) choose class `chirping_birds`, drop the 40 files from `demo_uploads/chirping_birds` and press *Upload* (saved to database). (2) Say that preprocessing happens when training starts. (3) Press *Retrain now* and narrate the stages. Show the result: new version, promoted, new class available. Go back to **Listen** and predict a chirping-birds clip. |
| 3:20 | **Monitor** tab | **Up-time**, request count, latency, confidence gauge, model versions, confusion matrix. |
| 3:40 | Terminal / Locust UI | Locust flood against 1 vs 3 containers; show the results table in the README. |
| 4:00 | Close | Where the code is, how to run it (`docker compose up`). |

Upload the video to YouTube (unlisted is fine) and paste the link at the top of the README.
