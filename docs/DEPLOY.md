# Deploying ForestEar to the cloud

The app is one Docker image (API + web UI), so any container host works. Two tested-in-principle routes:

## A. Render (free tier, public URL in ~10 minutes)
1. Push this repo to GitHub.
2. Render dashboard -> **New +** -> **Web Service** -> connect the repo.
3. Runtime: **Docker** (Render detects the `Dockerfile`). Instance type: the app uses about **730 MB of RAM** once the model is loaded (measured), so Render's *Free* and *Starter* instances (512 MB) will run out of memory. Choose an instance with **2 GB or more** (e.g. *Standard*), and switch it off or delete it after grading to stop the charges.
4. Environment variables: `PORT=8000`, `AUTO_RETRAIN_MIN_CLIPS=20`.
5. Add a **Disk** mounted at `/app/data` if you want uploads and the SQLite database to survive restarts (paid plans), and set `MODELS_DIR=/app/data/models` so retrained models survive too.
6. Deploy. Put the resulting `https://<name>.onrender.com` URL in the README.

## B. Any VM (Azure, AWS EC2, GCP, DigitalOcean): the one used for the scaling experiment
```bash
# on a fresh Ubuntu VM (2+ vCPU, 4+ GB RAM)
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
git clone <your-repo-url> forest_ear && cd forest_ear
sudo docker compose up -d --build --scale api=2      # nginx on port 80 -> 8080 mapping, see below
```
Open port 8080 in the VM's firewall / security group and browse to `http://<vm-ip>:8080`.
To serve on port 80, change `"8080:80"` to `"80:80"` in `docker-compose.yml`.

### Scaling and the Locust test
```bash
sudo docker compose up -d --scale api=1     # then 2, 3, 4
locust -f locustfile.py --host http://<vm-ip>:8080 --headless -u 50 -r 10 -t 60s --csv results/containers_1
```
or run everything in one go with `scripts/run_load_tests.sh`, which produces `results/summary.md` and `results/load_test.png`.
Run Locust from a *different* machine (your laptop) so it does not steal CPU from the containers it is measuring.

## Notes
* All replicas share one volume (`/shared`) holding the model registry, uploads and the SQLite database. A replica that finishes retraining writes the new model there and every other replica hot-reloads it on its next request.
* SQLite is plenty for a demo. For a production fleet swap `src/db.py` for PostgreSQL.
