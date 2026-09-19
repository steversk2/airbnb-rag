# Airbnb RAG Concierge

A retrieval-augmented generation concierge for an Airbnb cabin. Guests get a
check-in link and ask questions ("what's the wifi password?", "where's good
ice cream nearby?"); answers come from the cabin guide plus researched local
business info, with sources cited and guardrails against off-topic questions.

## Architecture

```
Guest ──HTTPS──> Traefik ──> concierge-api (FastAPI)
                                   ├─ embed question (BGE, 384-dim)
                                   ├─ Qdrant top-4 chunk retrieval
                                   └─ vLLM (Qwen2.5-7B-Instruct-AWQ) answers
                                        from retrieved context only
```

All on **one AWS `g4dn.xlarge`** (Tesla T4, 16 GB) running **k3s**.
Budget: **under $50/month** — the GPU instance is stopped between sessions
(only ~$8/mo EBS ticks over while parked).

## 1. Infrastructure

- EC2 `g4dn.xlarge`, AWS Deep Learning Base AMI (Ubuntu 24.04), 100 GB gp3.
- Install k3s: `curl -sfL https://get.k3s.io | sh -`
- NVIDIA device plugin — **pin v0.17.1**. v0.20.0 fails on this AMI with
  `NVML: ERROR_LIBRARY_NOT_FOUND`:
  ```
  kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml
  ```
  Verify: `kubectl get nodes -o json | grep nvidia.com/gpu` should show `1`.
- Allocate an **Elastic IP** and associate it with the instance (survives
  stop/start; free while attached).
- Security group inbound: **80** and **443** from `0.0.0.0/0` (+ 22 for SSH).
- DNS: A record (e.g. `concierge`) → the Elastic IP.

## 2. vLLM (the AI brain)

```
kubectl apply -f k8s/vllm.yaml
```

Serves `Qwen/Qwen2.5-7B-Instruct-AWQ` via an OpenAI-compatible API on
port 8000. Check: `curl localhost:8000/v1/models` (port-forward first).

## 3. Qdrant (the memory)

```
kubectl apply -f k8s/qdrant.yaml
```

v1.19.1 with a 10 Gi persistent volume — knowledge survives pod restarts.

## 4. Ingest knowledge

On a machine that can reach the cluster (port-forward Qdrant):

```
kubectl port-forward svc/qdrant 6333:6333 &
cd ingest
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Cabin guide** (kept out of git — contains the wifi password; place the
`.docx` next to the script):

```
python ingest_cabin.py     # 16 section chunks, recreates collection
```

**Local business research** (`data/idaho-city-business-guide.md`):

```
python ingest_web.py       # 18 chunks, upserts alongside cabin chunks
```

Verify: `curl localhost:6333/collections/concierge` → 34 points.

## 5. Retrieval API + web chat

```
cd api
docker build -t concierge-api:latest .
docker save concierge-api:latest | sudo k3s ctr -n k8s.io images import -

# guest key: the check-in link IS the credential - generate, never commit
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: concierge-key
stringData:
  guest-key: $(openssl rand -hex 16)   # <-- generate for real, don't paste this literally
EOF
kubectl apply -f k8s/concierge-api.yaml
```

Test locally: `kubectl port-forward svc/concierge-api 8080:8080 &`
(`nohup ... & disown` so it survives terminal disconnects), then
`curl "localhost:8080/?key=<key>"` → chat HTML, `curl localhost:8080/health`
→ `{"ok":true}`.

## 6. Public HTTPS

```
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.2/cert-manager.yaml
kubectl apply -f k8s/cluster-issuer.yaml   # edit the email first
kubectl apply -f k8s/ingress.yaml          # edit the hostname first
```

- Free Let's Encrypt certificate via HTTP-01 (needs DNS + port 80 reachable).
- Traefik rate limiting: 20 req/min average per IP.
- Guest-key auth on every route except `/health`.

Guest link: `https://<your-host>/?key=<guest-key>`

## 7. Verify end to end

| Test | Expected |
|---|---|
| `GET /?key=<wrong>` | 403 |
| `GET /?key=<right>` | 200, chat HTML |
| "What is the wifi password?" | Answer from cabin manual + citations |
| "Where can we get ice cream?" | Sarsaparilla + phone + call-ahead hedge |
| "What is the capital of France?" | Declined, host-contact pointer (guardrail) |

## Operational notes

- **Port-forwards die** when the terminal/SSH session ends — use `nohup`.
- After a `rollout restart`, re-attach port-forwards (they pin to the old pod).
- **Stop the instance after each session** (EC2 console → Stop). EBS keeps
  everything; GPU billing stops.
- Watch node disk (`df -h /`): torch-based images are large; prune with
  `docker image prune` if over ~85%.
- Refresh web chunks periodically (hours change seasonally); re-run
  `ingest_web.py` with an updated `retrieved_at`.

## Layout

```
api/        FastAPI retrieval API + mobile chat UI + Dockerfile
k8s/        vLLM, Qdrant, API, ingress, TLS issuer manifests
ingest/     .docx/.md chunking + embedding + Qdrant load scripts
data/       local business/attraction research (source for web chunks)
```
