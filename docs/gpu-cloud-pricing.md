# Serverless GPU Pricing Comparison for StemScribe
**Last updated:** 2026-03-19
**Use case:** Audio stem separation with RoFormer model (~1GB weights, ~3 min GPU per song)
**Target GPU:** T4 / L4 equivalent (16-24GB VRAM is plenty)

---

## Quick Recommendation

| Scenario | Best Pick | Why |
|----------|-----------|-----|
| Beta (10 songs/day) | **Modal** | $30/mo free credits cover it entirely |
| Launch (100 songs/day) | **RunPod Serverless** | Cheapest per-second rate, FlashBoot cold starts |
| Growth (1000 songs/day) | **RunPod Serverless** or **Google Cloud Run** | RunPod cheapest raw cost; Cloud Run if you want GCP ecosystem |

---

## Provider Comparison Table

| Feature | RunPod Serverless | Modal | Google Cloud Run (GPU) | Replicate | Baseten |
|---------|------------------|-------|----------------------|-----------|---------|
| **T4 rate** | ~$0.00016/sec ($0.58/hr) | $0.000164/sec ($0.59/hr) | N/A (no T4) | $0.000225/sec ($0.81/hr) | $0.01052/min ($0.63/hr) |
| **L4 rate** | $0.00019/sec ($0.68/hr) | $0.000222/sec ($0.80/hr) | $0.000187/sec ($0.67/hr) | N/A (not listed) | $0.01414/min ($0.85/hr) |
| **A10 rate** | ~$0.00034/sec ($1.22/hr) | $0.000306/sec ($1.10/hr) | N/A | Not listed | $0.02012/min ($1.21/hr) |
| **A100 80GB rate** | $0.00076/sec ($2.74/hr) | $0.000694/sec ($2.50/hr) | N/A | $0.001400/sec ($5.04/hr) | $0.06667/min ($4.00/hr) |
| **Cold start** | ~2-5s (FlashBoot) | ~1s container + 10-60s model load (GPU snapshots help) | ~5s (GPU driver preloaded) | 10-180s depending on model | ~9-10s (cold-start snapshots) |
| **Scale to zero** | Yes | Yes (default 60s idle) | Yes (may idle up to 10 min with GPU) | Yes | Yes |
| **Idle cost** | $0 (Flex workers) | $0 | $0 (but possible 10 min idle window) | $0 (public models) | $0 |
| **Min billing** | Per second (rounded up) | Per second | Per second (rounded to 100ms) | Per second | Per minute |
| **Max exec time** | 5s - 7 days (configurable) | 1s - 24 hours (configurable) | 1 hour (GPU tasks) | Not published (likely ~30 min) | Not published |
| **Free tier/credits** | None standard (promos available) | $30/mo (Starter), $100/mo (Team) | None for GPU | Small new-user credit | New account credits (amount varies) |
| **Deployment** | Docker container | Python decorator (@app.function) | Docker container (Cloud Build) | Cog (Docker wrapper) or push model | Truss framework (Python) |
| **Model storage** | Container disk ($0.10/GB/mo) or network volumes ($0.05-0.07/GB/mo) | Included in container image or Modal Volumes | Container image (Artifact Registry) | Built into model version | Included in Truss bundle |
| **Ecosystem** | Standalone | Standalone (great Python DX) | GCP (IAM, Logging, Monitoring) | Standalone (now owned by Cloudflare) | Standalone |

---

## Dead / Not Applicable

| Provider | Status |
|----------|--------|
| **Banana.dev** | Shut down March 31, 2024. Gone. |
| **AWS Lambda with GPU** | Does not exist as of March 2026. AWS still has no native GPU support in Lambda. Use SageMaker Serverless instead (~$0.0015/sec, expensive, complex). |

---

## Cost Per Song (3 minutes = 180 seconds GPU time)

Using the cheapest viable GPU (T4 where available, L4 otherwise):

| Provider | GPU | Rate/sec | Cost per song |
|----------|-----|----------|---------------|
| **RunPod Serverless** | T4-class | $0.00016 | **$0.029** |
| **Modal** | T4 | $0.000164 | **$0.030** |
| **Google Cloud Run** | L4 | $0.000187 | **$0.034** |
| **Replicate** | T4 | $0.000225 | **$0.041** |
| **Baseten** | T4 | $0.01052/min | **$0.032** |

Note: These are pure GPU compute costs. Cloud Run also charges for CPU + memory (~$0.01-0.02 additional per song).

---

## Monthly Cost Projections

### 10 songs/day (Beta) = 300 songs/month

| Provider | GPU | Monthly Cost | After Free Credits |
|----------|-----|-------------|-------------------|
| **RunPod** | T4-class | $8.64 | $8.64 |
| **Modal** | T4 | $8.86 | **$0.00** (covered by $30 free) |
| **Cloud Run** | L4 | $10.08 + ~$4 CPU/mem | ~$14.08 |
| **Replicate** | T4 | $12.15 | ~$12.15 |
| **Baseten** | T4 | $9.47 | Partially covered by signup credits |

### 100 songs/day (Launch) = 3,000 songs/month

| Provider | GPU | Monthly Cost | After Free Credits |
|----------|-----|-------------|-------------------|
| **RunPod** | T4-class | **$86.40** | $86.40 |
| **Modal** | T4 | $88.56 | **$58.56** ($30 credit) |
| **Cloud Run** | L4 | $100.80 + ~$40 CPU/mem | ~$140.80 |
| **Replicate** | T4 | $121.50 | $121.50 |
| **Baseten** | T4 | $94.68 | $94.68 |

### 1,000 songs/day (Growth) = 30,000 songs/month

| Provider | GPU | Monthly Cost | After Free Credits |
|----------|-----|-------------|-------------------|
| **RunPod** | T4-class | **$864** | $864 |
| **Modal** | T4 | $886 | $856 ($30 credit) |
| **Cloud Run** | L4 | $1,008 + ~$400 CPU/mem | ~$1,408 |
| **Replicate** | T4 | $1,215 | $1,215 |
| **Baseten** | T4 | $947 | $947 |

---

## Detailed Provider Notes

### 1. RunPod Serverless
- **Jeff already has an account** -- just use Serverless, not GPU Pods
- FlashBoot gets cold starts down to ~2 seconds (some report sub-500ms)
- Flex Workers = pure pay-per-use, $0 when idle
- Active Workers = 20-30% discount if you want always-warm instances
- Default execution timeout: 600s (configurable up to 7 days)
- $80/hr default spend cap (protects against runaway costs)
- Deploy via Docker container pushed to RunPod registry
- Network volumes for persistent model weight storage ($0.05-0.07/GB/mo)
- **Cheapest option at scale**

### 2. Modal
- **Best developer experience** -- pure Python, no Docker needed
- `@app.function(gpu="T4")` decorator, deploy with `modal deploy`
- $30/month free credits on Starter plan (covers beta volume entirely)
- Container boots in ~1 second; model loading adds time
- GPU memory snapshots can cut cold starts dramatically (10x improvement)
- Default timeout 300s, configurable up to 24 hours
- Scales to zero by default (60s idle timeout)
- Great for prototyping and iteration speed
- **Best for beta/development phase**

### 3. Google Cloud Run (GPU)
- Only NVIDIA L4 available (no T4) -- L4 is fine for RoFormer
- $0.000187/sec for L4 in Tier 1 regions
- Also charges for CPU (min 4 vCPU) + memory (min 16 GiB) separately
- Cold start ~5 seconds (GPU driver preloaded)
- GPU instances may idle up to 10 minutes before scaling to zero (potential minor cost)
- Max 1 hour for GPU tasks
- Requires min 4 CPU + 16 GiB memory alongside GPU
- Available in limited regions: us-central1, europe-west1, europe-west4, asia-southeast1, asia-south1
- Fits Jeff's Google ecosystem but more expensive due to CPU/mem overhead
- **Good if you want everything in GCP**

### 4. Replicate
- Now owned by Cloudflare (acquired early 2026)
- Simple API: push model with Cog, call via REST API
- T4 at $0.000225/sec is ~40% more expensive than RunPod/Modal
- Cold starts 10-180 seconds (worse than competitors)
- Public models: no setup/idle charges. Private models: pay for setup + idle
- Good for sharing models publicly, less ideal for private production use
- **Best for public model distribution, not cost-optimized for private**

### 5. Baseten
- Uses Truss framework (open-source, Python-based)
- Per-minute billing (not per-second) -- slightly less granular
- T4 at $0.01052/min = $0.63/hr (competitive)
- Cold-start snapshots bring models online in ~9-10 seconds
- Scale to zero supported
- Good autoscaling features
- Less community/docs compared to RunPod/Modal
- **Decent option but per-minute billing is a downside for short jobs**

---

## Migration Path Recommendation

```
Phase 1 (Now - Beta):     Modal with T4
                           - $0 cost with free credits
                           - Fastest to deploy (Python decorator)
                           - Perfect for 10 songs/day

Phase 2 (Launch):          RunPod Serverless with T4-class GPU
                           - Cheapest at 100+ songs/day
                           - FlashBoot minimizes cold starts
                           - Jeff already has an account
                           - Docker-based (slightly more setup)

Phase 3 (Growth):          RunPod Serverless (continue)
                           - Consider Active Workers for 20-30% discount
                           - at 1000 songs/day, saves ~$170-260/mo vs Active Workers
                           - Or negotiate volume pricing
```

---

## Key Warnings

1. **RunPod GPU Pods are NOT serverless** -- Jeff got burned with $100 idle charges before. Serverless Flex Workers have $0 idle cost.
2. **Google Cloud Run GPU idle window** -- instances may stay warm up to 10 minutes, causing minor charges even with no traffic.
3. **Replicate private model costs** -- you pay for setup time + idle time on private models, unlike public models.
4. **Baseten per-minute billing** -- if your job takes 3 min 1 sec, you pay for 4 minutes. Less impactful at scale but worth noting.
5. **Cold start + model loading** -- the 3-minute GPU time estimate assumes model is already loaded. First request after cold start adds 5-60 seconds of billable time for model weight loading depending on provider.

---

## Sources

- [RunPod Pricing](https://www.runpod.io/pricing)
- [RunPod Serverless Pricing Docs](https://docs.runpod.io/serverless/pricing)
- [RunPod FlashBoot](https://www.runpod.io/blog/introducing-flashboot-serverless-cold-start)
- [Modal Pricing](https://modal.com/pricing)
- [Modal Cold Start Docs](https://modal.com/docs/guide/cold-start)
- [Modal Timeouts Docs](https://modal.com/docs/guide/timeouts)
- [Google Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [Google Cloud Run GPU Docs](https://docs.cloud.google.com/run/docs/configuring/services/gpu)
- [Replicate Pricing](https://replicate.com/pricing)
- [Baseten Pricing](https://www.baseten.co/pricing/)
- [Baseten Cold Starts](https://docs.baseten.co/performance/cold-starts)
- [Banana.dev Sunset Announcement](https://www.banana.dev/blog/sunset)
