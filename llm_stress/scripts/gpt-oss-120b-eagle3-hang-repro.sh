#!/usr/bin/env bash
# Stress scenario: gpt-oss-120b + EAGLE3, TP2 — decode-wedge HANG REPRO.
#
# This is a regression gate for the TRT-LLM all-reduce-fusion decode wedge.
# It runs the *unmitigated* path on purpose, so:
#
#   * While the bug is OPEN this scenario is EXPECTED TO FAIL — the harness's
#     fatal global-stall detector fires within a couple of minutes (exit 2).
#   * Once the bug is FIXED it should run clean (exit 0) and then guards
#     against regressions.
#
# The wedge is NOT model-specific: the fused norm+allreduce kernel is
# auto-enabled on single-node TP (see server_args: enable_allreduce_fusion is
# auto-on for supported TP configs), and gpt-oss-120b mxfp4 reproduces it at
# just TP2 — a cheap, fast repro (~1 min load, wedges in ~75 s).
#
# Mitigation (the actual off-switch): --comm-fusion-max-num-tokens 0 disables
# the fused kernel (it only fires for num_tokens <= that threshold, default
# 2048). NOTE: --disable-custom-all-reduce does NOT disable this fused path.
#
# Tunables (env): OUT_DIR, DURATION_S (default 600), MAX_CONCURRENCY (default 24).
set -uo pipefail
cd "$(dirname "$0")/../.."

OUT_DIR="${OUT_DIR:-$PWD/stress-out}"
DURATION_S="${DURATION_S:-600}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-24}"
# Server batch cap must track the load cap, else concurrency above it just queues
# and decode batches never grow into the regime that surfaces the wedge.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$MAX_CONCURRENCY}"
# Sawtooth period: shorter => faster cycling through continuous-batching size
# transitions (the documented trigger), more transitions per unit time.
TRIANGLE_PERIOD="${TRIANGLE_PERIOD:-180}"
# Workload intensity knobs. Raising VERY_LONG_WEIGHT (vs the fixed short=45 /
# medium=30 / long=20 gen weights) makes long-decode requests dominate, so the
# decode batch stays large and KV actually fills — the sustained-decode regime
# the fused-allreduce wedge needs. Lower CANCEL_FRACTION keeps those decodes
# alive instead of freeing KV early.
VERY_LONG_WEIGHT="${VERY_LONG_WEIGHT:-20}"
CANCEL_FRACTION="${CANCEL_FRACTION:-0.15}"
MAX_TOKENS_CAP="${MAX_TOKENS_CAP:-32768}"
PROMPT_TOKENS_MAX="${PROMPT_TOKENS_MAX:-40000}"
# PORT lets this run alongside an existing server (e.g. pin GPUs with
# CUDA_VISIBLE_DEVICES and bump PORT to avoid colliding on :8000). Host is fixed
# to loopback on purpose — do NOT read $HOST from the env (conda exports HOST as
# the build triplet, e.g. x86_64-conda-linux-gnu, which would poison --host).
SERVE_HOST="127.0.0.1"
# SMG pins its Prometheus exporter to 8413 by default; override when running
# alongside another tokenspeed server so the second SMG can bind a free port.
PROM_PORT="${PROM_PORT:-8413}"
# Tensor-parallel degree (applied to both attn and moe). TP2 is the cheap repro;
# bump to 4/8 to probe whether the all-reduce-fusion path / spec-acceptance
# collapse turns into a wedge at higher TP (mirrors prod, which runs higher TP).
TP="${TP:-2}"
# Fused norm+allreduce mitigation. Empty => leave the fused path ON (auto-enabled
# for single-node TP; this is the unmitigated wedge gate). Set to 0 to DISABLE the
# fused kernel (prod runs with `--comm-fusion-max-num-tokens 0`): use this to A/B
# whether the wedge is the AR-fusion path or survives the prod mitigation.
COMM_FUSION_MAX_NUM_TOKENS="${COMM_FUSION_MAX_NUM_TOKENS:-}"
COMM_FUSION_FLAG=""
[ -n "$COMM_FUSION_MAX_NUM_TOKENS" ] && \
  COMM_FUSION_FLAG="--comm-fusion-max-num-tokens $COMM_FUSION_MAX_NUM_TOKENS"
# HiCache KVStore allocates host_to_device_ratio (default 2.0) x device-pool of
# HOST RAM *per rank* — ~261 GB x TP at these sizes, ~1 TB at TP4, which the
# OOM-killer trips on a contended box. Set DISABLE_KVSTORE=1 for a lean repro
# (the AR-fusion decode wedge is unrelated to HiCache).
KVSTORE_FLAG=""
[ -n "${DISABLE_KVSTORE:-}" ] && KVSTORE_FLAG="--disable-kvstore"
# Shrink the HiCache host allocation (default ratio 2.0 => ~1 TB at TP4) without
# disabling it, so the wedge still reproduces on a RAM-contended box.
[ -n "${KVSTORE_RATIO:-}" ] && KVSTORE_FLAG="$KVSTORE_FLAG --kvstore-ratio $KVSTORE_RATIO"
mkdir -p "$OUT_DIR"

# Custom all-reduce + fused allreduce left ON (auto-enabled for single-node TP).
SERVE="tokenspeed serve \
  --model openai/gpt-oss-120b \
  --trust-remote-code \
  --host $SERVE_HOST --port $PORT --prometheus-port $PROM_PORT \
  --attn-tp-size $TP --moe-tp-size $TP \
  --max-model-len 80000 --max-num-seqs $MAX_NUM_SEQS \
  --gpu-memory-utilization 0.9 \
  --attention-backend trtllm --moe-backend flashinfer_mxfp4 \
  --reasoning-parser base \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path nvidia/gpt-oss-120b-Eagle3-long-context \
  --speculative-num-steps 3 $COMM_FUSION_FLAG $KVSTORE_FLAG \
  --enable-cache-report --enable-metrics"

# Text-only reality_mix (gpt-oss has no vision). grammar disabled (irrelevant to
# the wedge; avoids gpt-oss structured-output noise). Prompts capped to fit the
# 80k context; aggressive long-decode mirrors the regime that surfaces the wedge.
python3 -m llm_stress run \
  --launch-cmd "$SERVE" \
  --launch-timeout 1800 \
  --base-url http://$SERVE_HOST:$PORT \
  --model openai/gpt-oss-120b \
  --workload reality_mix \
  --workload-arg grammar_fraction=0 \
  --workload-arg cancel_fraction=$CANCEL_FRACTION \
  --workload-arg cached_fraction=0.5 \
  --workload-arg prompt_tokens_max=$PROMPT_TOKENS_MAX \
  --workload-arg max_tokens_cap=$MAX_TOKENS_CAP \
  --workload-arg very_long_weight=$VERY_LONG_WEIGHT \
  --arrival sawtooth --min-concurrency 1 \
  --max-concurrency "$MAX_CONCURRENCY" --triangle-period "$TRIANGLE_PERIOD" \
  --duration "$DURATION_S" --request-timeout 1200 \
  --stall-timeout 20 --global-stall-timeout "${GLOBAL_STALL_TIMEOUT:-20}" \
  --metrics-interval 10 --accept-len-min 1.1 \
  --out "$OUT_DIR"
rc=$?

if [ -f "$OUT_DIR/events.jsonl" ]; then
  python3 -m llm_stress summarize --events "$OUT_DIR/events.jsonl" \
    | tee "$OUT_DIR/summary.txt" || true
fi

exit "$rc"
