# MORI

MORI (**M**emory **O**ffloader with **R**elative **I**dleness) is a scheduler for
serving LLM agents. It decides which agent programs keep their KV cache in GPU
memory, which are offloaded to CPU memory, and which wait. The design is described in
[arXiv:2606.00866](https://arxiv.org/abs/2606.00866).

An agent program alternates between LLM inference and tool calls, and each request
extends the context of the previous one. Keeping a program's KV cache on the GPU
across tool calls avoids re-prefilling that context at every step, but the KV cache
of many concurrent programs does not fit in GPU memory. Offloading to CPU memory
helps only if the right programs are offloaded, and that is hard to decide: tool
calls last from milliseconds to minutes, and their durations are unknown in advance.

MORI ranks programs by **idleness**, the fraction of a program's recent time spent in
tool calls. The least idle programs stay in GPU memory, the next are offloaded to CPU
memory, and the rest wait with their KV cache dropped. The capacity of each tier
decides where the ranking is cut, so MORI uses no duration thresholds, does not
predict how long a tool call will take, and needs no tuning for a different
GPU-to-CPU memory ratio. It also proactively demotes idle programs to make room for
programs whose tool calls have returned.

On real coding-agent traces under high load, MORI achieves on average 64% higher
throughput and 72% lower p90 TTFT than the best offloading baseline.

## Repository

MORI has two parts, and both are required:

| Directory | Part |
|---|---|
| `ThunderAgent/` | The scheduler. An OpenAI-compatible proxy that tracks every program and makes MORI's decisions. Based on [ThunderAgent](https://github.com/ThunderAgent-org/ThunderAgent) at `76cd325`. |
| `sglang/` | The inference engine. It tags KV cache with the program that owns it, and evicts and offloads KV in the order that the scheduler's decisions imply. Based on [SGLang](https://github.com/sgl-project/sglang) at `1264a2e9f`. |

```
agent clients ── OpenAI API + program_id ──▶ scheduler (ThunderAgent) ──▶ engine (SGLang)
```

## Usage

### 1. Install

Install both parts from source:

```bash
git clone https://github.com/cblmemo/mori.git
cd mori
pip install -e "sglang/python"
pip install -e ThunderAgent
```

### 2. Start the engine

Start the SGLang server:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 --port 30000 \
  --enable-hierarchical-cache --hicache-write-policy write_back --hicache-ratio 1.0 \
  --schedule-policy lpm --enable-metrics --enable-cache-report
```

| Flag | Why MORI needs it |
|---|---|
| `--enable-hierarchical-cache --hicache-write-policy write_back` | The HiCache host pool is MORI's CPU tier. Use `write_back`; upstream's default is `write_through`. |
| `--hicache-ratio R` (or `--hicache-size GB`) | Sets the CPU tier size relative to the GPU KV pool. We evaluated 1.0 and 2.0. Without HiCache, MORI still runs, but with only the GPU and waiting tiers. |
| `--enable-metrics` | The scheduler reads the pool sizes `sglang:max_total_num_tokens` and `sglang:hicache_host_total_tokens`. |
| `--enable-cache-report` | Responses carry `usage.prompt_tokens_details.cached_tokens`, which feeds the shared-prefix estimate. |
| `--schedule-policy lpm` | The policy we validated with (upstream's default is `fcfs`). |

Keep the radix cache enabled (the default). This SGLang build adds no new server
arguments. Requests without `program_id` keep stock SGLang scheduling and eviction;
only the HiCache correctness fixes apply to them.

### 3. Start the scheduler

```bash
thunderagent --backend-type sglang --backends http://localhost:30000 --port 8300 --metrics
```

- `--backends` is the URL of the SGLang server.
- `--scheduler-interval` (default 5 s) is the MORI tick period.
- `--acting-token-weight` (default 1.0) weights idle programs' tokens in the GPU book.
- MORI requires `--backend-type sglang` (the default). It polls the backends' metrics
  even without `--metrics`.

### 4. Send requests

Clients talk to the scheduler's OpenAI-compatible endpoint, not to SGLang.

```python
import requests
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8300/v1", api_key="none")
pid = "session-42"            # one id per agent session; each subagent gets its own

resp = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    messages=messages,
    extra_body={"program_id": pid},
)
# ... tool calls, more requests with the same pid ...

requests.post("http://localhost:8300/programs/release", json={"program_id": pid})
```

- **`program_id`**: the scheduling unit is one agent program (one session). Use one
  id for all of that session's requests. Give every agent that runs concurrently,
  including each subagent, its own id. Ids go into URL query strings, so use
  URL-safe characters. Requests without an id share the program `"default"`.
- **Release**: call `POST /programs/release` when the session ends. The scheduler
  drops the program, and the engine forgets its type, so its KV becomes the first to
  be evicted.
- **Streaming**: the scheduler turns on `stream_options.include_usage` itself; it
  needs the usage to track tokens.

### 5. Check that it works

- Scheduler `GET /metrics`, under the backend's entry:
  - `cache_config.cpu_total_tokens_capacity > 0`: the scheduler sees the host pool.
  - `warm_queue_count` and `warm_queue_tokens`: the CPU tier is in use.
- The scheduler log has `[MORI] ...` lines for demotions and promotions.
- The engine endpoints respond, for example:
  `curl -X POST "localhost:30000/mark_program_type?program_id=t&program_type=async"`
  returns `{"success": true, ...}`.
