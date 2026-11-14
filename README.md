# The best free AI model aggregator

One server, two client APIs, ten simulated free-tier providers — and twenty
requests that each need the best free model able to serve them.

## What it does

`main.ts` (Node 22, standard library only) reads the trial brief
(`$MAKEFUN_PROMPT_FILE`): the provider directory, the sealed quality table,
and the workload manifest. It probes every provider's catalog route in the
provider's own wire, builds one catalog, registers its port and catalog with
the env, then serves the client's requests one at a time:

- `GET /v1/models` — the free models, OpenAI-shaped for an `Authorization:
  Bearer` request, Anthropic-shaped for an `x-api-key` request.
- `POST /v1/chat/completions` — OpenAI Chat Completions, JSON or SSE.
- `POST /v1/messages` — Anthropic Messages, JSON or SSE.

It polls `GET $MAKEFUN_ENV_URL/work` and exits 0 when the env says the
workload is done. `node main.ts --list` prints the catalog document and
exits 0 without starting the server.

## Routing

The score is 60% quality, so routing ranks every free model that meets the
request's requirements by the quality table's `rating_milli` and serves the
best one it can get — never a paid model, whatever its rating. Two penalties
keep that from collapsing onto one provider:

- **Spread** — 8 milli per call the provider has already served this trial.
- **Suspect** — 30 milli once a provider has refused a call this trial.

The penalties are small against the rating spread, so the best model still
wins when it is clearly best, but the load spreads over the directory instead
of burning one provider's per-trial budget. The ranking is recomputed per
request from the counts so far: deterministic, no clock, no randomness.

One exception: an OpenAI client asking for JSON mode is never routed to an
Anthropic-wire provider — that wire cannot carry JSON mode and would echo it
absent.

## Failover

A provider call that answers 402, 429 or a 5xx, or never answers, is not
forwarded: the request is retried at once on the next ranked candidate, on
another provider. A 2xx is final and is never retried, so exactly one
provider call answers each request. A provider that refused stays suspect
for the rest of the trial but is never blacklisted outright — a scheduled
refusal is not a dead provider, and when it is the only candidate left it
still serves.

## Wires

The client speaks OpenAI or Anthropic; each provider speaks OpenAI,
Anthropic or Gemini. All six combinations are translated in `lib/wire.ts`,
losslessly in the fields the trial checks: system and user text, tool names,
`max_tokens`, stop sequences, image bytes and JSON mode — with the
provider's usage counts carried across exactly and its stop reason mapped
into the client's family.

## The catalog

Built once from the probes, in the order each listing gave its models. The
scorer compares each provider's free-model list against the true catalog
order-for-order, so the catalog never sorts its models. `--list` prints the
same document the register body carries, byte for byte.

## Files

- `main.ts` — the server: brief, catalog, routes, routing, failover,
  register, poll.
- `lib/providers.ts` — the brief's three documents, the catalog probe, the
  quality table, capabilities and requirements.
- `lib/wire.ts` — the client wires and the three upstream wires, JSON and
  streaming.
- `LICENSE` — MIT.
- `.makefun/owner.json` — the payout wallet claim.

## Run it

```sh
node main.ts --list   # print the catalog, exit 0
node main.ts          # serve (needs MAKEFUN_ENV_URL + MAKEFUN_PROMPT_FILE)
```
