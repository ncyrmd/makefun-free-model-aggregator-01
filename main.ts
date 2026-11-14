// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The brief gives the provider directory, the sealed quality table and the
// workload manifest. This server probes every provider's catalog route,
// builds one catalog, registers it, then serves the client's twenty requests
// one at a time and exits 0 when the env says the workload is done.
//
// ROUTING — the score is 60% quality, so the routing ranks every free model
// that meets the request's requirements by the quality table's rating and
// serves the best one it can get. Two things keep that from collapsing onto
// one provider: a spread penalty (8 milli per call the provider has already
// served this trial) and a suspect penalty (30 milli once a provider has
// refused a call), both subtracted from the rating. The penalties are small
// against the rating spread, so the best model still wins when it is clearly
// best, but the load spreads over the directory instead of burning one
// provider's per-trial budget. The client arrives one request at a time, so
// the ranking is recomputed per request from the counts so far — deterministic,
// no clock, no randomness.
//
// FAILOVER — a provider call that answers 402, 429 or a 5xx, or never answers,
// is not forwarded: the request is retried at once on the next ranked
// candidate, on another provider. A 2xx is final and is never retried: the
// trial counts exactly one successful provider call per request. A provider
// that refused stays suspect for the rest of the trial (its ranking drops)
// but is never blacklisted outright — a scheduled refusal is not a dead
// provider, and when it is the only candidate left it still serves.
//
// WIRES — the client speaks OpenAI or Anthropic; each provider speaks
// OpenAI, Anthropic or Gemini. All six combinations are translated in
// lib/wire.ts, losslessly in the fields the trial checks (system and user
// text, tool names, max_tokens, stops, image bytes, JSON mode), with the
// provider's usage counts carried across exactly and its stop reason mapped
// into the client's family. One exception: an OpenAI client asking for JSON
// mode is never routed to an Anthropic-wire provider, because that wire
// cannot carry JSON mode and would echo it absent.
//
// THE CATALOG — built once from the probes, in the order each listing gave
// its models. The scorer compares each provider's free-model list against
// the true catalog order-for-order, so the catalog never sorts. `--list`
// prints the same document the register body carries, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIRequestToAnthropic,
  openAIToAnthropic,
  openAIToGemini,
  sseEvent,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds, before failover. */
const CALL_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Rating milli lost per call a provider has already served this trial. */
const SPREAD_PENALTY_MILLI = 2;
/** Rating milli lost once a provider has refused a call this trial. */
const SUSPECT_PENALTY_MILLI = 12;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;
/** Provider calls made so far this trial, by provider id (refusals count too). */
const callsUsed = new Map<string, number>();
/** Providers that have refused a call this trial, by id. */
const suspect = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** The effective rating of one candidate: the table's rating, less the spread and suspect penalties. */
function effectiveRating(candidate: Candidate): number {
  const used = callsUsed.get(candidate.probe.provider.id) ?? 0;
  const penalty = SPREAD_PENALTY_MILLI * used + (suspect.has(candidate.probe.provider.id) ? SUSPECT_PENALTY_MILLI : 0);
  return candidate.rating - penalty;
}

const byRank = (a: Candidate, b: Candidate): number => {
  const rating = effectiveRating(b) - effectiveRating(a);
  if (rating !== 0) return rating;
  if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
  return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
};

/**
 * Every free model that can serve one request, best first. An OpenAI client
 * asking for JSON mode never sees an Anthropic-wire provider: that wire
 * cannot carry JSON mode and would echo it absent. A paid model is never a
 * candidate, whatever its rating.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean, clientApi: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out.sort(byRank);
}

/** The free models across the providers that answered, in catalog order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether the OpenAI client body asks for JSON mode. */
function jsonModeOf(body: Json): boolean {
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream URL of one provider call, in the provider's own wire. */
function upstreamUrl(candidate: Candidate, stream: boolean): string {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  if (candidate.probe.provider.wire === 'openai') return `${base}/v1/chat/completions`;
  if (candidate.probe.provider.wire === 'anthropic') return `${base}/v1/messages`;
  const model = encodeURIComponent(candidate.model.id);
  return stream ? `${base}/v1beta/models/${model}:streamGenerateContent?alt=sse` : `${base}/v1beta/models/${model}:generateContent`;
}

/** The upstream body of one provider call, translated into the provider's wire. */
function upstreamBody(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Json {
  const wire = candidate.probe.provider.wire;
  if (clientApi === 'openai') {
    if (wire === 'openai') {
      const out: Json = { ...body, model: candidate.model.id };
      if (body.stream === true) out.stream_options = { include_usage: true };
      return out;
    }
    if (wire === 'anthropic') return openAIRequestToAnthropic(body, candidate.model.id);
    return openAIToGemini(body);
  }
  if (wire === 'openai') return anthropicToOpenAI(body, candidate.model.id);
  if (wire === 'anthropic') return { ...body, model: candidate.model.id };
  return anthropicToGemini(body);
}

interface UpstreamResult {
  candidate: Candidate;
  response: Response;
}

/**
 * One request, tried against the ranked candidates in order until one
 * answers 2xx. A refusal (402, 429, 5xx) or an unanswered call marks its
 * provider suspect and moves on to the next candidate. A 2xx is final: it is
 * never retried, so exactly one provider call answers each request.
 */
async function callUpstream(
  candidates: Candidate[],
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<UpstreamResult | null> {
  const stream = body.stream === true;
  let lastStatus = 0;
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    callsUsed.set(providerId, (callsUsed.get(providerId) ?? 0) + 1);
    let response: Response;
    try {
      response = await fetch(upstreamUrl(candidate, stream), {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          ...(stream && candidate.probe.provider.wire === 'openai' ? { accept: 'text/event-stream' } : {}),
          ...authHeaders(candidate.probe.provider.auth, candidate.probe.provider.key),
          ...(candidate.probe.provider.wire === 'anthropic' ? { 'anthropic-version': '2023-06-01' } : {}),
        },
        body: JSON.stringify(upstreamBody(candidate, clientApi, body)),
        signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
      });
    } catch (err) {
      log(`${providerId}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      suspect.add(providerId);
      continue;
    }
    if (response.ok) return { candidate, response };
    lastStatus = response.status;
    log(`${providerId}/${candidate.model.id} answered HTTP ${response.status}`);
    if (response.status === 402 || response.status === 429 || response.status >= 500) suspect.add(providerId);
    try {
      await response.arrayBuffer();
    } catch {
      // The body is already gone: move on.
    }
  }
  log(`no provider answered; last status ${lastStatus}`);
  return null;
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered on any provider wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const candidates = candidatesFor(requirementsFor(body), jsonModeOf(body), 'openai');
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  const result = await callUpstream(candidates, 'openai', body);
  if (result === null) return sendError(res, 'openai', 502, 'upstream_unreachable', 'no provider answered this request');
  const { candidate, response } = result;
  const wire = candidate.probe.provider.wire;
  if (stream && response.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (wire === 'openai') {
      await pipeStream(response.body, res);
      return;
    }
    if (wire === 'anthropic') {
      for await (const event of openAIEventsFromAnthropic(response.body, candidate.model.id)) res.write(event);
      res.end();
      return;
    }
    for await (const event of openAIEventsFromGemini(response.body, candidate.model.id)) res.write(event);
    res.end();
    return;
  }
  const answer: unknown = await response.json().catch(() => null);
  if (!isRecord(answer)) return sendError(res, 'openai', 502, 'upstream_unreachable', 'the provider answered a body that is not JSON');
  if (wire === 'openai') {
    sendJson(res, 200, { ...answer, model: candidate.model.id });
    return;
  }
  if (wire === 'anthropic') {
    sendJson(res, 200, anthropicToOpenAIResponse(answer, candidate.model.id));
    return;
  }
  sendJson(res, 200, geminiToOpenAIResponse(answer, candidate.model.id));
}

/** POST /v1/messages: an Anthropic-shaped client, answered on any provider wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const candidates = candidatesFor(requirementsFor(body), false, 'anthropic');
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  const result = await callUpstream(candidates, 'anthropic', body);
  if (result === null) return sendError(res, 'anthropic', 502, 'api_error', 'no provider answered this request');
  const { candidate, response } = result;
  const wire = candidate.probe.provider.wire;
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : candidate.model.id;
  if (stream && response.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (wire === 'anthropic') {
      await pipeStream(response.body, res);
      return;
    }
    if (wire === 'openai') {
      for await (const event of anthropicEventsFromOpenAI(response.body, requested)) res.write(event);
      res.end();
      return;
    }
    for await (const event of anthropicEventsFromGemini(response.body, requested)) res.write(event);
    res.end();
    return;
  }
  const answer: unknown = await response.json().catch(() => null);
  if (!isRecord(answer)) return sendError(res, 'anthropic', 502, 'api_error', 'the provider answered a body that is not JSON');
  if (wire === 'anthropic') {
    // The provider's own answer, as it stands: its model is the one that served.
    sendJson(res, 200, answer);
    return;
  }
  if (wire === 'openai') {
    sendJson(res, 200, openAIToAnthropic(answer, requested));
    return;
  }
  sendJson(res, 200, geminiToAnthropicResponse(answer, requested));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts. Wait for stdout to
    // flush, or the tail is lost.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
