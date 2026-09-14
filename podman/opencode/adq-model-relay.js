#!/usr/bin/env node
"use strict";

const http = require("node:http");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const upstreamHost = process.env.ADq_UPSTREAM_HOST;
const upstreamPort = Number(process.env.ADq_UPSTREAM_PORT);
const listenPort = Number(process.env.ADq_LISTEN_PORT || "8012");
const captureDir = process.env.ADq_CAPTURE_DIR || "";
let requestSequence = 0;

if (!upstreamHost || !Number.isInteger(upstreamPort) || upstreamPort <= 0) {
  throw new Error("ADq_UPSTREAM_HOST and a positive ADq_UPSTREAM_PORT are required");
}

const blockedHeaders = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

function filteredHeaders(headers) {
  return Object.fromEntries(
    Object.entries(headers).filter(([name]) => !blockedHeaders.has(name.toLowerCase())),
  );
}

function prepareUpstreamBody(method, requestUrl, requestChunks) {
  const incomingBody = Buffer.concat(requestChunks);
  if (method !== "POST" || requestUrl !== "/v1/chat/completions") {
    return { body: incomingBody, greedy: false };
  }

  let payload;
  try {
    payload = JSON.parse(incomingBody.toString("utf8"));
  } catch (error) {
    throw new Error(`ADQ chat request is not valid JSON: ${error.message}`);
  }
  if (payload === null || Array.isArray(payload) || typeof payload !== "object") {
    throw new Error("ADQ chat request must be a JSON object");
  }
  if (Object.hasOwn(payload, "temperature") && payload.temperature !== 0) {
    throw new Error(
      `ADQ requires temperature 0, received ${JSON.stringify(payload.temperature)}`,
    );
  }
  payload.temperature = 0;
  return { body: Buffer.from(JSON.stringify(payload)), greedy: true };
}

const server = http.createServer((request, response) => {
  if (!request.url.startsWith("/v1/")) {
    response.writeHead(403, { "content-type": "text/plain" });
    response.end("ADQ relay permits only the fixed Krasis /v1 endpoint\n");
    return;
  }
  if (request.method !== "GET" && request.method !== "POST") {
    response.writeHead(405, { "content-type": "text/plain" });
    response.end("method not permitted\n");
    return;
  }

  const sequence = ++requestSequence;
  const requestChunks = [];
  const responseChunks = [];
  request.on("data", (chunk) => requestChunks.push(Buffer.from(chunk)));
  request.on("end", () => {
    let prepared;
    try {
      prepared = prepareUpstreamBody(request.method, request.url, requestChunks);
    } catch (error) {
      response.writeHead(400, { "content-type": "text/plain" });
      response.end(`${error.message}\n`);
      return;
    }
    const requestHeaders = {
      ...filteredHeaders(request.headers),
      host: `${upstreamHost}:${upstreamPort}`,
    };
    if (request.method === "POST") {
      requestHeaders["content-length"] = String(prepared.body.length);
    }
    const upstream = http.request(
      {
        hostname: upstreamHost,
        port: upstreamPort,
        method: request.method,
        path: request.url,
        headers: requestHeaders,
      },
      (upstreamResponse) => {
      let responseTail = "";
      response.writeHead(
        upstreamResponse.statusCode || 502,
        filteredHeaders(upstreamResponse.headers),
      );
      upstreamResponse.on("data", (chunk) => {
        response.write(chunk);
        responseChunks.push(Buffer.from(chunk));
        responseTail = (responseTail + chunk.toString("utf8")).slice(-262144);
      });
      upstreamResponse.on("end", () => {
        const match = responseTail.match(/"prompt_tokens"\s*:\s*([0-9]+)/)
          || responseTail.match(/"prompt_n"\s*:\s*([0-9]+)/);
        if (match) {
          process.stdout.write(
            `ADQ_RELAY_TIMING prompt_tokens=${match[1]} path=${request.url}\n`,
          );
        }
        if (captureDir) {
          fs.mkdirSync(captureDir, { recursive: true });
          const stem = String(sequence).padStart(4, "0");
          const requestBody = prepared.body;
          const responseBody = Buffer.concat(responseChunks);
          fs.writeFileSync(path.join(captureDir, `${stem}.request.raw`), requestBody);
          fs.writeFileSync(path.join(captureDir, `${stem}.response.raw`), responseBody);
          fs.writeFileSync(
            path.join(captureDir, `${stem}.meta.json`),
            JSON.stringify({
              sequence,
              method: request.method,
              path: request.url,
              status: upstreamResponse.statusCode || 502,
              request_bytes: requestBody.length,
              response_bytes: responseBody.length,
              request_sha256: crypto.createHash("sha256").update(requestBody).digest("hex"),
              response_sha256: crypto.createHash("sha256").update(responseBody).digest("hex"),
              greedy_temperature_zero: prepared.greedy,
            }, null, 2) + "\n",
          );
        }
        response.end();
      });
      },
    );
    upstream.on("error", (error) => {
      if (!response.headersSent) {
        response.writeHead(502, { "content-type": "text/plain" });
      }
      response.end(`fixed Krasis upstream unavailable: ${error.message}\n`);
    });
    upstream.end(prepared.body);
  });
});

server.listen(listenPort, "0.0.0.0", () => {
  process.stdout.write(
    `ADQ fixed relay listening on ${listenPort} for ${upstreamHost}:${upstreamPort}\n`,
  );
});

for (const signal of ["SIGTERM", "SIGINT"]) {
  process.on(signal, () => server.close(() => process.exit(0)));
}
