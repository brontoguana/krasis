#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const outputDir = process.env.ADq_CAPTURE_OUTPUT_DIR;
const modelID = process.env.ADq_CAPTURE_MODEL_ID;
const listenPort = Number(process.env.ADq_CAPTURE_LISTEN_PORT || "8012");

if (!outputDir || !modelID || !Number.isInteger(listenPort) || listenPort <= 0) {
  throw new Error("ADq_CAPTURE_OUTPUT_DIR, ADq_CAPTURE_MODEL_ID, and a positive port are required");
}

fs.mkdirSync(outputDir, { recursive: true });

function sendJSON(response, status, value) {
  const body = JSON.stringify(value);
  response.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  });
  response.end(body);
}

function completeCapture(response, streaming) {
  if (!streaming) {
    sendJSON(response, 200, {
      id: "adq-replay-capture",
      object: "chat.completion",
      model: modelID,
      choices: [{ index: 0, message: { role: "assistant", content: "CAPTURE_ONLY" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    });
    return;
  }
  response.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  response.write(`data: ${JSON.stringify({
    id: "adq-replay-capture",
    object: "chat.completion.chunk",
    model: modelID,
    choices: [{ index: 0, delta: { role: "assistant", content: "CAPTURE_ONLY" }, finish_reason: null }],
  })}\n\n`);
  response.write(`data: ${JSON.stringify({
    id: "adq-replay-capture",
    object: "chat.completion.chunk",
    model: modelID,
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  })}\n\n`);
  response.end("data: [DONE]\n\n");
}

let postCount = 0;
const server = http.createServer((request, response) => {
  if (request.method === "GET" && request.url === "/v1/models") {
    sendJSON(response, 200, {
      object: "list",
      data: [{ id: modelID, object: "model", owned_by: "krasis" }],
    });
    return;
  }
  if (request.method !== "POST" || request.url !== "/v1/chat/completions") {
    sendJSON(response, 404, { error: { message: "capture relay accepts only /v1/models and /v1/chat/completions" } });
    return;
  }

  const chunks = [];
  request.on("data", (chunk) => chunks.push(chunk));
  request.on("end", () => {
    const body = Buffer.concat(chunks);
    postCount += 1;
    const stem = `request-${String(postCount).padStart(3, "0")}`;
    fs.writeFileSync(path.join(outputDir, `${stem}.json`), body);
    fs.writeFileSync(
      path.join(outputDir, `${stem}.headers.json`),
      `${JSON.stringify({ method: request.method, url: request.url, headers: request.headers }, null, 2)}\n`,
    );
    let parsed;
    try {
      parsed = JSON.parse(body.toString("utf8"));
    } catch (error) {
      sendJSON(response, 400, { error: { message: `captured invalid JSON: ${error.message}` } });
      return;
    }
    completeCapture(response, parsed.stream === true);
  });
});

server.listen(listenPort, "0.0.0.0", () => {
  process.stdout.write(`ADQ replay capture relay ready on ${listenPort}\n`);
});

for (const signal of ["SIGTERM", "SIGINT"]) {
  process.on(signal, () => server.close(() => process.exit(0)));
}
