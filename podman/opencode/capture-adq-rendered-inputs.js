#!/usr/bin/env node
"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

if (process.env.KRASIS_DEV_SCRIPT !== "1") {
  throw new Error("Run through ./dev adq-opencode; direct execution is unsupported");
}
if (process.argv.length !== 4) {
  throw new Error("usage: capture-adq-rendered-inputs.js <http-base-url> <relay-capture-dir>");
}

const base = new URL(process.argv[2]);
const captureDir = process.argv[3];

function sha256(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function postJson(endpoint, body) {
  const payload = Buffer.from(JSON.stringify(body));
  return new Promise((resolve, reject) => {
    const request = http.request(
      {
        hostname: base.hostname,
        port: base.port,
        path: endpoint,
        method: "POST",
        headers: {
          "content-type": "application/json",
          "content-length": payload.length,
        },
      },
      (response) => {
        const chunks = [];
        response.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
        response.on("end", () => {
          const raw = Buffer.concat(chunks);
          if ((response.statusCode || 500) >= 300) {
            reject(new Error(`${endpoint} returned ${response.statusCode}: ${raw.toString("utf8")}`));
            return;
          }
          try {
            resolve(JSON.parse(raw.toString("utf8")));
          } catch (error) {
            reject(new Error(`${endpoint} returned invalid JSON: ${error.message}`));
          }
        });
      },
    );
    request.on("error", reject);
    request.end(payload);
  });
}

async function main() {
  const requestFiles = fs.readdirSync(captureDir)
    .filter((name) => name.endsWith(".request.raw"))
    .sort();
  let captured = 0;
  for (const requestName of requestFiles) {
    const stem = requestName.slice(0, -".request.raw".length);
    const meta = JSON.parse(fs.readFileSync(path.join(captureDir, `${stem}.meta.json`), "utf8"));
    if (meta.path !== "/v1/chat/completions") continue;
    const rawRequest = fs.readFileSync(path.join(captureDir, requestName));
    const request = JSON.parse(rawRequest.toString("utf8"));
    const templateRequest = {
      messages: request.messages,
      tools: request.tools,
      add_generation_prompt: true,
    };
    if (Object.prototype.hasOwnProperty.call(request, "tool_choice")) {
      templateRequest.tool_choice = request.tool_choice;
    }
    if (Object.prototype.hasOwnProperty.call(request, "enable_thinking")) {
      templateRequest.enable_thinking = request.enable_thinking;
    }
    const rendered = await postJson("/apply-template", templateRequest);
    if (typeof rendered.prompt !== "string") {
      throw new Error(`${stem}: /apply-template omitted prompt`);
    }
    const tokenized = await postJson("/tokenize", {
      content: rendered.prompt,
      add_special: false,
    });
    if (!Array.isArray(tokenized.tokens) || !tokenized.tokens.every(Number.isInteger)) {
      throw new Error(`${stem}: /tokenize omitted integer token IDs`);
    }
    const rawResponse = fs.readFileSync(path.join(captureDir, `${stem}.response.raw`), "utf8");
    const promptTokenMatches = [...rawResponse.matchAll(/"prompt_tokens"\s*:\s*([0-9]+)/g)];
    if (promptTokenMatches.length === 0) {
      throw new Error(`${stem}: inference response omitted prompt_tokens`);
    }
    const inferencePromptTokens = Number(promptTokenMatches.at(-1)[1]);
    if (inferencePromptTokens !== tokenized.tokens.length) {
      throw new Error(
        `${stem}: rendered token count ${tokenized.tokens.length} disagrees with ` +
        `inference prompt_tokens ${inferencePromptTokens}`,
      );
    }
    const tokenBytes = Buffer.from(JSON.stringify(tokenized.tokens));
    const artifact = {
      format: "krasis_adq_rendered_input_capture",
      format_version: 1,
      sequence: meta.sequence,
      request_sha256: meta.request_sha256,
      rendered_prompt: rendered.prompt,
      rendered_prompt_bytes: Buffer.byteLength(rendered.prompt),
      rendered_prompt_sha256: sha256(Buffer.from(rendered.prompt)),
      input_token_ids: tokenized.tokens,
      input_token_count: tokenized.tokens.length,
      input_token_ids_sha256: sha256(tokenBytes),
      inference_prompt_tokens: inferencePromptTokens,
      inference_token_count_match: true,
    };
    fs.writeFileSync(
      path.join(captureDir, `${stem}.rendered.json`),
      JSON.stringify(artifact, null, 2) + "\n",
    );
    process.stdout.write(
      `ADQ_RENDERED_CAPTURE sequence=${meta.sequence} tokens=${tokenized.tokens.length} ` +
      `inference_tokens=${inferencePromptTokens} prompt_sha256=${artifact.rendered_prompt_sha256} ` +
      `token_ids_sha256=${artifact.input_token_ids_sha256}\n`,
    );
    captured += 1;
  }
  if (captured === 0) throw new Error("no captured /v1/chat/completions requests found");
}

main().catch((error) => {
  process.stderr.write(`ADQ rendered-input capture failed: ${error.stack || error.message}\n`);
  process.exit(1);
});
