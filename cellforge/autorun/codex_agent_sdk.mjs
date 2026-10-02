#!/usr/bin/env node
import fs from "fs";
import path from "path";
import { Codex } from "@openai/codex-sdk";

function getArg(name, fallback = "") {
  const i = process.argv.indexOf(name);
  if (i >= 0 && i + 1 < process.argv.length) return process.argv[i + 1];
  return fallback;
}

const promptFile = getArg("--prompt-file", "");
const outFile = getArg("--out-file", "codex_agent_output.md");
const cwd = getArg("--cwd", process.cwd());
const model = getArg("--model", "");

let prompt = "";
if (promptFile && fs.existsSync(promptFile)) {
  prompt = fs.readFileSync(promptFile, "utf-8");
} else {
  prompt = process.env.CODEX_AGENT_PROMPT || "";
}

if (!prompt.trim()) {
  console.error("Empty prompt for codex agent.");
  process.exit(2);
}

if (process.env.CODEX_API_KEY) {
  process.env.OPENAI_API_KEY = process.env.CODEX_API_KEY;
}
if (process.env.CODEX_BASE_URL) {
  process.env.OPENAI_BASE_URL = process.env.CODEX_BASE_URL;
}

const codex = model ? new Codex({ model }) : new Codex();
const thread = codex.startThread();
const result = await thread.run(prompt);

let text = "";
if (typeof result === "string") {
  text = result;
} else if (result?.output_text) {
  text = result.output_text;
} else if (result?.text) {
  text = result.text;
} else {
  text = JSON.stringify(result, null, 2);
}

const outPath = path.isAbsolute(outFile) ? outFile : path.join(cwd, outFile);
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, text, "utf-8");
console.log(`Wrote codex agent output: ${outPath}`);
