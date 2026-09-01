import { spawn } from "child_process";
import fs from "fs/promises";
import fsSync from "fs";
import path from "path";
import { load, dump } from "js-yaml";

const agenticDir = path.resolve(process.cwd(), "..", "agentic");
const configPath = path.join(agenticDir, "config.yaml");
const anacondaPython = "C:\\Program Files\\Anaconda3\\python.exe";

function resolvePythonCommand(): string {
  if (process.env.PYTHON_EXECUTABLE) return process.env.PYTHON_EXECUTABLE;
  if (process.platform === "win32" && fsSync.existsSync(anacondaPython))
    return anacondaPython;
  return "python";
}

export type WorkflowConfig = {
  provider: string;
  api_key?: string;
  base_url: string;
  api_version: string;
  deployment: string;
  model: string;
  temperature: number;
  max_tokens: number;
  request_timeout: number;
  max_retries: number;
  run: {
    limit: number | null;
    batch_size: number;
    concurrency: number;
    discovery_sample: number;
  };
  taxonomy: {
    mode: "discover" | "seed_from_rudolf" | "fixed_list";
    allow_new: boolean;
    labels: string[];
  };
  features: { use_capri: boolean; part_card: string[] };
};

export type SafeWorkflowConfig = Omit<WorkflowConfig, "api_key"> & {
  hasApiKey: boolean;
};

type RunState = {
  state: "idle" | "running" | "finished" | "failed";
  output: string[];
  startedAt?: string;
  finishedAt?: string;
  exitCode?: number | null;
};

let runState: RunState = { state: "idle", output: [] };

export async function readWorkflowConfig(): Promise<WorkflowConfig> {
  return load(await fs.readFile(configPath, "utf8")) as WorkflowConfig;
}

export async function readSafeWorkflowConfig(): Promise<SafeWorkflowConfig> {
  const config = await readWorkflowConfig();
  const { api_key, ...safeConfig } = config;
  return { ...safeConfig, hasApiKey: Boolean(api_key) };
}

export async function saveWorkflowConfig(
  input: WorkflowConfig,
): Promise<SafeWorkflowConfig> {
  const previous = await readWorkflowConfig();
  const updatable = { ...input } as WorkflowConfig & { hasApiKey?: boolean };
  delete updatable.api_key;
  delete updatable.hasApiKey;
  const next = {
    ...previous,
    ...updatable,
    run: { ...previous.run, ...updatable.run },
    taxonomy: { ...previous.taxonomy, ...updatable.taxonomy },
    features: { ...previous.features, ...updatable.features },
    api_key: input.api_key?.trim() || previous.api_key || "",
  };
  await fs.writeFile(
    configPath,
    dump(next, { lineWidth: -1, noRefs: true }),
    "utf8",
  );
  return readSafeWorkflowConfig();
}

export function getRunState(): RunState {
  return runState;
}

export function startWorkflow(options: {
  dataset: "to_cluster" | "cla";
  limit: number;
  checkOnly: boolean;
}) {
  if (runState.state === "running")
    throw new Error("The workflow is already running.");
  const args = ["run_clustering.py", "--dataset", options.dataset];
  if (options.checkOnly) args.push("--check");
  else args.push("--limit", String(options.limit));
  const pythonCommand = resolvePythonCommand();
  runState = {
    state: "running",
    output: ["> " + pythonCommand + " " + args.join(" ")],
    startedAt: new Date().toISOString(),
  };
  const child = spawn(pythonCommand, args, {
    cwd: agenticDir,
    windowsHide: true,
  });
  const write = (chunk: Buffer) => {
    runState.output.push(...chunk.toString().split(/\r?\n/).filter(Boolean));
    runState.output = runState.output.slice(-300);
  };
  child.stdout.on("data", write);
  child.stderr.on("data", write);
  child.on("error", (error) => {
    runState = {
      ...runState,
      state: "failed",
      output: [...runState.output, error.message],
      finishedAt: new Date().toISOString(),
    };
  });
  child.on("close", (exitCode) => {
    runState = {
      ...runState,
      state: exitCode === 0 ? "finished" : "failed",
      exitCode,
      finishedAt: new Date().toISOString(),
    };
  });
}
