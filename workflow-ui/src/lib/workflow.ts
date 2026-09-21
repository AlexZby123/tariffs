import { spawn, spawnSync } from "child_process";
import fs from "fs/promises";
import path from "path";
import { load, dump } from "js-yaml";

export const agenticDir = path.resolve(process.cwd(), "..", "agentic");
const configPath = path.join(agenticDir, "config.yaml");

/** Pakiety, bez ktorych run_local.py nie wystartuje. */
const WYMAGANE_MODULY = ["pandas", "numpy", "sklearn", "scipy", "yaml"];

/** Czy ten interpreter ma komplet wymaganych pakietow (szybki, ~100 ms). */
function maWymaganePakiety(python: string): boolean {
  try {
    const wynik = spawnSync(python, ["-c", `import ${WYMAGANE_MODULY.join(", ")}`], {
      windowsHide: true,
      timeout: 20000,
    });
    return wynik.status === 0;
  } catch {
    return false;
  }
}

let zapamietany: string | null = null;

/**
 * Interpreter, ktorym uruchamiamy run_local.py.
 *
 * PYTHON_EXECUTABLE ma pierwszenstwo ZAWSZE, nawet gdy jest zepsuty - to
 * swiadomy wybor uzytkownika i lepiej o nim powiedziec niz go po cichu
 * obchodzic (od tego jest opiszProblemZPythonem).
 *
 * Bez tej zmiennej NIE bierzemy pierwszego lepszego. Wczesniej kod na Windows
 * wpadal na sztywno w Anaconde base, ktora nie ma pandas - uzytkownik dostawal
 * goly ModuleNotFoundError i nie wiedzial, ze uruchomil nie to srodowisko.
 * Teraz sprawdzamy kandydatow i bierzemy pierwszego, ktory ma komplet pakietow.
 */
export function resolvePythonCommand(): string {
  if (process.env.PYTHON_EXECUTABLE) return process.env.PYTHON_EXECUTABLE;
  if (zapamietany) return zapamietany;

  const kandydaci =
    process.platform === "win32"
      ? ["python", "py", "C:\\Program Files\\Anaconda3\\python.exe"]
      : ["/usr/local/bin/python3", "/usr/bin/python3", "python3", "python"];

  zapamietany = kandydaci.find(maWymaganePakiety) ?? kandydaci[0];
  return zapamietany;
}

/**
 * Sprawdza interpreter PRZED uruchomieniem dlugiego zadania.
 * Zwraca komunikat do pokazania uzytkownikowi albo null, gdy wszystko gra.
 */
export function opiszProblemZPythonem(python: string): string | null {
  if (maWymaganePakiety(python)) return null;
  const zmienna = process.env.PYTHON_EXECUTABLE
    ? `PYTHON_EXECUTABLE points at "${python}".`
    : `PYTHON_EXECUTABLE is not set, so "${python}" was picked automatically.`;
  return [
    `Python at "${python}" is missing some of: ${WYMAGANE_MODULY.join(", ")}.`,
    zmienna,
    "Point it at the environment that has them, for example by creating",
    "workflow-ui/.env.local with a single line:",
    "    PYTHON_EXECUTABLE=C:\\Users\\<you>\\.conda\\envs\\<env>\\python.exe",
    "then restart the server. To check an interpreter, run in agentic/:",
    '    "<path to python.exe>" sprawdz_srodowisko.py',
  ].join("\n");
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
