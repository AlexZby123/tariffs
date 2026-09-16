/**
 * Tor LOKALNY (agentic/run_local.py) - uruchamianie, odczyt wyniku i overrides.
 *
 * UI nie parsuje CSV: run_local.py zapisuje wynik pod stala sciezka
 * agentic/wyniki/_ostatni_lokalny.json, a overrides zapisujemy przez samo
 * run_local.py --override, zeby format pliku mial jedno zrodlo prawdy.
 */
import { spawn } from "child_process";
import fs from "fs/promises";
import path from "path";
import { load } from "js-yaml";
import { agenticDir, resolvePythonCommand } from "./workflow";

const wynikPath = path.join(agenticDir, "wyniki", "_ostatni_lokalny.json");
const overridesPath = path.join(agenticDir, "overrides.yaml");
const configPath = path.join(agenticDir, "config.yaml");

export type LocalPart = {
  pn: string;
  opis: string;
  klaster: string;
  zrodlo: "override" | "model" | "odkryty";
  pewnosc: number;
  hs6: string;
  bu: string;
  n_wierszy: number;
};

export type LocalCluster = {
  nazwa: string;
  n_pn: number;
  n_wierszy: number;
  pewnosc_srednia: number;
  zrodla: Record<string, number>;
  propozycja: boolean;
};

export type LocalResult = {
  wygenerowano: string;
  katalog: string;
  enkoder: string;
  nazywanie: string;
  taksonomia: string[];
  metryki: Record<string, number>;
  symulacja: Record<string, number>;
  klastry: LocalCluster[];
  czesci: LocalPart[];
  etykieta_przegladu: string;
};

export type LocalRunOptions = {
  encoder: string;
  nazywaj: "ctfidf" | "llm" | "brak";
  celTrafnosci: number;
  progOdkrywania: number;
  limit: number | null;
};

type LocalRunState = {
  state: "idle" | "running" | "finished" | "failed";
  output: string[];
  startedAt?: string;
  finishedAt?: string;
  exitCode?: number | null;
};

let localRunState: LocalRunState = { state: "idle", output: [] };

export function getLocalRunState(): LocalRunState {
  return localRunState;
}

/** Ostatni wynik toru lokalnego; null gdy jeszcze nic nie policzono. */
export async function readLocalResult(): Promise<LocalResult | null> {
  try {
    return JSON.parse(await fs.readFile(wynikPath, "utf8")) as LocalResult;
  } catch {
    return null;
  }
}

/**
 * Czy nazywanie grup przez LLM jest gotowe do uzycia: istnieje config.yaml
 * z realnym tokenem (albo provider mock, ktory dziala offline).
 */
export async function isLlmReady(): Promise<boolean> {
  try {
    const cfg = load(await fs.readFile(configPath, "utf8")) as {
      provider?: string;
      api_key?: string;
    };
    if (!cfg) return false;
    if (cfg.provider === "mock") return true;
    const key = String(cfg.api_key ?? "");
    return key.length > 0 && !key.includes("PASTE");
  } catch {
    return false;
  }
}

/** Slownik eksperta PN -> klaster (warstwa 0). */
export async function readOverrides(): Promise<Record<string, string>> {
  try {
    const parsed = load(await fs.readFile(overridesPath, "utf8"));
    if (!parsed || typeof parsed !== "object") return {};
    return Object.fromEntries(
      Object.entries(parsed as Record<string, unknown>).map(([k, v]) => [
        String(k).trim(),
        String(v).trim(),
      ]),
    );
  } catch {
    return {};
  }
}

function runPython(args: string[], label: string) {
  if (localRunState.state === "running")
    throw new Error("Tor lokalny jest juz uruchomiony.");
  const python = resolvePythonCommand();
  localRunState = {
    state: "running",
    output: [`> ${python} ${args.join(" ")}`, `# ${label}`],
    startedAt: new Date().toISOString(),
  };
  const child = spawn(python, args, { cwd: agenticDir, windowsHide: true });
  const write = (chunk: Buffer) => {
    localRunState.output.push(
      ...chunk.toString().split(/\r?\n/).filter(Boolean),
    );
    localRunState.output = localRunState.output.slice(-400);
  };
  child.stdout.on("data", write);
  child.stderr.on("data", write);
  child.on("error", (error) => {
    localRunState = {
      ...localRunState,
      state: "failed",
      output: [...localRunState.output, error.message],
      finishedAt: new Date().toISOString(),
    };
  });
  child.on("close", (exitCode) => {
    localRunState = {
      ...localRunState,
      state: exitCode === 0 ? "finished" : "failed",
      exitCode,
      finishedAt: new Date().toISOString(),
    };
  });
}

export function startLocalRun(options: LocalRunOptions) {
  const args = [
    "run_local.py",
    "--encoder",
    options.encoder,
    "--nazywaj",
    options.nazywaj,
    "--cel-trafnosci",
    String(options.celTrafnosci),
    "--prog-odkrywania",
    String(options.progOdkrywania),
  ];
  if (options.limit) args.push("--limit", String(options.limit));
  runPython(args, "klastrowanie lokalne");
}

/**
 * WARSTWA 3: zapisuje korekty eksperta i - gdy poproszono - od razu przelicza
 * model, zeby nauczyl sie na nich. Zapis idzie przez run_local.py --override,
 * ktory konczy sie natychmiast (nie wczytuje danych).
 */
export async function saveOverridesAndRetrain(
  przypisania: Record<string, string>,
  doucz: boolean,
  runOptions: LocalRunOptions,
): Promise<{ zapisane: number }> {
  const wpisy = Object.entries(przypisania).filter(
    ([pn, klaster]) => pn.trim() && klaster.trim(),
  );
  if (!wpisy.length) throw new Error("Brak korekt do zapisania.");

  const args = ["run_local.py"];
  for (const [pn, klaster] of wpisy)
    args.push("--override", `${pn.trim()}=${klaster.trim()}`);

  await new Promise<void>((resolve, reject) => {
    const child = spawn(resolvePythonCommand(), args, {
      cwd: agenticDir,
      windowsHide: true,
    });
    let stderr = "";
    child.stderr.on("data", (c: Buffer) => (stderr += c.toString()));
    child.on("error", reject);
    child.on("close", (code) =>
      code === 0
        ? resolve()
        : reject(new Error(stderr.trim() || `zapis overrides zwrocil ${code}`)),
    );
  });

  if (doucz) startLocalRun(runOptions);
  return { zapisane: wpisy.length };
}
