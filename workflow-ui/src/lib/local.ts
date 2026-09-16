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
import { dump, load } from "js-yaml";
import { agenticDir, resolvePythonCommand } from "./workflow";

const wynikPath = path.join(agenticDir, "wyniki", "_ostatni_lokalny.json");
const overridesPath = path.join(agenticDir, "overrides.yaml");
const configPath = path.join(agenticDir, "config.yaml");

export type LocalPart = {
  pn: string;
  description: string;
  cluster: string;
  source: "override" | "model" | "discovered";
  confidence: number;
  hs6: string;
  bu: string;
  n_rows: number;
  type_phrase: string;
};

export type LocalCluster = {
  name: string;
  n_pn: number;
  n_rows: number;
  mean_confidence: number;
  sources: Record<string, number>;
  proposed: boolean;
};

export type LocalResult = {
  mode: "discover" | "classify";
  generated_at: string;
  directory: string;
  encoder: string;
  naming: string;
  taxonomy: string[];
  metrics: Record<string, number>;
  simulation: Record<string, number>;
  clusters: LocalCluster[];
  parts: LocalPart[];
  review_label: string;
};

export type LocalRunOptions = {
  mode: "discover" | "classify";
  grouping: "cloud" | "local";
  physicsWeight: number;
  accuracyTarget: number;
  discoveryThreshold: number;
  limit: number | null;
};

/**
 * Ustawienia zaszyte na stale - w UI nie ma ich po co pokazywac.
 * tfidf wygral pomiary (ARI 0.954 vs 0.937 dla tfidf+supcon), a nazywanie
 * przez LLM kosztuje 1 zapytanie na caly bieg i samo spada na c-TF-IDF,
 * gdy klucz jest nieustawiony albo endpoint nie odpowiada.
 * Pozostale backendy zostaja dostepne z CLI: run_local.py --encoder ...
 */
const STALY_ENKODER = "tfidf";
const STALE_NAZYWANIE = "llm";

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

/**
 * Ostatni wynik toru lokalnego; null gdy nic nie policzono ALBO gdy plik jest
 * w niezgodnym formacie.
 *
 * Katalog wyniki/ jest w .gitignore, wiec po pullu nowego kodu potrafi lezec
 * tam plik zapisany przez starsza wersje (miala polskie klucze: czesci,
 * klastry, metryki). Bez tej walidacji komponent dostawal obiekt bez pola
 * parts i cala strona sie wywracala. Traktujemy taki plik jak brak wyniku -
 * kolejne uruchomienie klastrowania go nadpisze.
 */
export async function readLocalResult(): Promise<LocalResult | null> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(await fs.readFile(wynikPath, "utf8"));
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const kandydat = parsed as Partial<LocalResult>;
  const zgodny =
    Array.isArray(kandydat.parts) &&
    Array.isArray(kandydat.clusters) &&
    kandydat.metrics != null &&
    typeof kandydat.metrics === "object";
  return zgodny ? (kandydat as LocalResult) : null;
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

/**
 * Zapisuje klucz Model Farm do agentic/config.yaml. Gdy pliku nie ma, zaklada
 * go na bazie config.example.yaml, zeby uzytkownik nie musial nic kopiowac
 * recznie. Klucz nigdy nie wraca do przegladarki - UI dostaje tylko flage.
 */
export async function saveApiKey(apiKey: string): Promise<void> {
  const klucz = apiKey.trim();
  if (!klucz) throw new Error("Empty API key.");

  let cfg: Record<string, unknown> = {};
  for (const sciezka of [configPath, path.join(agenticDir, "config.example.yaml")]) {
    try {
      const parsed = load(await fs.readFile(sciezka, "utf8"));
      if (parsed && typeof parsed === "object") {
        cfg = parsed as Record<string, unknown>;
        break;
      }
    } catch {
      // brak pliku - probujemy nastepny, a na koncu zapisujemy same minimum
    }
  }
  cfg.api_key = klucz;
  if (!cfg.provider || cfg.provider === "mock") cfg.provider = "bosch";
  await fs.writeFile(configPath, dump(cfg, { lineWidth: -1, noRefs: true }), "utf8");
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
  const args = ["run_local.py", "--tryb", options.mode];
  if (options.mode === "discover") {
    args.push(
      "--grupowanie",
      options.grouping,
      "--waga-fizyki",
      String(options.physicsWeight),
    );
  } else {
    args.push(
      "--encoder",
      STALY_ENKODER,
      "--nazywaj",
      STALE_NAZYWANIE,
      "--cel-trafnosci",
      String(options.accuracyTarget),
      "--prog-odkrywania",
      String(options.discoveryThreshold),
    );
  }
  if (options.limit) args.push("--limit", String(options.limit));
  runPython(args, `local clustering (${options.mode})`);
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
