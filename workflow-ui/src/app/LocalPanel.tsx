"use client";

import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  FileUp,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleDashed,
  Cpu,
  KeyRound,
  Pin,
  Play,
  Search,
  Sparkles,
  Terminal,
  Trash2,
  XCircle,
} from "lucide-react";

type Part = {
  pn: string;
  description: string;
  cluster: string;
  source: "override" | "model" | "discovered";
  confidence: number;
  hs6: string;
  bu: string;
  n_rows: number;
  type_phrase: string;
  needs_review: boolean;
  description_raw: string;
  weight_g: number | null;
};
type Cluster = {
  name: string;
  n_pn: number;
  n_rows: number;
  mean_confidence: number;
  sources: Record<string, number>;
  proposed: boolean;
};
type Result = {
  mode: "discover" | "classify";
  generated_at: string;
  encoder: string;
  naming: string;
  taxonomy: string[];
  metrics: Record<string, number>;
  clusters: Cluster[];
  parts: Part[];
  review_label: string;
};
type RunState = {
  state: "idle" | "running" | "finished" | "failed";
  output: string[];
};

const pct = (x: number | undefined) => `${((x ?? 0) * 100).toFixed(1)}%`;

/** Waga na sztuke w jednostce, ktora czyta sie bez liczenia zer. */
function waga(g: number | null): string {
  if (g == null || !Number.isFinite(g)) return "—";
  if (g >= 1000) return `${(g / 1000).toFixed(g >= 10000 ? 0 : 1)} kg`;
  if (g >= 10) return `${Math.round(g)} g`;
  if (g >= 1) return `${g.toFixed(1)} g`;
  return `${g.toFixed(2)} g`;
}

/**
 * Margines to surowy odstep miedzy najlepsza a druga klasa - liczba sama w
 * sobie nic nie mowi. Pokazujemy ja jako pasek wzgledem progu, przy ktorym
 * model przestaje ufac sobie samemu: pelny pasek = pewne, krotki = na granicy.
 */
function pewnoscProc(margines: number, prog: number): number {
  if (!Number.isFinite(margines) || !prog) return 4;
  // minimum 4%, zeby pasek przy marginesie bliskim zera czytal sie jako
  // "bardzo malo", a nie jako brakujaca wartosc
  return Math.max(4, Math.min(100, (margines / (prog * 2)) * 100));
}

export default function LocalPanel() {
  const [result, setResult] = useState<Result | null>(null);
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  const [run, setRun] = useState<RunState>({ state: "idle", output: [] });
  const [notice, setNotice] = useState("");

  const [llmReady, setLlmReady] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [editingKey, setEditingKey] = useState(false);

  // Jeden proces, bez wyboru trybu: model uczy sie na czesciach juz opisanych,
  // a to, czego nie jest pewny albo czego nie zna, przejmuje warstwa odkrywcza
  // i nazywa LLM. Tryb "discover" (budowanie od zera, bez zadnych etykiet)
  // zostaje dostepny z CLI - to scenariusz zimnego startu, nie codzienna praca.
  const mode = "classify" as const;
  const grouping = "cloud" as const;
  const physicsWeight = 0.3;
  const [inputFile, setInputFile] = useState<{ path: string; name: string } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [accuracyTarget, setAccuracyTarget] = useState(0.999);
  const [discoveryThreshold, setDiscoveryThreshold] = useState(0.6);
  const [limit, setLimit] = useState<number | null>(null);

  const [query, setQuery] = useState("");
  const [clusterFilter, setClusterFilter] = useState<string | null>(null);
  const [reviewOnly, setReviewOnly] = useState(false);
  const [pending, setPending] = useState<Record<string, string>>({});

  const runOptions = {
    inputFile: inputFile?.path ?? null,
    mode,
    grouping,
    physicsWeight,
    accuracyTarget,
    discoveryThreshold,
    limit,
  };

  const load = async () => {
    const data = await (await fetch("/api/local")).json();
    setResult(data.result);
    setOverrides(data.overrides ?? {});
    setLlmReady(Boolean(data.llmReady));
  };

  useEffect(() => {
    void load();
  }, []);

  useEffect(() => {
    if (run.state !== "running") return;
    const timer = window.setInterval(async () => {
      const state: RunState = await (await fetch("/api/local/run")).json();
      setRun(state);
      if (state.state !== "running") void load();
    }, 1500);
    return () => window.clearInterval(timer);
  }, [run.state]);

  const start = async () => {
    const response = await fetch("/api/local/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(runOptions),
    });
    const body = await response.json();
    if (!response.ok) return setNotice(body.error ?? "Could not start the run.");
    setRun(body);
    setNotice("Clustering started.");
  };

  const wgrajPlik = async (pliki: FileList | null) => {
    const plik = pliki?.[0];
    if (!plik) return;
    setUploading(true);
    const dane = new FormData();
    dane.append("file", plik);
    const response = await fetch("/api/local/upload", { method: "POST", body: dane });
    const body = await response.json();
    setUploading(false);
    if (!response.ok) return setNotice(body.error ?? "Upload failed.");
    setInputFile({ path: body.path, name: body.name });
    setNotice(`${body.name} uploaded. Run the clustering to assign its parts.`);
  };

  const storeApiKey = async () => {
    const response = await fetch("/api/local/apikey", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ apiKey }),
    });
    const body = await response.json();
    if (!response.ok) return setNotice(body.error ?? "Could not save the key.");
    setLlmReady(Boolean(body.llmReady));
    setApiKey("");
    setEditingKey(false);
    setNotice("API key saved to agentic/config.yaml.");
  };

  const saveCorrections = async (retrain: boolean) => {
    const response = await fetch("/api/local/override", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ przypisania: pending, doucz: retrain, runOptions }),
    });
    const body = await response.json();
    if (!response.ok) return setNotice(body.error ?? "Saving failed.");
    setOverrides(body.overrides ?? {});
    setPending({});
    setNotice(
      retrain
        ? `Saved ${body.zapisane} correction(s). The model is retraining on them now.`
        : `Saved ${body.zapisane} correction(s). Retrain to apply them.`,
    );
    if (retrain) setRun({ state: "running", output: [] });
  };

  const clusterNames = useMemo(
    () =>
      Array.from(
        new Set([
          ...(result?.taxonomy ?? []),
          ...(result?.clusters ?? []).map((c) => c.name),
        ]),
      ).sort(),
    [result],
  );

  const visible = useMemo(() => {
    if (!result) return [];
    const needle = query.trim().toLowerCase();
    return (result.parts ?? [])
      .filter((p) => !clusterFilter || p.cluster === clusterFilter)
      .filter(
        (p) =>
          !reviewOnly ||
          p.needs_review ||
          p.cluster === result.review_label ||
          (result.mode === "classify" && p.source === "discovered"),
      )
      .filter(
        (p) =>
          !needle ||
          p.pn.toLowerCase().includes(needle) ||
          p.description.toLowerCase().includes(needle) ||
          p.cluster.toLowerCase().includes(needle),
      )
      .sort((a, b) => a.confidence - b.confidence)
      .slice(0, 300);
  }, [result, query, clusterFilter, reviewOnly]);

  /** Typy zaproponowane przez system, spoza taksonomii, ktorej sie nauczyl. */
  const newTypeCount = useMemo(
    () => (result?.clusters ?? []).filter((c) => c.proposed).length,
    [result],
  );

  const reviewCount = useMemo(() => {
    if (!result) return 0;
    const parts = result.parts ?? [];
    return result.mode === "discover"
      ? parts.filter((p) => p.needs_review).length
      : parts.filter((p) => p.source === "discovered").length;
  }, [result]);

  const statusIcon =
    run.state === "failed" ? (
      <XCircle />
    ) : run.state === "finished" ? (
      <CheckCircle2 />
    ) : (
      <CircleDashed className={run.state === "running" ? "spin" : ""} />
    );

  return (
    <div className="local">
      {notice && <div className="notice">{notice}</div>}

      <section className="run-card">
        <div>
          <p className="eyebrow">LOCAL TRACK — RUNS ON THIS MACHINE</p>
          <h2>Cluster parts</h2>
          <p className="hint">
            Parts of a type your experts already described are assigned
            automatically. Anything the model is unsure about, or a type it has
            never seen, is grouped separately, named by the cloud model and put
            in your review queue. Every correction you make there is folded back
            into training.
          </p>
        </div>

        <div className="apikey-row">
          <FileUp size={16} />
          {inputFile ? (
            <>
              <span>
                Clustering <b>{inputFile.name}</b> — the model stays trained on
                the parts your experts already described.
              </span>
              <button className="link" onClick={() => setInputFile(null)}>
                use the default dataset
              </button>
            </>
          ) : (
            <>
              <span>Using the built-in dataset.</span>
              <label className="secondary jak-przycisk">
                {uploading ? "Uploading…" : "Choose a file…"}
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls,.xlsm,.txt"
                  hidden
                  onChange={(event) => wgrajPlik(event.target.files)}
                />
              </label>
              <span className="hint">
                A .csv or .xlsx with your parts. It needs a
                <code> Product Number ACDC </code> column and at least one
                <code> Material Description </code> column — no labels required.
              </span>
            </>
          )}
        </div>

        <div className="apikey-row">
          <KeyRound size={16} />
          {llmReady && !editingKey ? (
            <>
              <span>
                <b>Cloud naming active.</b> Discovered groups get real names on
                every run — one request per run, so the cost is negligible.
              </span>
              <button className="link" onClick={() => setEditingKey(true)}>
                replace key
              </button>
            </>
          ) : (
            <>
              <label className="key-input">
                Model Farm API key
                <input
                  type="password"
                  placeholder="paste the token — saved to agentic/config.yaml"
                  value={apiKey}
                  onChange={(event) => setApiKey(event.target.value)}
                />
              </label>
              <button
                className="secondary"
                disabled={!apiKey.trim()}
                onClick={storeApiKey}
              >
                Save key
              </button>
              {!llmReady && (
                <span className="hint">
                  Without a key the run still works — new groups fall back to
                  offline keyword naming.
                </span>
              )}
            </>
          )}
        </div>

        <button
          className="disclosure"
          onClick={() => setAdvanced((value) => !value)}
        >
          {advanced ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
          Advanced settings
        </button>

        {advanced && (
          <div className="run-inputs local-inputs">
            <label>
              Review workload
              <select
                value={accuracyTarget}
                onChange={(event) =>
                  setAccuracyTarget(Number(event.target.value))
                }
              >
                <option value={0.98}>Smallest queue — about 5% of parts</option>
                <option value={0.99}>Small queue — about 9%</option>
                <option value={0.995}>Balanced — about 12%</option>
                <option value={0.999}>Safest — about 17% (default)</option>
              </select>
              <small className="hint">
                How accurate the model must be on what it assigns by itself.
                Safer means fewer silent mistakes and better detection of
                genuinely new part types, at the cost of more parts landing in
                your queue. At the default it made no mistakes at all on the
                parts it accepted.
              </small>
            </label>

            <label>
              Grouping of unknown parts
              <input
                type="number"
                step="0.05"
                min="0.1"
                max="0.95"
                value={discoveryThreshold}
                onChange={(event) =>
                  setDiscoveryThreshold(Number(event.target.value))
                }
              />
              <small className="hint">
                How far apart two unknown parts may be and still form one
                proposed type. Affects only the review queue, never parts the
                model assigned itself. Lower splits into more, smaller, cleaner
                groups; higher merges into fewer and risks mixing types. Merging
                two clean groups costs you one click — untangling a wrong merge
                does not.
              </small>
            </label>

            <label>
              Part-number limit
              <input
                type="number"
                min="1"
                placeholder="empty = all"
                value={limit ?? ""}
                onChange={(event) =>
                  setLimit(event.target.value ? Number(event.target.value) : null)
                }
              />
              <small className="hint">
                For quick tests only. Takes the first N part numbers in file
                order, and the file is grouped by product family — so a small
                sample is badly skewed. Leave empty to judge quality.
              </small>
            </label>
          </div>
        )}

        <div className="actions">
          <button
            className="primary"
            disabled={run.state === "running"}
            onClick={start}
          >
            <Play size={17} fill="currentColor" /> Run clustering
          </button>
        </div>
      </section>

      {result && (
        <section className="metrics">
          {/* Same liczniki z TEGO przebiegu. Procenty jakosci (trafnosc,
              pokrycie) pochodzily z walidacji na danych treningowych - przy
              wgranym, nowym zbiorze nie opisuja niczego, co uzytkownik przed
              chwila policzyl, wiec na ekranie byly mylace. Zostaja w logu. */}
          <div className="metric">
            <b>{(result.parts ?? []).length}</b>
            <span>parts clustered</span>
          </div>
          <div className="metric">
            <b>{(result.parts ?? []).length - reviewCount}</b>
            <span>assigned to known types</span>
          </div>
          <div className="metric warn">
            <b>{reviewCount}</b>
            <span>need your review</span>
          </div>
          <div className="metric">
            <b>{newTypeCount}</b>
            <span>new types proposed</span>
          </div>
        </section>
      )}

      {result && (
        <div className="local-grid">
          <aside className="cluster-list">
            <div className="section-head">
              <Cpu size={18} />
              <h2>Clusters</h2>
            </div>
            <button
              className={clusterFilter ? "chip" : "chip active"}
              onClick={() => setClusterFilter(null)}
            >
              all ({(result.parts ?? []).length} PN)
            </button>
            <div className="cluster-scroll">
              {(result.clusters ?? []).map((c) => (
                <button
                  key={c.name}
                  className={
                    clusterFilter === c.name ? "cluster-row active" : "cluster-row"
                  }
                  onClick={() =>
                    setClusterFilter(clusterFilter === c.name ? null : c.name)
                  }
                  title={c.proposed ? "Proposed by discovery — needs your approval" : c.name}
                >
                  <span className="cluster-name">
                    {c.proposed && <Sparkles size={13} />} {c.name}
                  </span>
                  <span className="cluster-count">{c.n_pn}</span>
                </button>
              ))}
            </div>
          </aside>

          <section className="parts">
            <div className="parts-toolbar">
              <label className="search">
                <Search size={15} />
                <input
                  placeholder="Search by part number, description or cluster…"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
              </label>
              <label className="switch-row">
                <input
                  type="checkbox"
                  checked={reviewOnly}
                  onChange={(event) => setReviewOnly(event.target.checked)}
                />
                <span>review queue only</span>
              </label>
            </div>

            <div className="parts-scroll">
              <table className="parts-table">
                <thead>
                  <tr>
                    <th>Part number</th>
                    <th>Description</th>
                    <th className="do-prawej">Weight</th>
                    <th>Cluster</th>
                    <th>Certainty</th>
                    <th>Pin permanently</th>
                  </tr>
                </thead>
                <tbody>
                  {visible.map((p) => {
                    const pinned = overrides[p.pn];
                    const staged = pending[p.pn];
                    return (
                      <tr key={p.pn} className={staged ? "pending" : ""}>
                        <td className="mono">{p.pn}</td>
                        <td className="desc" title={p.description_raw || p.description}>
                          {p.description}
                        </td>
                        <td className="mono do-prawej">{waga(p.weight_g)}</td>
                        <td>
                          {p.cluster}
                          {p.needs_review && !pinned && (
                            <span className="badge review">needs a look</span>
                          )}
                          {pinned && (
                            <span className="badge pinned">
                              <Pin size={11} /> pinned
                            </span>
                          )}
                        </td>
                        <td>
                          <span
                            className="pasek"
                            title={`margin ${p.confidence.toFixed(2)} · model stops trusting itself below ${(result.metrics?.confidence_threshold ?? 0).toFixed(2)}`}
                          >
                            <i
                              className={p.needs_review ? "slaby" : ""}
                              style={{
                                width: `${pewnoscProc(
                                  p.confidence,
                                  result.metrics?.confidence_threshold ?? 1,
                                )}%`,
                              }}
                            />
                          </span>
                        </td>
                        <td>
                          <input
                            list="cluster-options"
                            className="pin-input"
                            placeholder="type or pick…"
                            value={staged ?? ""}
                            onChange={(event) => {
                              const value = event.target.value;
                              setPending((current) => {
                                const next = { ...current };
                                if (value.trim()) next[p.pn] = value;
                                else delete next[p.pn];
                                return next;
                              });
                            }}
                          />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <datalist id="cluster-options">
                {clusterNames.map((name) => (
                  <option key={name} value={name} />
                ))}
              </datalist>
              {!visible.length && (
                <p className="empty">No parts match the current filters.</p>
              )}
            </div>

            {Object.keys(pending).length > 0 && (
              <div className="basket">
                <div className="basket-head">
                  <AlertTriangle size={16} />
                  <strong>
                    {Object.keys(pending).length} correction(s) ready to save
                  </strong>
                  <button className="link" onClick={() => setPending({})}>
                    <Trash2 size={13} /> clear
                  </button>
                </div>
                <ul>
                  {Object.entries(pending).map(([pn, cluster]) => (
                    <li key={pn}>
                      <span className="mono">{pn}</span> → <b>{cluster}</b>
                    </li>
                  ))}
                </ul>
                <div className="actions">
                  <button
                    className="secondary"
                    onClick={() => saveCorrections(false)}
                  >
                    Save only
                  </button>
                  <button
                    className="primary"
                    disabled={run.state === "running"}
                    onClick={() => saveCorrections(true)}
                  >
                    <Pin size={16} /> Save and retrain
                  </button>
                </div>
              </div>
            )}
          </section>
        </div>
      )}

      {!result && run.state !== "running" && (
        <p className="empty">
          No results yet. Run the clustering to see clusters and the review
          queue.
        </p>
      )}

      <div className="console">
        <div className="console-head">
          <span>
            <Terminal size={16} /> Local track log
          </span>
          <span className={`run-status ${run.state}`}>
            {statusIcon} {run.state}
          </span>
        </div>
        <pre>
          {run.output.length
            ? run.output.join("\n")
            : "Ready to run. Results are written to agentic/wyniki/."}
        </pre>
      </div>
    </div>
  );
}
