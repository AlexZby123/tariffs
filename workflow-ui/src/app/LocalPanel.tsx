"use client";

import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
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

const pct = (x: number) => `${((x ?? 0) * 100).toFixed(1)}%`;

export default function LocalPanel() {
  const [result, setResult] = useState<Result | null>(null);
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  const [run, setRun] = useState<RunState>({ state: "idle", output: [] });
  const [notice, setNotice] = useState("");

  const [llmReady, setLlmReady] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [editingKey, setEditingKey] = useState(false);

  const [advanced, setAdvanced] = useState(false);
  const [accuracyTarget, setAccuracyTarget] = useState(0.999);
  const [discoveryThreshold, setDiscoveryThreshold] = useState(0.6);
  const [limit, setLimit] = useState<number | null>(null);

  const [query, setQuery] = useState("");
  const [clusterFilter, setClusterFilter] = useState<string | null>(null);
  const [reviewOnly, setReviewOnly] = useState(false);
  const [pending, setPending] = useState<Record<string, string>>({});

  const runOptions = { accuracyTarget, discoveryThreshold, limit };

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
    return result.parts
      .filter((p) => !clusterFilter || p.cluster === clusterFilter)
      .filter(
        (p) =>
          !reviewOnly ||
          p.source === "discovered" ||
          p.cluster === result.review_label,
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

  const reviewCount = useMemo(
    () => (result?.parts ?? []).filter((p) => p.source === "discovered").length,
    [result],
  );

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
          <h2>Hybrid clustering</h2>
          <p className="hint">
            Known part types are assigned by a classifier that reaches 95.6%
            accuracy out-of-fold. Parts it is not sure about are grouped
            separately and named by the cloud model, then land in your review
            queue.
          </p>
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
                <option value={0.98}>Smallest queue — 5% of parts</option>
                <option value={0.99}>Small queue — 9%</option>
                <option value={0.995}>Balanced — 12%</option>
                <option value={0.999}>Safest — 17% (default)</option>
              </select>
              <small className="hint">
                How accurate the classifier must be on what it assigns by
                itself. Safer means fewer silent mistakes and better detection
                of genuinely new part types, at the cost of more parts landing
                in your queue. At the default it made no mistakes at all on the
                83% it accepted.
              </small>
            </label>

            <label>
              Discovery threshold
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
                How far apart two parts may be and still form one discovered
                group. Affects only the review queue, never parts the classifier
                assigned. Lower (0.4) splits into more, smaller, cleaner groups;
                higher (0.8) merges into fewer, larger ones and risks mixing
                different types. Merging two clean groups costs you one click —
                untangling a wrong merge does not.
              </small>
            </label>

            <label>
              Part-number limit
              <input
                type="number"
                min="1"
                placeholder="empty = all 1025"
                value={limit ?? ""}
                onChange={(event) =>
                  setLimit(event.target.value ? Number(event.target.value) : null)
                }
              />
              <small className="hint">
                For quick tests only. Takes the first N part numbers in file
                order, and the file is grouped by product family — so a small
                sample is badly skewed and row-level evaluation is skipped.
                Leave empty to judge quality.
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
          <div className="metric">
            <b>{pct(result.metrics.oof_accuracy)}</b>
            <span>accuracy (out-of-fold)</span>
          </div>
          <div className="metric">
            <b>{result.metrics.ari?.toFixed(3)}</b>
            <span>ARI — cloud agents get 0.885</span>
          </div>
          <div className="metric">
            <b>{pct(result.metrics.auto_assigned_share)}</b>
            <span>assigned automatically</span>
          </div>
          <div className="metric">
            <b>{pct(result.metrics.auto_assigned_accuracy)}</b>
            <span>accuracy of those</span>
          </div>
          <div className="metric warn">
            <b>{reviewCount}</b>
            <span>parts to review</span>
          </div>
          <div className="metric">
            <b>{result.clusters.length}</b>
            <span>clusters ({result.metrics.n_classes} known)</span>
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
              all ({result.parts.length} PN)
            </button>
            <div className="cluster-scroll">
              {result.clusters.map((c) => (
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
                    <th>Cluster</th>
                    <th>Assigned by</th>
                    <th>Margin</th>
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
                        <td className="desc" title={p.description}>
                          {p.description}
                        </td>
                        <td>
                          {p.cluster}
                          {pinned && (
                            <span className="badge pinned">
                              <Pin size={11} /> pinned
                            </span>
                          )}
                        </td>
                        <td>
                          <span className={`badge src-${p.source}`}>
                            {p.source}
                          </span>
                        </td>
                        <td className="mono">{p.confidence.toFixed(2)}</td>
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
