"use client";

import { useEffect, useState } from "react";
import {
  CheckCircle2,
  CircleDashed,
  Cloud,
  Cpu,
  HardDrive,
  Play,
  Save,
  SlidersHorizontal,
  Terminal,
  XCircle,
} from "lucide-react";
import LocalPanel from "./LocalPanel";

type Config = {
  provider: string;
  deployment: string;
  model: string;
  temperature: number;
  max_tokens: number;
  hasApiKey: boolean;
  run: { batch_size: number; concurrency: number; discovery_sample: number };
  taxonomy: {
    mode: "discover" | "seed_from_rudolf" | "fixed_list";
    allow_new: boolean;
    labels: string[];
  };
  features: { part_card: string[] };
};
type RunState = {
  state: "idle" | "running" | "finished" | "failed";
  output: string[];
};

const featureOptions = [
  "desc",
  "hs6",
  "rbname",
  "parttype",
  "material_field",
  "bu",
  "descplus",
  "hstext",
  "hierarchy",
];

function Tabs({
  tab,
  setTab,
}: {
  tab: "chmura" | "lokalne";
  setTab: (t: "chmura" | "lokalne") => void;
}) {
  return (
    <nav className="tabs">
      <button
        className={tab === "lokalne" ? "tab active" : "tab"}
        onClick={() => setTab("lokalne")}
      >
        <HardDrive size={16} /> Lokalne (hybryda)
      </button>
      <button
        className={tab === "chmura" ? "tab active" : "tab"}
        onClick={() => setTab("chmura")}
      >
        <Cloud size={16} /> Chmurowe (agenci LLM)
      </button>
    </nav>
  );
}

export default function Home() {
  const [config, setConfig] = useState<Config | null>(null);
  const [run, setRun] = useState<RunState>({ state: "idle", output: [] });
  const [notice, setNotice] = useState("");
  const [dataset, setDataset] = useState<"to_cluster" | "cla">("to_cluster");
  const [limit, setLimit] = useState(60);
  const [tab, setTab] = useState<"chmura" | "lokalne">("lokalne");

  useEffect(() => {
    let cancelled = false;
    void fetch("/api/workflow")
      .then(async (response) => {
        // 500 = brak agentic/config.yaml. Body jest wtedy {error: ...}, ktore
        // jest truthy - wpisane do config wywracalo cala strone. Tor lokalny
        // nie potrzebuje tej konfiguracji, wiec zostawiamy config=null.
        if (!response.ok) throw new Error(String((await response.json())?.error));
        return response.json();
      })
      .then((data) => {
        if (!cancelled) setConfig(data);
      })
      .catch(() => {
        if (!cancelled)
          setNotice(
            "Brak agentic/config.yaml - tor chmurowy nieaktywny. " +
              "Skopiuj config.example.yaml, aby go wlaczyc. Tor lokalny dziala bez niego.",
          );
      });
    return () => {
      cancelled = true;
    };
  }, []);
  useEffect(() => {
    if (run.state !== "running") return;
    const timer = window.setInterval(
      async () => setRun(await (await fetch("/api/run")).json()),
      1200,
    );
    return () => window.clearInterval(timer);
  }, [run.state]);

  const save = async () => {
    if (!config) return;
    const response = await fetch("/api/workflow", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    if (!response.ok) return setNotice("Save failed.");
    setConfig(await response.json());
    setNotice("Configuration saved.");
  };
  const launch = async (checkOnly = false) => {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, limit, checkOnly }),
    });
    const body = await response.json();
    if (!response.ok)
      return setNotice(body.error || "Unable to start the workflow.");
    setRun(body);
    setNotice(checkOnly ? "Connection test started." : "Workflow started.");
  };
  const update = <K extends keyof Config>(key: K, value: Config[K]) =>
    config && setConfig({ ...config, [key]: value });
  const toggleFeature = (name: string) =>
    config &&
    update("features", {
      part_card: config.features.part_card.includes(name)
        ? config.features.part_card.filter((item) => item !== name)
        : [...config.features.part_card, name],
    });

  if (!config)
    return (
      <main className="shell">
        <Tabs tab={tab} setTab={setTab} />
        {notice && <div className="notice">{notice}</div>}
        {tab === "lokalne" ? (
          <LocalPanel />
        ) : (
          <p className="loading">
            <CircleDashed className="spin" /> Wczytywanie konfiguracji chmurowej…
            <br />
            <small>
              Tor chmurowy wymaga agentic/config.yaml (skopiuj z
              config.example.yaml). Tor lokalny działa bez niego.
            </small>
          </p>
        )}
      </main>
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
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <Cpu size={22} />
          <span>CLA</span>
          <strong>Agent Studio</strong>
        </div>
        <div className="status">
          <i className={config.hasApiKey ? "live" : "missing"} />
          {config.hasApiKey ? "Model Farm configured" : "API key missing"}
        </div>
      </header>
      <section className="title-row">
        <div>
          <p className="eyebrow">WORKFLOW CONTROL</p>
          <h1>Klastrowanie części</h1>
          <p className="subtitle">
            {tab === "lokalne"
              ? "Tor lokalny: bez chmury, bez tokenów. Przypnij część do klastra, a model się na tym nauczy."
              : "Tor chmurowy: taksonomia, kontekst danych i ustawienia agentów."}
          </p>
        </div>
        {tab === "chmura" && (
          <button className="primary" onClick={save}>
            <Save size={17} /> Zapisz zmiany
          </button>
        )}
      </section>
      <Tabs tab={tab} setTab={setTab} />
      {tab === "lokalne" && <LocalPanel />}
      {notice && tab === "chmura" && <div className="notice">{notice}</div>}
      <div className="workspace" hidden={tab !== "chmura"}>
        <section className="settings">
          <div className="section-head">
            <SlidersHorizontal size={18} />
            <h2>Configuration</h2>
          </div>
          <label>
            Provider
            <select
              value={config.provider}
              onChange={(event) => update("provider", event.target.value)}
            >
              <option value="bosch">Bosch Model Farm</option>
              <option value="mock">Mock offline</option>
            </select>
          </label>
          <div className="two-cols">
            <label>
              Deployment
              <input
                value={config.deployment}
                onChange={(event) => update("deployment", event.target.value)}
              />
            </label>
            <label>
              Model
              <input
                value={config.model}
                onChange={(event) => update("model", event.target.value)}
              />
            </label>
          </div>
          <div className="two-cols">
            <label>
              Temperature
              <input
                type="number"
                step="0.1"
                min="0"
                max="2"
                value={config.temperature}
                onChange={(event) =>
                  update("temperature", Number(event.target.value))
                }
              />
            </label>
            <label>
              Max tokens
              <input
                type="number"
                min="100"
                value={config.max_tokens}
                onChange={(event) =>
                  update("max_tokens", Number(event.target.value))
                }
              />
            </label>
          </div>
          <hr />
          <h3>Taxonomy</h3>
          <label>
            Mode
            <select
              value={config.taxonomy.mode}
              onChange={(event) =>
                update("taxonomy", {
                  ...config.taxonomy,
                  mode: event.target.value as Config["taxonomy"]["mode"],
                })
              }
            >
              <option value="discover">Discover and consolidate</option>
              <option value="seed_from_rudolf">Seed from Rudolf</option>
              <option value="fixed_list">Fixed list</option>
            </select>
          </label>
          <label className="switch-row">
            <input
              type="checkbox"
              checked={config.taxonomy.allow_new}
              onChange={(event) =>
                update("taxonomy", {
                  ...config.taxonomy,
                  allow_new: event.target.checked,
                })
              }
            />
            <span>Allow agents to propose new clusters</span>
          </label>
          {config.taxonomy.mode === "fixed_list" && (
            <label>
              Classes (one per line)
              <textarea
                value={config.taxonomy.labels.join("\n")}
                onChange={(event) =>
                  update("taxonomy", {
                    ...config.taxonomy,
                    labels: event.target.value
                      .split("\n")
                      .map((item) => item.trim())
                      .filter(Boolean),
                  })
                }
              />
            </label>
          )}
          <hr />
          <h3>Part card</h3>
          <div className="chips">
            {featureOptions.map((feature) => (
              <button
                key={feature}
                className={
                  config.features.part_card.includes(feature)
                    ? "chip active"
                    : "chip"
                }
                onClick={() => toggleFeature(feature)}
              >
                {feature}
              </button>
            ))}
          </div>
        </section>
        <section className="workbench">
          <div className="flow">
            <div className="step active">
              <b>01</b>
              <span>Discovery</span>
              <small>Sample: {config.run.discovery_sample} PNs</small>
            </div>
            <div className="connector" />
            <div className="step">
              <b>02</b>
              <span>Consolidate</span>
              <small>One taxonomy</small>
            </div>
            <div className="connector" />
            <div className="step">
              <b>03</b>
              <span>Classify</span>
              <small>Batches of {config.run.batch_size}</small>
            </div>
          </div>
          <div className="run-card">
            <div>
              <p className="eyebrow">RUN CONSOLE</p>
              <h2>Run a controlled test</h2>
            </div>
            <div className="run-inputs">
              <label>
                Dataset
                <select
                  value={dataset}
                  onChange={(event) =>
                    setDataset(event.target.value as "to_cluster" | "cla")
                  }
                >
                  <option value="to_cluster">to_cluster</option>
                  <option value="cla">cla</option>
                </select>
              </label>
              <label>
                PN limit
                <input
                  type="number"
                  min="1"
                  value={limit}
                  onChange={(event) => setLimit(Number(event.target.value))}
                />
              </label>
            </div>
            <div className="actions">
              <button
                className="secondary"
                disabled={run.state === "running"}
                onClick={() => launch(true)}
              >
                Test connection
              </button>
              <button
                className="primary"
                disabled={run.state === "running"}
                onClick={() => launch()}
              >
                <Play size={17} fill="currentColor" /> Run sample
              </button>
            </div>
          </div>
          <div className="console">
            <div className="console-head">
              <span>
                <Terminal size={16} /> Execution log
              </span>
              <span className={`run-status ${run.state}`}>
                {statusIcon} {run.state}
              </span>
            </div>
            <pre>
              {run.output.length
                ? run.output.join("\n")
                : "Ready to run. Pipeline results remain in agentic/wyniki/."}
            </pre>
          </div>
        </section>
      </div>
    </main>
  );
}
