"use client";

import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  CircleDashed,
  Cpu,
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
  opis: string;
  klaster: string;
  zrodlo: "override" | "model" | "odkryty";
  pewnosc: number;
  hs6: string;
  bu: string;
  n_wierszy: number;
};
type Cluster = {
  nazwa: string;
  n_pn: number;
  n_wierszy: number;
  pewnosc_srednia: number;
  zrodla: Record<string, number>;
  propozycja: boolean;
};
type Result = {
  wygenerowano: string;
  enkoder: string;
  nazywanie: string;
  taksonomia: string[];
  metryki: Record<string, number>;
  klastry: Cluster[];
  czesci: Part[];
  etykieta_przegladu: string;
};
type RunState = {
  state: "idle" | "running" | "finished" | "failed";
  output: string[];
};

const encoderOptions = [
  { value: "tfidf", label: "tfidf — offline, domyślny (ARI 0.954)" },
  { value: "tfidf+supcon", label: "tfidf+supcon — sieć metryczna (PyTorch)" },
  { value: "minilm", label: "minilm — bi-encoder (HuggingFace)" },
  { value: "bge", label: "bge-small — bi-encoder (HuggingFace)" },
];

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export default function LocalPanel() {
  const [result, setResult] = useState<Result | null>(null);
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  const [run, setRun] = useState<RunState>({ state: "idle", output: [] });
  const [notice, setNotice] = useState("");

  const [encoder, setEncoder] = useState("tfidf");
  const [nazywaj, setNazywaj] = useState<"ctfidf" | "llm" | "brak">("ctfidf");
  const [llmGotowy, setLlmGotowy] = useState(false);
  const [wybranoNazywanie, setWybranoNazywanie] = useState(false);
  const [celTrafnosci, setCelTrafnosci] = useState(0.999);
  const [progOdkrywania, setProgOdkrywania] = useState(0.6);
  const [limit, setLimit] = useState<number | null>(null);

  const [szukaj, setSzukaj] = useState("");
  const [filtrKlaster, setFiltrKlaster] = useState<string | null>(null);
  const [tylkoPrzeglad, setTylkoPrzeglad] = useState(false);
  const [koszyk, setKoszyk] = useState<Record<string, string>>({});

  const runOptions = { encoder, nazywaj, celTrafnosci, progOdkrywania, limit };

  const wczytaj = async () => {
    const data = await (await fetch("/api/local")).json();
    setResult(data.wynik);
    setOverrides(data.overrides ?? {});
    setLlmGotowy(Boolean(data.llmGotowy));
    // token skonfigurowany -> nazywanie przez LLM ma sens jako domyslne,
    // ale nie nadpisujemy wyboru, ktorego uzytkownik juz dokonal
    if (data.llmGotowy && !wybranoNazywanie) setNazywaj("llm");
  };

  useEffect(() => {
    void wczytaj();
  }, []);

  // odpytuj stan biegu; po zakonczeniu przeladuj wynik
  useEffect(() => {
    if (run.state !== "running") return;
    const timer = window.setInterval(async () => {
      const state: RunState = await (await fetch("/api/local/run")).json();
      setRun(state);
      if (state.state !== "running") void wczytaj();
    }, 1500);
    return () => window.clearInterval(timer);
  }, [run.state]);

  const uruchom = async () => {
    const response = await fetch("/api/local/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(runOptions),
    });
    const body = await response.json();
    if (!response.ok) return setNotice(body.error ?? "Nie udało się uruchomić.");
    setRun(body);
    setNotice("Klastrowanie lokalne uruchomione.");
  };

  const zapiszKorekty = async (doucz: boolean) => {
    const response = await fetch("/api/local/override", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ przypisania: koszyk, doucz, runOptions }),
    });
    const body = await response.json();
    if (!response.ok) return setNotice(body.error ?? "Zapis nie powiódł się.");
    setOverrides(body.overrides ?? {});
    setKoszyk({});
    setNotice(
      doucz
        ? `Zapisano ${body.zapisane} korekt. Model uczy się na nich teraz.`
        : `Zapisano ${body.zapisane} korekt. Doucz model, aby je uwzględnić.`,
    );
    if (doucz) setRun({ state: "running", output: [] });
  };

  const nazwyKlastrow = useMemo(
    () =>
      Array.from(
        new Set([
          ...(result?.taksonomia ?? []),
          ...(result?.klastry ?? []).map((c) => c.nazwa),
        ]),
      ).sort(),
    [result],
  );

  const widoczne = useMemo(() => {
    if (!result) return [];
    const fraza = szukaj.trim().toLowerCase();
    return result.czesci
      .filter((p) => !filtrKlaster || p.klaster === filtrKlaster)
      .filter(
        (p) =>
          !tylkoPrzeglad ||
          p.zrodlo === "odkryty" ||
          p.klaster === result.etykieta_przegladu,
      )
      .filter(
        (p) =>
          !fraza ||
          p.pn.toLowerCase().includes(fraza) ||
          p.opis.toLowerCase().includes(fraza) ||
          p.klaster.toLowerCase().includes(fraza),
      )
      .sort((a, b) => a.pewnosc - b.pewnosc)
      .slice(0, 300);
  }, [result, szukaj, filtrKlaster, tylkoPrzeglad]);

  const doPrzegladu = useMemo(
    () => (result?.czesci ?? []).filter((p) => p.zrodlo === "odkryty").length,
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
          <p className="eyebrow">TOR LOKALNY — BEZ CHMURY</p>
          <h2>Klastrowanie hybrydowe</h2>
        </div>
        <div className="run-inputs local-inputs">
          <label>
            Enkoder
            <select
              value={encoder}
              onChange={(event) => setEncoder(event.target.value)}
            >
              {encoderOptions.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Nazywanie nowych grup
            <select
              value={nazywaj}
              onChange={(event) => {
                setWybranoNazywanie(true);
                setNazywaj(event.target.value as typeof nazywaj);
              }}
            >
              <option value="ctfidf">c-TF-IDF — offline, 0 zł</option>
              <option value="llm" disabled={!llmGotowy}>
                LLM — 1 zapytanie na wszystkie grupy
                {llmGotowy ? "" : " (brak config.yaml)"}
              </option>
              <option value="brak">bez nazw (NOWY_1, NOWY_2…)</option>
            </select>
          </label>
          <label>
            Cel trafności warstwy 1
            <select
              value={celTrafnosci}
              onChange={(event) => setCelTrafnosci(Number(event.target.value))}
            >
              <option value={0.98}>0.98 — maks. automatyzacja</option>
              <option value={0.99}>0.99</option>
              <option value={0.995}>0.995</option>
              <option value={0.999}>0.999 — maks. wykrywanie nowości</option>
            </select>
          </label>
          <label>
            Próg odkrywania
            <input
              type="number"
              step="0.05"
              min="0.1"
              max="0.95"
              value={progOdkrywania}
              onChange={(event) =>
                setProgOdkrywania(Number(event.target.value))
              }
            />
          </label>
          <label>
            Limit PN (puste = całość)
            <input
              type="number"
              min="1"
              value={limit ?? ""}
              onChange={(event) =>
                setLimit(event.target.value ? Number(event.target.value) : null)
              }
            />
          </label>
        </div>
        <div className="actions">
          <button
            className="primary"
            disabled={run.state === "running"}
            onClick={uruchom}
          >
            <Play size={17} fill="currentColor" /> Uruchom klastrowanie
          </button>
        </div>
      </section>

      {result && (
        <section className="metrics">
          <div className="metric">
            <b>{pct(result.metryki.trafnosc_oof)}</b>
            <span>trafność (out-of-fold)</span>
          </div>
          <div className="metric">
            <b>{result.metryki.ari?.toFixed(3)}</b>
            <span>ARI — chmura ma 0.885</span>
          </div>
          <div className="metric">
            <b>{pct(result.metryki.udzial_przyjetych)}</b>
            <span>przypisane automatycznie</span>
          </div>
          <div className="metric">
            <b>{pct(result.metryki.trafnosc_przyjetych)}</b>
            <span>trafność automatycznych</span>
          </div>
          <div className="metric warn">
            <b>{doPrzegladu}</b>
            <span>części do przeglądu</span>
          </div>
          <div className="metric">
            <b>{result.klastry.length}</b>
            <span>klastrów ({result.metryki.n_klas} znanych)</span>
          </div>
        </section>
      )}

      {result && (
        <div className="local-grid">
          <aside className="cluster-list">
            <div className="section-head">
              <Cpu size={18} />
              <h2>Klastry</h2>
            </div>
            <button
              className={filtrKlaster ? "chip" : "chip active"}
              onClick={() => setFiltrKlaster(null)}
            >
              wszystkie ({result.czesci.length} PN)
            </button>
            <div className="cluster-scroll">
              {result.klastry.map((c) => (
                <button
                  key={c.nazwa}
                  className={
                    filtrKlaster === c.nazwa ? "cluster-row active" : "cluster-row"
                  }
                  onClick={() =>
                    setFiltrKlaster(filtrKlaster === c.nazwa ? null : c.nazwa)
                  }
                >
                  <span className="cluster-name">
                    {c.propozycja && <Sparkles size={13} />} {c.nazwa}
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
                  placeholder="Szukaj po PN, opisie lub klastrze…"
                  value={szukaj}
                  onChange={(event) => setSzukaj(event.target.value)}
                />
              </label>
              <label className="switch-row">
                <input
                  type="checkbox"
                  checked={tylkoPrzeglad}
                  onChange={(event) => setTylkoPrzeglad(event.target.checked)}
                />
                <span>tylko do przeglądu</span>
              </label>
            </div>

            <div className="parts-scroll">
              <table className="parts-table">
                <thead>
                  <tr>
                    <th>PN</th>
                    <th>Opis</th>
                    <th>Klaster</th>
                    <th>Źródło</th>
                    <th>Margines</th>
                    <th>Przypnij na stałe</th>
                  </tr>
                </thead>
                <tbody>
                  {widoczne.map((p) => {
                    const przypiety = overrides[p.pn];
                    const wKoszyku = koszyk[p.pn];
                    return (
                      <tr key={p.pn} className={wKoszyku ? "pending" : ""}>
                        <td className="mono">{p.pn}</td>
                        <td className="desc" title={p.opis}>
                          {p.opis}
                        </td>
                        <td>
                          {p.klaster}
                          {przypiety && (
                            <span className="badge pinned">
                              <Pin size={11} /> przypięty
                            </span>
                          )}
                        </td>
                        <td>
                          <span className={`badge src-${p.zrodlo}`}>
                            {p.zrodlo}
                          </span>
                        </td>
                        <td className="mono">{p.pewnosc.toFixed(2)}</td>
                        <td>
                          <input
                            list="lista-klastrow"
                            className="pin-input"
                            placeholder="wpisz lub wybierz…"
                            value={wKoszyku ?? ""}
                            onChange={(event) => {
                              const v = event.target.value;
                              setKoszyk((k) => {
                                const next = { ...k };
                                if (v.trim()) next[p.pn] = v;
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
              <datalist id="lista-klastrow">
                {nazwyKlastrow.map((n) => (
                  <option key={n} value={n} />
                ))}
              </datalist>
              {!widoczne.length && (
                <p className="empty">Brak części spełniających filtry.</p>
              )}
            </div>

            {Object.keys(koszyk).length > 0 && (
              <div className="basket">
                <div className="basket-head">
                  <AlertTriangle size={16} />
                  <strong>
                    {Object.keys(koszyk).length} korekt do zapisania
                  </strong>
                  <button className="link" onClick={() => setKoszyk({})}>
                    <Trash2 size={13} /> wyczyść
                  </button>
                </div>
                <ul>
                  {Object.entries(koszyk).map(([pn, klaster]) => (
                    <li key={pn}>
                      <span className="mono">{pn}</span> → <b>{klaster}</b>
                    </li>
                  ))}
                </ul>
                <div className="actions">
                  <button
                    className="secondary"
                    onClick={() => zapiszKorekty(false)}
                  >
                    Zapisz bez douczania
                  </button>
                  <button
                    className="primary"
                    disabled={run.state === "running"}
                    onClick={() => zapiszKorekty(true)}
                  >
                    <Pin size={16} /> Zapisz i doucz model
                  </button>
                </div>
              </div>
            )}
          </section>
        </div>
      )}

      {!result && run.state !== "running" && (
        <p className="empty">
          Brak wyniku. Uruchom klastrowanie, aby zobaczyć klastry i kolejkę do
          przeglądu.
        </p>
      )}

      <div className="console">
        <div className="console-head">
          <span>
            <Terminal size={16} /> Log toru lokalnego
          </span>
          <span className={`run-status ${run.state}`}>
            {statusIcon} {run.state}
          </span>
        </div>
        <pre>
          {run.output.length
            ? run.output.join("\n")
            : "Gotowe do uruchomienia. Wyniki trafiają do agentic/wyniki/."}
        </pre>
      </div>
    </div>
  );
}
