"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, type ExerciseMetrics, type Health, type CI } from "@/lib/api";

function Badge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
        ok ? "bg-green-100 text-green-700" : "bg-amber-100 text-amber-700"
      }`}
    >
      {label}
    </span>
  );
}

function Stat({
  label,
  value,
  ci,
}: {
  label: string;
  value: number | null | undefined;
  ci?: CI;
}) {
  const fmt = (x: number | null | undefined) =>
    x == null || Number.isNaN(x) ? "—" : x.toFixed(3);
  return (
    <div className="bg-gray-50 rounded-lg p-3">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="text-lg font-semibold text-gray-800">{fmt(value)}</p>
      {ci && Number.isFinite(ci.lower) && Number.isFinite(ci.upper) && (
        <p className="text-[10px] text-gray-500 mt-0.5">
          95% CI [{fmt(ci.lower)}, {fmt(ci.upper)}]
          {ci.cluster ? " · cluster" : ""}
        </p>
      )}
    </div>
  );
}

function ExerciseReport({ m }: { m: ExerciseMetrics }) {
  const cm = m.confusion_matrix;
  return (
    <section className="bg-white rounded-xl border border-gray-200 p-5 mb-6">
      <header className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900 capitalize">
          {m.exercise.replace(/_/g, " ")}
        </h2>
        <p className="text-xs text-gray-500">
          n={m.n_test} reps · {m.n_sessions} sessions · threshold {m.threshold.toFixed(2)}
        </p>
      </header>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
        <Stat label="AUC" value={m.auc} ci={m.bootstrap_auc_ci} />
        <Stat label="Precision @ t" value={m.precision} ci={m.bootstrap_precision_ci} />
        <Stat label="Recall @ t" value={m.recall} ci={m.bootstrap_recall_ci} />
        <Stat label="F1" value={m.f1} />
      </div>

      {/* Confusion matrix */}
      {cm && cm.length === 2 && (
        <div className="mb-6">
          <p className="text-xs text-gray-500 mb-1">Confusion matrix</p>
          <table className="text-xs border border-gray-200 rounded-md overflow-hidden">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-3 py-1 text-gray-500" />
                <th className="px-3 py-1 text-gray-500">pred good</th>
                <th className="px-3 py-1 text-gray-500">pred bad</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="px-3 py-1 text-gray-500 bg-gray-50">actual good</td>
                <td className="px-3 py-1 text-center">{cm[0][0]}</td>
                <td className="px-3 py-1 text-center">{cm[0][1]}</td>
              </tr>
              <tr>
                <td className="px-3 py-1 text-gray-500 bg-gray-50">actual bad</td>
                <td className="px-3 py-1 text-center">{cm[1][0]}</td>
                <td className="px-3 py-1 text-center">{cm[1][1]}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {/* Per-label recall */}
      {m.per_label_model_recall && Object.keys(m.per_label_model_recall).length > 0 && (
        <div className="mb-2">
          <p className="text-xs text-gray-500 mb-1">Per-label recall</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(m.per_label_model_recall).map(([lab, s]) => (
              <span
                key={lab}
                className="px-2 py-0.5 rounded bg-gray-50 border border-gray-200 text-xs"
              >
                <span className="text-gray-500">{lab.replace(/_/g, " ")}:</span>{" "}
                <span className="font-mono">
                  {s.flagged_bad}/{s.n}
                </span>
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

export default function ReportsPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [metrics, setMetrics] = useState<Record<string, ExerciseMetrics>>({});
  const [figures, setFigures] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.reports.health().catch(() => null),
      api.reports.metrics().catch(() => ({ metrics: {} })),
      api.reports.figures().catch(() => ({ figures: {} })),
    ])
      .then(([h, m, f]) => {
        setHealth(h);
        setMetrics(m.metrics);
        setFigures(f.figures);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-center py-20 text-gray-500">Loading…</div>;
  if (error)
    return (
      <div className="rounded-lg bg-red-50 border border-red-200 p-4 text-red-700 text-sm">
        {error}
      </div>
    );

  const exercises = Object.keys(metrics).sort();
  const forestUrl = figures["forest_test_metrics"];

  return (
    <div className="pb-20">
      <header className="mb-6">
        <Link href="/" className="text-sm text-brand-600 hover:underline">
          &larr; Dashboard
        </Link>
        <h1 className="text-xl font-bold text-gray-900 mt-2">Model reports</h1>
        <p className="text-sm text-gray-500">
          Phase 16 evaluation metrics on the held-out test split.
        </p>
      </header>

      {/* Health */}
      {health && (
        <section className="bg-white rounded-xl border border-gray-200 p-4 mb-6">
          <div className="flex items-center gap-3 text-sm text-gray-700">
            <Badge ok={health.status === "ok"} label={health.status} />
            <span>
              {health.sessions} processed session{health.sessions === 1 ? "" : "s"} ·
              models: {health.models_available.length > 0 ? health.models_available.join(", ") : "none"}
            </span>
          </div>
        </section>
      )}

      {/* Forest plot (headline figure) */}
      {forestUrl && (
        <section className="bg-white rounded-xl border border-gray-200 p-5 mb-6">
          <h2 className="text-sm font-semibold text-gray-900 mb-2">
            Test metrics across exercises (95% cluster-bootstrap CIs)
          </h2>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={forestUrl}
            alt="Forest plot of test metrics"
            className="w-full rounded-lg border border-gray-200"
          />
        </section>
      )}

      {/* Per-exercise metrics */}
      {exercises.length === 0 ? (
        <div className="text-center py-16 bg-white rounded-xl border border-gray-200">
          <p className="text-gray-500 text-sm">
            No reports yet. Run <code className="bg-gray-100 px-1 py-0.5 rounded">python scripts/eval_all.py</code>
            .
          </p>
        </div>
      ) : (
        exercises.map((ex) => <ExerciseReport key={ex} m={metrics[ex]} />)
      )}

      {/* Other figures */}
      {Object.keys(figures).length > 1 && (
        <section className="bg-white rounded-xl border border-gray-200 p-5">
          <h2 className="text-sm font-semibold text-gray-900 mb-3">All figures</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {Object.entries(figures)
              .filter(([k]) => k !== "forest_test_metrics")
              .map(([k, url]) => (
                <figure key={k} className="border border-gray-200 rounded-lg p-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={url} alt={k} className="w-full rounded" />
                  <figcaption className="text-xs text-gray-500 mt-1 text-center">
                    {k.replace(/_/g, " ")}
                  </figcaption>
                </figure>
              ))}
          </div>
        </section>
      )}
    </div>
  );
}
