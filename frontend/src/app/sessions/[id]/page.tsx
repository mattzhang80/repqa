"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, type SessionDetail, type Rep } from "@/lib/api";

const LABEL_COLORS: Record<string, string> = {
  good: "bg-green-100 text-green-800",
  bad_tempo: "bg-orange-100 text-orange-800",
  bad_rom_partial: "bg-red-100 text-red-800",
  bad_elbow_drift_mild: "bg-purple-100 text-purple-800",
  unknown: "bg-gray-100 text-gray-600",
};

function LabelBadge({ label }: { label: string }) {
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${LABEL_COLORS[label] ?? LABEL_COLORS.unknown}`}>
      {label.replace(/_/g, " ")}
    </span>
  );
}

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 text-center">
      <p className="text-2xl font-bold text-gray-900">{value}</p>
      <p className="text-xs text-gray-500 mt-0.5">{label}</p>
      {sub && <p className="text-xs text-gray-400">{sub}</p>}
    </div>
  );
}

function RepRow({ rep }: { rep: Rep }) {
  return (
    <tr className={rep.flagged ? "bg-orange-50" : ""}>
      <td className="px-4 py-2 text-sm font-mono text-gray-700">{rep.rep_id}</td>
      <td className="px-4 py-2 text-sm text-gray-600">
        {rep.start_time_s.toFixed(1)}s – {rep.end_time_s.toFixed(1)}s
        <span className="text-gray-400 ml-1">({rep.duration_s}s)</span>
      </td>
      <td className="px-4 py-2 text-sm text-gray-700">
        {rep.features["rom_proxy_max"] != null
          ? rep.features["rom_proxy_max"].toFixed(3)
          : "—"}
      </td>
      <td className="px-4 py-2 text-sm text-gray-700">
        {rep.features["tempo_s"] != null ? rep.features["tempo_s"].toFixed(1) + "s" : "—"}
      </td>
      <td className="px-4 py-2"><LabelBadge label={rep.predicted_label} /></td>
      <td className="px-4 py-2 text-xs text-gray-500 max-w-[200px]">
        {rep.reasons.length > 0 ? rep.reasons.join(", ") : "—"}
      </td>
      <td className="px-4 py-2">
        {rep.existing_label ? (
          <LabelBadge label={rep.existing_label} />
        ) : (
          <span className="text-gray-300 text-xs">unlabeled</span>
        )}
      </td>
      <td className="px-4 py-2">
        {rep.thumbnail_url && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={rep.thumbnail_url}
            alt={`rep ${rep.rep_id} thumbnail`}
            className="w-16 h-12 object-cover rounded"
          />
        )}
      </td>
    </tr>
  );
}

export default function SessionDetailPage({ params }: { params: { id: string } }) {
  const { id } = params;
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.sessions
      .detail(id)
      .then(setDetail)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="text-center py-20 text-gray-500">Loading…</div>;
  if (error) return <div className="rounded-lg bg-red-50 border border-red-200 p-4 text-red-700 text-sm">{error}</div>;
  if (!detail) return null;

  const { summary, meta, reps, plot_url, report_url } = detail;
  const exercise = String(meta["exercise"] ?? "");
  const displayName = String(meta["display_name"] ?? exercise);

  return (
    <div className="pb-20">
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <Link href="/" className="text-sm text-brand-600 hover:underline">&larr; Sessions</Link>
          <h1 className="text-xl font-bold text-gray-900 mt-2">{id}</h1>
          <p className="text-sm text-gray-500">{displayName} &middot; {String(meta["user_id"] ?? "")}</p>
        </div>
        <div className="flex gap-2">
          <Link
            href={`/sessions/${id}/review`}
            className="px-3 py-1.5 bg-brand-600 text-white rounded-lg text-sm font-medium hover:bg-brand-700"
          >
            Review flags
          </Link>
          <a
            href={`/api/ui/${id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="px-3 py-1.5 bg-gray-100 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-200"
          >
            Label reps
          </a>
          {report_url && (
            <a
              href={report_url}
              target="_blank"
              rel="noopener noreferrer"
              className="px-3 py-1.5 bg-gray-100 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-200"
            >
              Full report
            </a>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
        <StatCard label="Total reps" value={summary.total_reps} />
        <StatCard label="Good reps" value={summary.good_reps} />
        <StatCard label="Flagged" value={summary.flagged_reps} sub={`${summary.pct_flagged}%`} />
        <StatCard label="Exercise" value={displayName.split(" ").slice(0, 2).join(" ")} />
      </div>

      {/* Segmentation plot */}
      {plot_url && (
        <div className="bg-white rounded-xl border border-gray-200 p-4 mb-6">
          <h2 className="text-sm font-semibold text-gray-700 mb-3">Rep Segmentation Signal</h2>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={plot_url} alt="segmentation plot" className="w-full rounded" />
        </div>
      )}

      {/* Reps table */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-100">
          <h2 className="text-sm font-semibold text-gray-700">Reps ({reps.length})</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-gray-50 text-xs font-medium text-gray-500 uppercase tracking-wide">
              <tr>
                <th className="px-4 py-2">Rep</th>
                <th className="px-4 py-2">Time</th>
                <th className="px-4 py-2">ROM proxy</th>
                <th className="px-4 py-2">Tempo</th>
                <th className="px-4 py-2">Predicted</th>
                <th className="px-4 py-2">Reasons</th>
                <th className="px-4 py-2">Human label</th>
                <th className="px-4 py-2">Thumb</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {reps.map((rep) => <RepRow key={rep.rep_id} rep={rep} />)}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
