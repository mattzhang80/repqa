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

type Decision = "confirmed" | "dismissed" | null;

function RepCard({ rep, sessionId }: { rep: Rep; sessionId: string }) {
  const [decision, setDecision] = useState<Decision>(null);

  function decide(d: Decision) {
    setDecision(d);
    // Notify API (fire-and-forget; silently ignore if API not running)
    fetch(`/api/sessions/${sessionId}/reps/${rep.rep_id}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision: d }),
    }).catch(() => undefined);
    // Persist in localStorage
    if (typeof window !== "undefined") {
      localStorage.setItem(`review:${sessionId}:${rep.rep_id}`, d ?? "");
    }
  }

  // Restore from localStorage on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem(`review:${sessionId}:${rep.rep_id}`) as Decision | null;
      if (saved) setDecision(saved);
    }
  }, [sessionId, rep.rep_id]);

  return (
    <div
      className={`bg-white rounded-xl border p-5 ${
        decision === "confirmed"
          ? "border-orange-300"
          : decision === "dismissed"
          ? "border-gray-200 opacity-60"
          : "border-gray-200"
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <span className="text-sm font-semibold text-gray-900">Rep {rep.rep_id}</span>
          <span className="text-xs text-gray-400 ml-2">
            {rep.start_time_s.toFixed(1)}s – {rep.end_time_s.toFixed(1)}s
          </span>
        </div>
        <div className="flex flex-col items-end gap-1">
          <LabelBadge label={rep.predicted_label} />
          {rep.model_prediction && (
            <span
              className={`inline-block px-2 py-0.5 rounded text-[10px] font-medium ${
                rep.model_prediction.predicted_bad
                  ? "bg-red-50 text-red-700 border border-red-200"
                  : "bg-green-50 text-green-700 border border-green-200"
              }`}
              title={`Model threshold: ${rep.model_prediction.threshold.toFixed(2)}`}
            >
              model: {(rep.model_prediction.prob_bad * 100).toFixed(0)}% bad
            </span>
          )}
        </div>
      </div>

      {/* Video player */}
      {rep.clip_url ? (
        <video
          src={rep.clip_url}
          controls
          loop
          playsInline
          className="w-full rounded-lg bg-black mb-4"
          style={{ maxHeight: 280 }}
        />
      ) : rep.thumbnail_url ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={rep.thumbnail_url} alt={`rep ${rep.rep_id}`} className="w-full rounded-lg mb-4 object-cover" style={{ maxHeight: 280 }} />
      ) : (
        <div className="w-full rounded-lg bg-gray-100 mb-4 flex items-center justify-center text-gray-400 text-sm" style={{ height: 160 }}>
          No clip available
        </div>
      )}

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-2 text-center mb-4">
        <div className="bg-gray-50 rounded-lg p-2">
          <p className="text-xs text-gray-500">ROM proxy</p>
          <p className="text-sm font-semibold text-gray-800">
            {rep.features["rom_proxy_max"] != null ? rep.features["rom_proxy_max"].toFixed(3) : "—"}
          </p>
        </div>
        <div className="bg-gray-50 rounded-lg p-2">
          <p className="text-xs text-gray-500">Tempo</p>
          <p className="text-sm font-semibold text-gray-800">
            {rep.features["tempo_s"] != null ? `${rep.features["tempo_s"].toFixed(1)}s` : "—"}
          </p>
        </div>
        <div className="bg-gray-50 rounded-lg p-2">
          <p className="text-xs text-gray-500">Confidence</p>
          <p className="text-sm font-semibold text-gray-800 capitalize">{rep.confidence_level}</p>
        </div>
      </div>

      {/* Reasons */}
      {rep.reasons.length > 0 && (
        <p className="text-xs text-gray-500 mb-4">
          <span className="font-medium">Reasons:</span> {rep.reasons.join(", ")}
        </p>
      )}

      {/* Confirm / Dismiss */}
      <div className="flex gap-2">
        <button
          onClick={() => decide("confirmed")}
          className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
            decision === "confirmed"
              ? "bg-orange-500 text-white"
              : "bg-orange-50 text-orange-700 hover:bg-orange-100"
          }`}
        >
          Confirm flag
        </button>
        <button
          onClick={() => decide("dismissed")}
          className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
            decision === "dismissed"
              ? "bg-gray-400 text-white"
              : "bg-gray-100 text-gray-600 hover:bg-gray-200"
          }`}
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}

export default function ReviewPage({ params }: { params: { id: string } }) {
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

  const flagged = detail.reps.filter((r) => r.flagged);
  const displayName = String(detail.meta["display_name"] ?? detail.meta["exercise"] ?? "");

  return (
    <div className="pb-20">
      {/* Header */}
      <div className="mb-6">
        <Link href={`/sessions/${id}`} className="text-sm text-brand-600 hover:underline">
          &larr; {id}
        </Link>
        <h1 className="text-xl font-bold text-gray-900 mt-2">Review Flagged Reps</h1>
        <p className="text-sm text-gray-500">{displayName} &middot; {flagged.length} flag{flagged.length !== 1 ? "s" : ""}</p>
      </div>

      {flagged.length === 0 ? (
        <div className="text-center py-16 bg-white rounded-xl border border-gray-200">
          <p className="text-gray-500">No flagged reps in this session.</p>
          <Link href={`/sessions/${id}`} className="mt-3 inline-block text-sm text-brand-600 hover:underline">
            View session detail
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {flagged.map((rep) => (
            <RepCard key={rep.rep_id} rep={rep} sessionId={id} />
          ))}
        </div>
      )}
    </div>
  );
}
