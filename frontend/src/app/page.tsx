"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, type Session } from "@/lib/api";

const EXERCISE_LABELS: Record<string, string> = {
  wall_slide: "Wall Slide",
  band_er_side: "Band ER Side",
};

function LabelBadge({ label }: { label: string }) {
  const colors: Record<string, string> = {
    good: "bg-green-100 text-green-800",
    bad_tempo: "bg-orange-100 text-orange-800",
    bad_rom_partial: "bg-red-100 text-red-800",
    bad_elbow_drift_mild: "bg-purple-100 text-purple-800",
    unknown: "bg-gray-100 text-gray-600",
  };
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${colors[label] ?? colors.unknown}`}>
      {label.replace(/_/g, " ")}
    </span>
  );
}

function SessionCard({ session }: { session: Session }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-semibold text-gray-900 text-sm">{session.session_id}</h3>
          <p className="text-xs text-gray-500 mt-0.5">
            {EXERCISE_LABELS[session.exercise] ?? session.exercise} &middot; {session.user_id}
          </p>
        </div>
        <LabelBadge label={session.exercise} />
      </div>

      <p className="text-2xl font-bold text-gray-800 mb-4">{session.reps_detected ?? "—"}<span className="text-sm font-normal text-gray-500 ml-1">reps</span></p>

      <div className="flex gap-2 flex-wrap">
        <Link
          href={`/sessions/${session.session_id}`}
          className="text-xs px-3 py-1.5 bg-brand-50 text-brand-700 rounded-lg hover:bg-brand-100 font-medium"
        >
          Detail
        </Link>
        <Link
          href={`/sessions/${session.session_id}/review`}
          className="text-xs px-3 py-1.5 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 font-medium"
        >
          Review flags
        </Link>
        <a
          href={`/api/ui/${session.session_id}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs px-3 py-1.5 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 font-medium"
        >
          Label
        </a>
        <a
          href={`/api/sessions/${session.session_id}/report`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs px-3 py-1.5 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 font-medium"
        >
          Report
        </a>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.sessions
      .list()
      .then(setSessions)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="pb-16">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Sessions</h1>
          <p className="text-sm text-gray-500 mt-1">Processed exercise sessions</p>
        </div>
        <div className="flex items-center gap-2">
          <Link
            href="/reports"
            className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-200"
          >
            Reports
          </Link>
          <Link
            href="/upload"
            className="px-4 py-2 bg-brand-600 text-white rounded-lg text-sm font-medium hover:bg-brand-700"
          >
            + Upload video
          </Link>
        </div>
      </div>

      {loading && (
        <div className="text-center py-16 text-gray-500">Loading sessions…</div>
      )}

      {error && (
        <div className="rounded-lg bg-red-50 border border-red-200 p-4 text-red-700 text-sm">
          Could not connect to API: {error}
          <br />
          <span className="text-xs text-red-500">Make sure the backend is running: <code>uvicorn src.api.main:app --reload</code></span>
        </div>
      )}

      {!loading && !error && sessions.length === 0 && (
        <div className="text-center py-16">
          <p className="text-gray-500 mb-4">No sessions yet.</p>
          <Link href="/upload" className="px-4 py-2 bg-brand-600 text-white rounded-lg text-sm font-medium hover:bg-brand-700">
            Upload your first video
          </Link>
        </div>
      )}

      {sessions.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {sessions.map((s) => (
            <SessionCard key={s.session_id} session={s} />
          ))}
        </div>
      )}
    </div>
  );
}
