"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import Link from "next/link";
import { api, type SessionDetail, type Rep } from "@/lib/api";

const LABEL_COLORS: Record<string, string> = {
  good: "bg-green-100 text-green-800 border-green-400",
  bad_tempo: "bg-orange-100 text-orange-800 border-orange-400",
  bad_rom_partial: "bg-red-100 text-red-800 border-red-400",
  bad_elbow_drift_mild: "bg-purple-100 text-purple-800 border-purple-400",
};

const LABEL_ACTIVE: Record<string, string> = {
  good: "bg-green-500 text-white border-green-500",
  bad_tempo: "bg-orange-500 text-white border-orange-500",
  bad_rom_partial: "bg-red-500 text-white border-red-500",
  bad_elbow_drift_mild: "bg-purple-500 text-white border-purple-500",
};

function RepCard({
  rep,
  sessionId,
  allowedLabels,
  videoUrl,
  active,
  onLabel,
}: {
  rep: Rep;
  sessionId: string;
  allowedLabels: string[];
  videoUrl: string | null;
  active: boolean;
  onLabel: (repId: number, label: string) => void;
}) {
  const [label, setLabel] = useState<string | null>(rep.existing_label);
  const videoRef = useRef<HTMLVideoElement>(null);

  const hasClip = !!rep.clip_url;
  const start = rep.start_time_s;
  const end = rep.end_time_s;

  function handleLabel(lbl: string) {
    setLabel(lbl);
    onLabel(rep.rep_id, lbl);
    api.sessions.saveLabel(sessionId, rep.rep_id, lbl).catch(console.warn);
  }

  // Time-range playback for non-clipped reps
  useEffect(() => {
    const video = videoRef.current;
    if (!video || hasClip) return;

    function onLoaded() {
      video!.currentTime = start;
    }
    function onTimeUpdate() {
      if (video!.currentTime >= end) {
        video!.pause();
        video!.currentTime = start;
      }
    }
    function onPlay() {
      if (video!.currentTime < start || video!.currentTime >= end) {
        video!.currentTime = start;
      }
    }

    video.addEventListener("loadedmetadata", onLoaded);
    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("play", onPlay);
    return () => {
      video.removeEventListener("loadedmetadata", onLoaded);
      video.removeEventListener("timeupdate", onTimeUpdate);
      video.removeEventListener("play", onPlay);
    };
  }, [hasClip, start, end]);

  const src = hasClip ? rep.clip_url! : videoUrl;

  return (
    <div
      id={`rep-${rep.rep_id}`}
      className={`bg-white rounded-xl border-2 overflow-hidden mb-5 transition-colors ${
        active ? "border-blue-400 shadow-md" : "border-gray-200"
      }`}
    >
      {src ? (
        <video
          ref={videoRef}
          controls
          loop={hasClip}
          playsInline
          preload="metadata"
          className="w-full bg-black"
          style={{ maxHeight: 380 }}
        >
          <source src={src} type="video/mp4" />
        </video>
      ) : (
        <div className="w-full bg-gray-100 flex items-center justify-center text-gray-400 text-sm" style={{ height: 140 }}>
          No video available
        </div>
      )}

      <div className="p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-semibold text-gray-900">Rep {rep.rep_id}</span>
          <span className="text-xs text-gray-400">
            {start.toFixed(1)}s – {end.toFixed(1)}s ({(end - start).toFixed(1)}s)
          </span>
        </div>

        {rep.flagged && (
          <p className="text-xs text-orange-600 bg-orange-50 rounded px-2 py-1 mb-3 inline-block">
            Suggested: {rep.predicted_label.replace(/_/g, " ")}
          </p>
        )}

        <div className="flex gap-2 flex-wrap">
          {allowedLabels.map((lbl, i) => (
            <button
              key={lbl}
              onClick={() => handleLabel(lbl)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium border-2 transition-colors ${
                label === lbl
                  ? LABEL_ACTIVE[lbl] ?? "bg-gray-500 text-white border-gray-500"
                  : `${LABEL_COLORS[lbl] ?? "bg-gray-100 text-gray-600 border-gray-300"} hover:opacity-80`
              }`}
            >
              {lbl.replace(/_/g, " ")}
              <span className="text-[10px] opacity-60 ml-1">[{i + 1}]</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function LabelPage({ params }: { params: { id: string } }) {
  const { id } = params;
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentRep, setCurrentRep] = useState(0);
  const [labels, setLabels] = useState<Record<number, string>>({});
  const [toast, setToast] = useState(false);

  useEffect(() => {
    api.sessions
      .detail(id)
      .then((d) => {
        setDetail(d);
        // Restore existing labels
        const existing: Record<number, string> = {};
        d.reps.forEach((r) => {
          if (r.existing_label) existing[r.rep_id] = r.existing_label;
        });
        setLabels(existing);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [id]);

  const handleLabel = useCallback((repId: number, label: string) => {
    setLabels((prev) => ({ ...prev, [repId]: label }));
    setToast(true);
    setTimeout(() => setToast(false), 1200);
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    if (!detail) return;
    const allowedLabels = detail.allowed_labels;
    const reps = detail.reps;

    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      const num = parseInt(e.key);
      if (num >= 1 && num <= allowedLabels.length) {
        e.preventDefault();
        const label = allowedLabels[num - 1];
        const rep = reps[currentRep];
        if (rep) {
          handleLabel(rep.rep_id, label);
          api.sessions.saveLabel(id, rep.rep_id, label).catch(console.warn);
        }
        return;
      }

      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        setCurrentRep((prev) => {
          const next = Math.min(prev + 1, reps.length - 1);
          document.getElementById(`rep-${reps[next].rep_id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
          return next;
        });
        return;
      }
      if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        setCurrentRep((prev) => {
          const next = Math.max(prev - 1, 0);
          document.getElementById(`rep-${reps[next].rep_id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
          return next;
        });
        return;
      }

      if (e.key === " ") {
        e.preventDefault();
        const card = document.getElementById(`rep-${reps[currentRep]?.rep_id}`);
        const video = card?.querySelector("video");
        if (video) {
          if (video.paused) video.play();
          else video.pause();
        }
      }
    }

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [detail, currentRep, id, handleLabel]);

  // Intersection observer to track current rep
  useEffect(() => {
    if (!detail) return;
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            const repId = parseInt(e.target.id.replace("rep-", ""));
            const idx = detail.reps.findIndex((r) => r.rep_id === repId);
            if (idx >= 0) setCurrentRep(idx);
          }
        });
      },
      { threshold: 0.5 }
    );
    document.querySelectorAll('[id^="rep-"]').forEach((el) => {
      if (el.id.match(/^rep-\d+$/)) obs.observe(el);
    });
    return () => obs.disconnect();
  }, [detail]);

  if (loading) return <div className="text-center py-20 text-gray-500">Loading...</div>;
  if (error) return <div className="rounded-lg bg-red-50 border border-red-200 p-4 text-red-700 text-sm">{error}</div>;
  if (!detail) return null;

  const { reps, allowed_labels: allowedLabels, video_url: videoUrl } = detail;
  const displayName = String(detail.meta["display_name"] ?? detail.meta["exercise"] ?? "");
  const labeledCount = Object.keys(labels).length;
  const pct = Math.round((100 * labeledCount) / Math.max(reps.length, 1));

  return (
    <div className="pb-20">
      {/* Header */}
      <div className="mb-4">
        <Link href={`/sessions/${id}`} className="text-sm text-brand-600 hover:underline">
          &larr; {id}
        </Link>
        <h1 className="text-xl font-bold text-gray-900 mt-2">Label Reps</h1>
        <p className="text-sm text-gray-500">
          {displayName} &middot; {reps.length} reps &middot; {labeledCount}/{reps.length} labeled
        </p>
      </div>

      {/* Shortcuts */}
      <div className="text-xs text-gray-500 bg-gray-50 rounded-lg px-3 py-2 mb-4">
        <kbd className="bg-white border border-gray-300 rounded px-1">1</kbd>–
        <kbd className="bg-white border border-gray-300 rounded px-1">{allowedLabels.length}</kbd> labels
        &nbsp;&middot;&nbsp;
        <kbd className="bg-white border border-gray-300 rounded px-1">&larr;</kbd>
        <kbd className="bg-white border border-gray-300 rounded px-1">&rarr;</kbd> navigate
        &nbsp;&middot;&nbsp;
        <kbd className="bg-white border border-gray-300 rounded px-1">Space</kbd> play/pause
      </div>

      {/* Progress */}
      <div className="bg-gray-200 rounded-full h-1.5 mb-6">
        <div className="bg-green-500 h-1.5 rounded-full transition-all" style={{ width: `${pct}%` }} />
      </div>

      {/* Rep cards */}
      {reps.map((rep, i) => (
        <RepCard
          key={rep.rep_id}
          rep={rep}
          sessionId={id}
          allowedLabels={allowedLabels}
          videoUrl={videoUrl}
          active={i === currentRep}
          onLabel={handleLabel}
        />
      ))}

      {/* Toast */}
      <div
        className={`fixed bottom-6 right-6 bg-green-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-opacity ${
          toast ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      >
        Saved
      </div>
    </div>
  );
}
