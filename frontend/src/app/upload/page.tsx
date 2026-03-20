"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { api } from "@/lib/api";

const EXERCISES = [
  { value: "wall_slide", label: "Wall Slide (Forearms on Wall)" },
  { value: "band_er_side", label: "Band External Rotation at Side" },
];

type UploadState = "idle" | "uploading" | "processing" | "done" | "error";

export default function UploadPage() {
  const router = useRouter();
  const fileRef = useRef<HTMLInputElement>(null);
  const [exercise, setExercise] = useState("wall_slide");
  const [userId, setUserId] = useState("user");
  const [state, setState] = useState<UploadState>("idle");
  const [progress, setProgress] = useState("");
  const [errMsg, setErrMsg] = useState("");
  const [fileName, setFileName] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const file = fileRef.current?.files?.[0];
    if (!file) return;

    setState("uploading");
    setProgress("Uploading video…");
    setErrMsg("");

    try {
      const { job_id } = await api.upload(file, exercise, userId);
      setState("processing");
      setProgress("Pipeline running…");

      // Poll job status
      const interval = setInterval(async () => {
        try {
          const job = await api.jobs.status(job_id);
          setProgress(job.progress ?? "processing…");
          if (job.state === "done" && job.session_id) {
            clearInterval(interval);
            setState("done");
            router.push(`/sessions/${job.session_id}`);
          } else if (job.state === "error") {
            clearInterval(interval);
            setState("error");
            setErrMsg(job.error ?? "Pipeline failed");
          }
        } catch {
          // transient poll error — keep trying
        }
      }, 2000);
    } catch (err: unknown) {
      setState("error");
      setErrMsg(err instanceof Error ? err.message : String(err));
    }
  }

  const busy = state === "uploading" || state === "processing";

  return (
    <div className="max-w-xl mx-auto pb-20">
      <h1 className="text-2xl font-bold text-gray-900 mb-2">Upload Video</h1>
      <p className="text-sm text-gray-500 mb-8">
        Run the full analysis pipeline on a new exercise recording.
      </p>

      <form onSubmit={handleSubmit} className="bg-white rounded-xl border border-gray-200 p-6 space-y-5">
        {/* File picker */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1.5">Video file</label>
          <div
            className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-brand-500 transition-colors"
            onClick={() => fileRef.current?.click()}
          >
            {fileName ? (
              <p className="text-sm text-gray-700 font-medium">{fileName}</p>
            ) : (
              <>
                <p className="text-sm text-gray-500">Click to select a .mp4 or .mov file</p>
                <p className="text-xs text-gray-400 mt-1">Landscape or portrait, side-view for wall slide</p>
              </>
            )}
          </div>
          <input
            ref={fileRef}
            type="file"
            accept=".mp4,.mov,video/*"
            className="hidden"
            onChange={(e) => setFileName(e.target.files?.[0]?.name ?? "")}
          />
        </div>

        {/* Exercise */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1.5">Exercise</label>
          <select
            value={exercise}
            onChange={(e) => setExercise(e.target.value)}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
          >
            {EXERCISES.map((ex) => (
              <option key={ex.value} value={ex.value}>{ex.label}</option>
            ))}
          </select>
        </div>

        {/* User ID */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1.5">User ID</label>
          <input
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="e.g. matthew"
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
          />
        </div>

        {/* Status */}
        {busy && (
          <div className="rounded-lg bg-brand-50 border border-brand-200 p-3 text-sm text-brand-700">
            <div className="flex items-center gap-2">
              <span className="animate-spin inline-block w-4 h-4 border-2 border-brand-500 border-t-transparent rounded-full" />
              {progress}
            </div>
            <p className="text-xs text-brand-500 mt-1">Pose extraction takes 1–2 min for a 3-min video.</p>
          </div>
        )}

        {state === "error" && (
          <div className="rounded-lg bg-red-50 border border-red-200 p-3 text-sm text-red-700">
            {errMsg}
          </div>
        )}

        <button
          type="submit"
          disabled={busy || !fileName}
          className="w-full py-2.5 bg-brand-600 text-white rounded-lg text-sm font-medium hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {busy ? "Processing…" : "Run pipeline"}
        </button>
      </form>
    </div>
  );
}
