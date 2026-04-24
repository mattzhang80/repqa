const BASE = "/api";

export interface Session {
  session_id: string;
  exercise: string;
  display_name: string;
  user_id: string;
  reps_detected: number;
}

export interface Rep {
  rep_id: number;
  start_time_s: number;
  end_time_s: number;
  duration_s: number;
  features: Record<string, number>;
  flagged: boolean;
  predicted_label: string;
  reasons: string[];
  confidence_level: string;
  existing_label: string | null;
  clip_url: string | null;
  thumbnail_url: string | null;
}

export interface SessionDetail {
  session_id: string;
  meta: Record<string, unknown>;
  reps: Rep[];
  allowed_labels: string[];
  video_url: string | null;
  summary: {
    total_reps: number;
    flagged_reps: number;
    good_reps: number;
    pct_flagged: number;
  };
  plot_url: string | null;
  report_url: string | null;
}

export interface JobStatus {
  job_id: string;
  state: "queued" | "running" | "done" | "error";
  progress: string;
  session_id?: string;
  error?: string;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  sessions: {
    list: () => get<Session[]>("/sessions"),
    detail: (id: string) => get<SessionDetail>(`/sessions/${id}`),
    saveLabel: async (sessionId: string, repId: number, label: string): Promise<void> => {
      const res = await fetch(`${BASE}/sessions/${sessionId}/reps/${repId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label }),
      });
      if (!res.ok) {
        const msg = await res.text().catch(() => res.statusText);
        throw new Error(msg);
      }
    },
  },
  jobs: {
    status: (id: string) => get<JobStatus>(`/jobs/${id}/status`),
  },
  upload: async (file: File, exercise: string, userId: string): Promise<{ job_id: string }> => {
    const form = new FormData();
    form.append("video_file", file);
    form.append("exercise", exercise);
    form.append("user_id", userId);
    const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
    if (!res.ok) {
      const msg = await res.text().catch(() => res.statusText);
      throw new Error(msg);
    }
    return res.json() as Promise<{ job_id: string }>;
  },
};
