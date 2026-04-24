const BASE = "/api";

export interface Session {
  session_id: string;
  exercise: string;
  display_name: string;
  user_id: string;
  reps_detected: number;
}

export interface ModelPrediction {
  prob_bad: number;
  predicted_bad: boolean;
  threshold: number;
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
  model_prediction: ModelPrediction | null;
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

export interface Health {
  status: "ok" | "degraded";
  sessions: number;
  models_available: string[];
  reports_dir_present: boolean;
  processed_dir_present: boolean;
}

export interface ExerciseMetrics {
  exercise: string;
  n_test: number;
  n_good: number;
  n_bad: number;
  n_sessions: number;
  threshold: number;
  feature_cols: string[];
  auc: number | null;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: number[][];
  bootstrap_auc_ci?: CI;
  bootstrap_precision_ci?: CI;
  bootstrap_recall_ci?: CI;
  label_detail_breakdown: Record<string, number>;
  per_label_model_recall: Record<
    string,
    { n: number; flagged_bad: number; recall: number | null }
  >;
}

export interface CI {
  point: number | null;
  lower: number;
  upper: number;
  method: string;
  cluster: boolean;
  n_groups: number;
  n_valid: number;
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
  reports: {
    health: () => get<Health>("/health"),
    metrics: () =>
      get<{ metrics: Record<string, ExerciseMetrics> }>("/reports/metrics"),
    figures: () => get<{ figures: Record<string, string> }>("/reports/figures"),
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
