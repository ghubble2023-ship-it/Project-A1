const BASE = process.env.EXPO_PUBLIC_BACKEND_URL;

if (!BASE) {
  // Fails fast if the env is missing (matches the coding guidelines).
  console.warn('EXPO_PUBLIC_BACKEND_URL is not set');
}

const API = `${BASE}/api`;

async function req<T>(path: string, method: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`API ${method} ${path} failed: ${res.status} ${txt}`);
  }
  return res.json();
}

export type TextResult = {
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  category_count: number;
  fired_categories: string[];
  matched_phrases: Record<string, string[]>;
  money_request_present: boolean;
};

export type PhotoResult = {
  flag_count: number;
  flags_triggered: string[];
  observation_quality: string;
  photo_status: string;
};

export type ProfileResult = {
  profile_concern: 'LOW' | 'MEDIUM' | 'ELEVATED';
  signal_count: number;
  signals_fired: string[];
};

export type PhoneResult = {
  valid: boolean;
  digits: string;
  region: string;
  inferred_area_code: string | null;
  flags: string[];
  summary: string;
};

export type FullResult = {
  engine: string;
  overall_risk: 'LOW' | 'MEDIUM' | 'HIGH';
  text: TextResult;
  photo: PhotoResult;
  profile: ProfileResult;
  phone: PhoneResult | null;
};

export type ScanRecord = {
  id: string;
  scan_type: string;
  input_summary: string;
  result: Record<string, any>;
  overall_risk: string;
  created_at: string;
};

export const api = {
  scanText: (body: { chat_messages: string; profile_text?: string | null }) =>
    req<TextResult>('/scan/text', 'POST', body),
  scanPhoto: (body: Record<string, any>) => req<PhotoResult>('/scan/photo', 'POST', body),
  scanProfile: (body: Record<string, any>) => req<ProfileResult>('/scan/profile', 'POST', body),
  scanPhone: (body: { phone_number: string; default_region?: string; claimed_location?: string | null }) =>
    req<PhoneResult>('/scan/phone', 'POST', body),
  scanFull: (body: Record<string, any>) => req<FullResult>('/scan/full', 'POST', body),
  saveScan: (body: { scan_type: string; input_summary: string; result: any; overall_risk: string }) =>
    req<ScanRecord>('/scans', 'POST', body),
  listScans: () => req<ScanRecord[]>('/scans', 'GET'),
  getScan: (id: string) => req<ScanRecord>(`/scans/${id}`, 'GET'),
  deleteScan: (id: string) => req<{ deleted: number }>(`/scans/${id}`, 'DELETE'),
  clearScans: () => req<{ deleted: number }>(`/scans`, 'DELETE'),
};
