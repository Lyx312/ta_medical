export type AuthUser = {
  id: number;
  full_name: string;
  email: string;
  role: 'user' | 'admin';
};

export type FamilyMember = {
  id: number;
  name: string;
  relation: string;
};

export type AIModelConfiguration = {
  id: number;
  display_name: string;
  model_name: string;
  provider: string;
  priority: number;
  is_active: boolean;
  supports_image_input: boolean;
  is_scan_compatible: boolean;
  activation_block_reason: string | null;
};

export type LoginResponse = {
  message: string;
  access_token: string;
  token_type: string;
  user: AuthUser;
};

export type RegisterResponse = {
  message: string;
  user: AuthUser;
  access_token?: string;
};

export type ApiError = {
  message: string;
  status?: number;
};

export type ApiErrorResponse = {
  message?: string;
  errors?: {
    field?: string;
    message?: string;
  }[];
};

export type AIDiagnosis = {
  description: string;
  doctor_name: string;
};

export type AIMedicine = {
  name: string;
  quantity: number;
  unit: string;
  dosage_instruction: string;
  unit_price: number;
  total_price: number;
};

export type AIOtherCharge = {
  category: string;
  description: string;
  amount: number;
};

export type AIMedicalScanResult = {
  scan_log_id: number;
  summary: {
    detected_documents: string[];
    medical_facility_name: string;
    service_date: string | null;
    transaction_date?: string | null;
    currency: string;
    total_amount: number;
  };
  patient_info: {
    name: string;
    date_of_birth: string;
    age: number;
    gender: string;
    address: string;
  };
  medical_context: {
    anamnesis: string;
    diagnoses: AIDiagnosis[];
    therapy_note: string;
  };
  medicines: AIMedicine[];
  other_charges: AIOtherCharge[];
  financial_recap: {
    total_medicines: number;
    total_other_charges: number;
    grand_total: number;
  };
};

export type ClaimResponse = {
  id: number;
  user_id: number;
  user_email: string;
  nama_pasien: string;
  tanggal_pelayanan: string;
  fasilitas_kesehatan: string;
  jenis_manfaat: string;
  nominal_klaim: number;
  keterangan_penyakit: string;
  kronologi: string;
  file_count: number;
  status: string;
  source_scan_log_id: number | null;
  created_at: string;
};

export type AdminClaimFile = {
  id: number;
  filename: string;
  mime_type: string;
  byte_size: number;
  sha256_hash: string;
  file_data_url: string;
};

export type AdminClaimDetail = ClaimResponse & {
  files: AdminClaimFile[];
  metadata?: Record<string, unknown> | null;
};

export type AdminClaimListSort = 'newest' | 'oldest' | 'amount_desc' | 'amount_asc';

export type AdminClaimListParams = {
  q?: string;
  linked?: boolean;
  limit?: number;
  sort?: AdminClaimListSort;
};

export type AdminClaimSummary = {
  total_claims: number;
  total_claim_amount: number;
  linked_scan_count: number;
  manual_submission_count: number;
};

export type LoginPayload = {
  email: string;
  password: string;
};

export type RegisterPayload = {
  full_name: string;
  email: string;
  password: string;
  confirm_password: string;
};

export type CreateClaimPayload = {
  nama_pasien: string;
  tanggal_pelayanan: string;
  fasilitas_kesehatan: string;
  jenis_manfaat: string;
  nominal_klaim: number;
  keterangan_penyakit: string;
  kronologi: string;
  fileUris: string[];
  source_scan_log_id?: number;
};

export type CreateFamilyMemberPayload = {
  name: string;
  relation: string;
};

export type UpdateFamilyMemberPayload = {
  name: string;
  relation: string;
};

export type UpdateAIModelPayload = {
  is_active?: boolean;
  priority?: number;
};

export type AdminScanLogSummary = {
  id: number;
  user_id: number;
  user_full_name: string;
  user_email: string;
  status: string;
  file_count: number;
  linked_claim_ids: number[];
  ai_model_provider: string | null;
  ai_model_name: string | null;
  created_at: string;
  updated_at: string;
};

export type AdminScanLogFile = {
  id: number;
  original_filename: string;
  original_mime_type: string;
  original_byte_size: number;
  original_sha256_hash: string;
  original_image_data_url: string;
  processed_filename: string | null;
  processed_mime_type: string | null;
  processed_byte_size: number | null;
  processed_sha256_hash: string | null;
  processed_image_data_url: string | null;
};

export type AdminScanLogDetail = AdminScanLogSummary & {
  validation_payload: Record<string, unknown> | null;
  extraction_payload: Record<string, unknown> | null;
  error_payload: Record<string, unknown> | null;
  files: AdminScanLogFile[];
};

export type AdminScanLogListParams = {
  status?: string;
  q?: string;
  linked?: boolean;
  limit?: number;
};

export type AdminScanLogQueueSummary = {
  total_logs: number;
  succeeded_count: number;
  failed_count: number;
  unlinked_count: number;
};
