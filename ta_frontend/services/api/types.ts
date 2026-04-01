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
  is_active: boolean;
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
  summary: {
    detected_documents: string[];
    medical_facility_name: string;
    transaction_date: string;
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
  created_at: string;
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
  is_active: boolean;
};
