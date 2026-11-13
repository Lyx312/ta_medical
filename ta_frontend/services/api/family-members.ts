import { deleteJson, getJson, postFormData, postJson, putJson } from './client';
import type {
  CreateFamilyMemberPayload,
  FamilyMember,
  UpdateFamilyMemberPayload,
} from './types';

export function getFamilyMembers(): Promise<FamilyMember[]> {
  return getJson<FamilyMember[]>('/family-members', true);
}

export function createFamilyMember(payload: CreateFamilyMemberPayload): Promise<FamilyMember> {
  return postJson<FamilyMember>('/family-members', payload, true);
}

export function updateFamilyMember(
  familyMemberId: number,
  payload: UpdateFamilyMemberPayload,
): Promise<FamilyMember> {
  return putJson<FamilyMember>(`/family-members/${familyMemberId}`, payload, true);
}

export function deleteFamilyMember(familyMemberId: number): Promise<void> {
  return deleteJson<void>(`/family-members/${familyMemberId}`, true);
}

export function uploadFamilyMemberInsuranceCard(familyMemberId: number, uri: string): Promise<FamilyMember> {
  const formData = new FormData();
  const filename = uri.split('/').pop() ?? `family_insurance_card_${Date.now()}.jpg`;
  const type = filename.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';

  formData.append('file', {
    uri,
    name: filename,
    type,
  } as never);

  return postFormData<FamilyMember>(`/family-members/${familyMemberId}/insurance-card`, formData, true);
}
