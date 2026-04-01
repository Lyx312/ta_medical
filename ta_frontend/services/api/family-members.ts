import { deleteJson, getJson, postJson, putJson } from './client';
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
