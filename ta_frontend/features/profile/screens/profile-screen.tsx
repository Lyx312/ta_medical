import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';

import { AppButton } from '@/components/ui/app-button';
import { FieldError, FieldLabel, TextField } from '@/components/ui/form-field';
import { ImageViewer } from '@/components/ui/image-viewer';
import { ScreenHeader } from '@/components/ui/screen-header';
import { SurfaceCard } from '@/components/ui/surface-card';
import { ThemedText } from '@/components/themed-text';
import { Colors, Opacity, Radii, Spacing, Typography } from '@/constants/theme';
import { useAuth } from '@/contexts/auth-context';
import { useThemePreference } from '@/contexts/theme-context';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { updateCurrentUserProfile, uploadCurrentUserInsuranceCard } from '@/services/api/auth';
import {
  createFamilyMember,
  deleteFamilyMember,
  getFamilyMembers,
  updateFamilyMember,
  uploadFamilyMemberInsuranceCard,
} from '@/services/api/family-members';
import type { ApiError, AuthUser, FamilyMember } from '@/services/api/types';

type ProfileScreenProps = {
  title: string;
};

type ProfileFormState = {
  nip: string;
  company_name: string;
  insurance_card_number: string;
  phone_number: string;
  mobile_number: string;
  bank_account_number: string;
  bank_account_holder_name: string;
  bank_name: string;
};

const USER_PROFILE_FIELDS: { key: keyof ProfileFormState; label: string; placeholder: string }[] = [
  { key: 'nip', label: 'NIP', placeholder: 'Nomor induk perusahaan' },
  { key: 'company_name', label: 'Nama Perusahaan', placeholder: 'Nama perusahaan' },
  { key: 'insurance_card_number', label: 'No. Kartu Asuransi', placeholder: 'Nomor kartu asuransi' },
  { key: 'phone_number', label: 'Nomor Telepon', placeholder: 'Nomor telepon' },
  { key: 'mobile_number', label: 'Nomor HP', placeholder: 'Nomor HP' },
  { key: 'bank_account_number', label: 'Nomor Rekening', placeholder: 'Nomor rekening' },
  { key: 'bank_account_holder_name', label: 'Atas Nama Rekening', placeholder: 'Nama pemilik rekening' },
  { key: 'bank_name', label: 'Bank', placeholder: 'Nama bank' },
];

const FAMILY_RELATION_OPTIONS = [
  { value: 'husband', label: 'Suami' },
  { value: 'wife', label: 'Istri' },
  { value: 'child', label: 'Anak' },
];

const emptyProfileForm: ProfileFormState = {
  nip: '',
  company_name: '',
  insurance_card_number: '',
  phone_number: '',
  mobile_number: '',
  bank_account_number: '',
  bank_account_holder_name: '',
  bank_name: '',
};

function profileFormFromUser(user: AuthUser | null): ProfileFormState {
  return {
    nip: user?.nip ?? '',
    company_name: user?.company_name ?? '',
    insurance_card_number: user?.insurance_card_number ?? '',
    phone_number: user?.phone_number ?? '',
    mobile_number: user?.mobile_number ?? '',
    bank_account_number: user?.bank_account_number ?? '',
    bank_account_holder_name: user?.bank_account_holder_name ?? '',
    bank_name: user?.bank_name ?? '',
  };
}

function trimOrNull(value: string) {
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function getFamilyRelationLabel(value: string) {
  return FAMILY_RELATION_OPTIONS.find((option) => option.value === value)?.label ?? value;
}

export function ProfileScreen({ title }: ProfileScreenProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const { user, refreshUser, signOut } = useAuth();
  const { setPreference } = useThemePreference();
  const [profileForm, setProfileForm] = useState<ProfileFormState>(emptyProfileForm);
  const [profileFormDirty, setProfileFormDirty] = useState(false);
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [isUploadingUserCard, setIsUploadingUserCard] = useState(false);
  const [profileError, setProfileError] = useState('');
  const [familyMembers, setFamilyMembers] = useState<FamilyMember[]>([]);
  const [isLoadingMembers, setIsLoadingMembers] = useState(false);
  const [isSavingMember, setIsSavingMember] = useState(false);
  const [memberError, setMemberError] = useState('');
  const [editingMemberId, setEditingMemberId] = useState<number | null>(null);
  const [memberName, setMemberName] = useState('');
  const [memberRelation, setMemberRelation] = useState('');
  const [memberInsuranceCardNumber, setMemberInsuranceCardNumber] = useState('');
  const [memberPhoneNumber, setMemberPhoneNumber] = useState('');
  const [memberMobileNumber, setMemberMobileNumber] = useState('');
  const [memberEmail, setMemberEmail] = useState('');
  const [memberInsuranceCardUri, setMemberInsuranceCardUri] = useState('');
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  const [viewerImageUri, setViewerImageUri] = useState<string | null>(null);
  const [viewerImageTitle, setViewerImageTitle] = useState('');
  const lastProfileUserIdRef = useRef<number | null>(null);

  const isUserProfile = user?.role === 'user';

  useEffect(() => {
    const currentUserId = user?.id ?? null;
    const userChanged = lastProfileUserIdRef.current !== currentUserId;

    if (userChanged || !profileFormDirty) {
      setProfileForm(profileFormFromUser(user));
      setProfileFormDirty(false);
      lastProfileUserIdRef.current = currentUserId;
    }
  }, [profileFormDirty, user]);

  const resetMemberForm = () => {
    setEditingMemberId(null);
    setMemberName('');
    setMemberRelation('');
    setMemberInsuranceCardNumber('');
    setMemberPhoneNumber('');
    setMemberMobileNumber('');
    setMemberEmail('');
    setMemberInsuranceCardUri('');
    setFormErrors({});
  };

  const loadFamilyMembers = useCallback(async () => {
    if (!isUserProfile) {
      return;
    }

    setIsLoadingMembers(true);
    setMemberError('');

    try {
      setFamilyMembers(await getFamilyMembers());
    } catch (err) {
      const apiError = err as ApiError;
      setMemberError(apiError.message ?? 'Unable to load family members right now.');
    } finally {
      setIsLoadingMembers(false);
    }
  }, [isUserProfile]);

  useFocusEffect(
    useCallback(() => {
      void refreshUser();
      void loadFamilyMembers();
    }, [loadFamilyMembers, refreshUser]),
  );

  const handleLogout = () => {
    Alert.alert('Logout', 'Are you sure you want to logout?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Logout', style: 'destructive', onPress: () => void signOut() },
    ]);
  };

  const requestCardImage = async (source: 'camera' | 'gallery') => {
    if (source === 'camera') {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Camera permission is required to take photos.');
        return null;
      }

      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ['images'],
        quality: 0.8,
      });
      return result.canceled ? null : result.assets[0]?.uri ?? null;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      allowsMultipleSelection: false,
      mediaTypes: ['images'],
      quality: 0.8,
    });
    return result.canceled ? null : result.assets[0]?.uri ?? null;
  };

  const uploadUserCardFrom = async (source: 'camera' | 'gallery') => {
    const uri = await requestCardImage(source);
    if (!uri) {
      return;
    }

    setIsUploadingUserCard(true);
    setProfileError('');

    try {
      await uploadCurrentUserInsuranceCard(uri);
      await refreshUser();
    } catch (err) {
      const apiError = err as ApiError;
      setProfileError(apiError.message ?? 'Unable to upload insurance card.');
    } finally {
      setIsUploadingUserCard(false);
    }
  };

  const selectMemberCardFrom = async (source: 'camera' | 'gallery') => {
    const uri = await requestCardImage(source);
    if (uri) {
      setMemberInsuranceCardUri(uri);
      setFormErrors((prev) => ({ ...prev, member_insurance_card: '' }));
    }
  };

  const validateProfileForm = () => {
    const nextErrors: Record<string, string> = {};

    for (const field of USER_PROFILE_FIELDS) {
      if (!profileForm[field.key].trim()) {
        nextErrors[field.key] = `${field.label} is required.`;
      }
    }

    setFormErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSaveProfile = async () => {
    if (!validateProfileForm()) {
      return;
    }

    setIsSavingProfile(true);
    setProfileError('');

    try {
      const updatedUser = await updateCurrentUserProfile({
        nip: trimOrNull(profileForm.nip),
        company_name: trimOrNull(profileForm.company_name),
        insurance_card_number: trimOrNull(profileForm.insurance_card_number),
        phone_number: trimOrNull(profileForm.phone_number),
        mobile_number: trimOrNull(profileForm.mobile_number),
        bank_account_number: trimOrNull(profileForm.bank_account_number),
        bank_account_holder_name: trimOrNull(profileForm.bank_account_holder_name),
        bank_name: trimOrNull(profileForm.bank_name),
      });
      await refreshUser();
      setProfileForm(profileFormFromUser(updatedUser));
      setProfileFormDirty(false);
    } catch (err) {
      const apiError = err as ApiError;
      setProfileError(apiError.message ?? 'Unable to update profile.');
    } finally {
      setIsSavingProfile(false);
    }
  };

  const validateMemberForm = () => {
    const nextErrors: Record<string, string> = {};

    if (!memberName.trim()) {
      nextErrors.name = 'Family member name is required.';
    }

    if (!memberRelation.trim()) {
      nextErrors.relation = 'Relation is required.';
    }

    if (!memberInsuranceCardNumber.trim()) {
      nextErrors.insurance_card_number = 'Insurance card number is required.';
    }

    if (!memberPhoneNumber.trim()) {
      nextErrors.phone_number = 'Phone number is required.';
    }

    if (!memberMobileNumber.trim()) {
      nextErrors.mobile_number = 'Mobile number is required.';
    }

    if (!memberEmail.trim()) {
      nextErrors.email = 'Email is required.';
    }

    if (!memberInsuranceCardUri.trim()) {
      nextErrors.member_insurance_card = 'Insurance card photo is required.';
    }

    setFormErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmitMember = async () => {
    if (!validateMemberForm()) {
      return;
    }

    setIsSavingMember(true);
    setMemberError('');

    try {
      const payload = {
        name: memberName.trim(),
        relation: memberRelation.trim(),
        insurance_card_number: trimOrNull(memberInsuranceCardNumber),
        phone_number: trimOrNull(memberPhoneNumber),
        mobile_number: trimOrNull(memberMobileNumber),
        email: trimOrNull(memberEmail),
      };
      let savedFamilyMember: FamilyMember;

      if (editingMemberId === null) {
        savedFamilyMember = await createFamilyMember(payload);
      } else {
        savedFamilyMember = await updateFamilyMember(editingMemberId, payload);
      }

      if (memberInsuranceCardUri && memberInsuranceCardUri !== savedFamilyMember.insurance_card_data_url) {
        savedFamilyMember = await uploadFamilyMemberInsuranceCard(savedFamilyMember.id, memberInsuranceCardUri);
      }

      setFamilyMembers((prev) => {
        const exists = prev.some((familyMember) => familyMember.id === savedFamilyMember.id);
        if (!exists) {
          return [...prev, savedFamilyMember];
        }
        return prev.map((familyMember) =>
          familyMember.id === savedFamilyMember.id ? savedFamilyMember : familyMember,
        );
      });
      resetMemberForm();
    } catch (err) {
      const apiError = err as ApiError;
      setMemberError(apiError.message ?? 'Unable to save family member.');
    } finally {
      setIsSavingMember(false);
    }
  };

  const handleEditMember = (familyMember: FamilyMember) => {
    setEditingMemberId(familyMember.id);
    setMemberName(familyMember.name);
    setMemberRelation(familyMember.relation);
    setMemberInsuranceCardNumber(familyMember.insurance_card_number ?? '');
    setMemberPhoneNumber(familyMember.phone_number ?? '');
    setMemberMobileNumber(familyMember.mobile_number ?? '');
    setMemberEmail(familyMember.email ?? '');
    setMemberInsuranceCardUri(familyMember.insurance_card_data_url ?? '');
    setFormErrors({});
    setMemberError('');
  };

  const confirmDeleteMember = (familyMember: FamilyMember) => {
    Alert.alert(
      'Delete Family Member',
      `Remove ${familyMember.name} from your saved family members?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => void handleDeleteMember(familyMember.id),
        },
      ],
    );
  };

  const handleDeleteMember = async (familyMemberId: number) => {
    setMemberError('');

    try {
      await deleteFamilyMember(familyMemberId);
      setFamilyMembers((prev) => prev.filter((familyMember) => familyMember.id !== familyMemberId));
      if (editingMemberId === familyMemberId) {
        resetMemberForm();
      }
    } catch (err) {
      const apiError = err as ApiError;
      setMemberError(apiError.message ?? 'Unable to delete family member.');
    }
  };

  const updateProfileField = (key: keyof ProfileFormState, value: string) => {
    setProfileForm((prev) => ({ ...prev, [key]: value }));
    setProfileFormDirty(true);
    setFormErrors((prev) => ({ ...prev, [key]: '' }));
  };

  const renderInsuranceCardPreview = (uri: string | null | undefined, title: string) => {
    if (!uri) {
      return <ThemedText style={styles.emptyText}>No insurance card photo uploaded.</ThemedText>;
    }

    return (
      <Pressable
        style={[styles.cardPreview, { borderColor: colors.divider }]}
        onPress={() => {
          setViewerImageUri(uri);
          setViewerImageTitle(title);
        }}>
        <Image source={{ uri }} style={styles.cardPreviewImage} />
        <View style={styles.cardPreviewText}>
          <ThemedText style={styles.cardPreviewTitle}>Insurance card uploaded</ThemedText>
          <ThemedText style={styles.cardPreviewMeta}>Tap to inspect</ThemedText>
        </View>
        <Ionicons name="expand-outline" size={18} color={colors.icon} />
      </Pressable>
    );
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <KeyboardAvoidingView
        style={styles.keyboardContainer}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
        <View style={styles.screen}>
          <ScrollView
            contentContainerStyle={styles.scrollContent}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}>
            <View style={styles.content}>
              <ScreenHeader title={title} />

              <SurfaceCard style={styles.card}>
                <ThemedText style={styles.cardTitle}>Profile Information</ThemedText>
                <ThemedText style={styles.info}>Name: {user?.full_name ?? 'Unknown'}</ThemedText>
                <ThemedText style={styles.info}>Email: {user?.email ?? 'Unknown'}</ThemedText>
                <ThemedText style={styles.info}>Role: {user?.role ?? 'Unknown'}</ThemedText>
              </SurfaceCard>

              {isUserProfile ? (
                <SurfaceCard style={styles.card}>
                  <View style={styles.sectionHeaderRow}>
                    <ThemedText style={styles.cardTitle}>Claim Profile</ThemedText>
                    <ThemedText
                      style={[styles.statusPill, { color: user?.profile_complete ? colors.cta : colors.error }]}>
                      {user?.profile_complete ? 'Complete' : 'Incomplete'}
                    </ThemedText>
                  </View>

                  {USER_PROFILE_FIELDS.map((field) => (
                    <View key={field.key} style={styles.fieldGroup}>
                      <FieldLabel>{field.label}</FieldLabel>
                      <TextField
                        value={profileForm[field.key]}
                        onChangeText={(text) => updateProfileField(field.key, text)}
                        placeholder={field.placeholder}
                      />
                      <FieldError message={formErrors[field.key]} />
                    </View>
                  ))}

                  <View style={styles.fieldGroup}>
                    <FieldLabel>Foto Kartu Asuransi</FieldLabel>
                    {renderInsuranceCardPreview(user?.insurance_card_data_url, 'Kartu Asuransi')}
                    <View style={styles.actionRow}>
                      <AppButton
                        label="Gallery"
                        variant="outline"
                        disabled={isUploadingUserCard}
                        onPress={() => void uploadUserCardFrom('gallery')}
                        style={styles.flexButton}
                      />
                      <AppButton
                        label="Camera"
                        variant="outline"
                        disabled={isUploadingUserCard}
                        onPress={() => void uploadUserCardFrom('camera')}
                        style={styles.flexButton}
                      />
                    </View>
                    {isUploadingUserCard ? <ActivityIndicator color={colors.primary} /> : null}
                  </View>

                  {user && !user.profile_complete ? (
                    <ThemedText style={[styles.missingText, { color: colors.error }]}>
                      Missing: {user.missing_profile_fields.join(', ')}
                    </ThemedText>
                  ) : null}
                  <FieldError message={profileError} />

                  <AppButton
                    label="Save Profile"
                    loading={isSavingProfile}
                    onPress={() => void handleSaveProfile()}
                  />
                </SurfaceCard>
              ) : null}

              <SurfaceCard style={styles.card}>
                <ThemedText style={styles.cardTitle}>Appearance</ThemedText>
                <View style={styles.preferenceRow}>
                  <View style={styles.preferenceTextBlock}>
                    <ThemedText style={styles.preferenceTitle}>Dark Mode</ThemedText>
                    <ThemedText style={styles.preferenceDescription}>
                      Switch between the light and dark app themes.
                    </ThemedText>
                  </View>
                  <Switch
                    value={colorScheme === 'dark'}
                    onValueChange={(value) => {
                      void setPreference(value ? 'dark' : 'light');
                    }}
                    trackColor={{ false: colors.divider, true: `${colors.primary}80` }}
                    thumbColor={colorScheme === 'dark' ? colors.primary : colors.cardBackground}
                    ios_backgroundColor={colors.divider}
                  />
                </View>
              </SurfaceCard>

              {isUserProfile ? (
                <SurfaceCard style={styles.card}>
                  <ThemedText style={styles.cardTitle}>Family Members</ThemedText>
                  <ThemedText style={styles.sectionDescription}>
                    Complete each family profile before submitting new claims.
                  </ThemedText>

                  <View style={styles.fieldGroup}>
                    <FieldLabel>Family Member Name</FieldLabel>
                    <TextField
                      value={memberName}
                      onChangeText={(text) => {
                        setMemberName(text);
                        setFormErrors((prev) => ({ ...prev, name: '' }));
                      }}
                      placeholder="Enter family member name"
                    />
                    <FieldError message={formErrors.name} />
                  </View>

                  <View style={styles.fieldGroup}>
                    <FieldLabel>Relation</FieldLabel>
                    <View style={styles.relationOptions}>
                      {FAMILY_RELATION_OPTIONS.map((option) => {
                        const selected = memberRelation === option.value;
                        return (
                          <Pressable
                            key={option.value}
                            style={[
                              styles.relationOption,
                              { borderColor: selected ? colors.primary : colors.divider },
                              selected && { backgroundColor: `${colors.primary}18` },
                            ]}
                            onPress={() => {
                              setMemberRelation(option.value);
                              setFormErrors((prev) => ({ ...prev, relation: '' }));
                            }}>
                            <ThemedText style={{ color: selected ? colors.primary : colors.text }}>{option.label}</ThemedText>
                          </Pressable>
                        );
                      })}
                    </View>
                    <FieldError message={formErrors.relation} />
                  </View>

                  <View style={styles.fieldGroup}>
                    <FieldLabel>No. Kartu Asuransi</FieldLabel>
                    <TextField
                      value={memberInsuranceCardNumber}
                      onChangeText={(text) => {
                        setMemberInsuranceCardNumber(text);
                        setFormErrors((prev) => ({ ...prev, insurance_card_number: '' }));
                      }}
                      placeholder="Nomor kartu asuransi"
                    />
                    <FieldError message={formErrors.insurance_card_number} />
                  </View>

                  <View style={styles.fieldGroup}>
                    <FieldLabel>Nomor Telepon</FieldLabel>
                    <TextField
                      value={memberPhoneNumber}
                      onChangeText={(text) => {
                        setMemberPhoneNumber(text);
                        setFormErrors((prev) => ({ ...prev, phone_number: '' }));
                      }}
                      placeholder="Nomor telepon"
                    />
                    <FieldError message={formErrors.phone_number} />
                  </View>

                  <View style={styles.fieldGroup}>
                    <FieldLabel>Nomor HP</FieldLabel>
                    <TextField
                      value={memberMobileNumber}
                      onChangeText={(text) => {
                        setMemberMobileNumber(text);
                        setFormErrors((prev) => ({ ...prev, mobile_number: '' }));
                      }}
                      placeholder="Nomor HP"
                    />
                    <FieldError message={formErrors.mobile_number} />
                  </View>

                  <View style={styles.fieldGroup}>
                    <FieldLabel>Email</FieldLabel>
                    <TextField
                      value={memberEmail}
                      onChangeText={(text) => {
                        setMemberEmail(text);
                        setFormErrors((prev) => ({ ...prev, email: '' }));
                      }}
                      placeholder="Email"
                      keyboardType="email-address"
                      autoCapitalize="none"
                    />
                    <FieldError message={formErrors.email} />
                  </View>

                  <View style={styles.fieldGroup}>
                    <FieldLabel>Foto Kartu Member Asuransi</FieldLabel>
                    {renderInsuranceCardPreview(memberInsuranceCardUri, 'Kartu Member Asuransi')}
                    <View style={styles.actionRow}>
                      <AppButton
                        label="Gallery"
                        variant="outline"
                        disabled={isSavingMember}
                        onPress={() => void selectMemberCardFrom('gallery')}
                        style={styles.flexButton}
                      />
                      <AppButton
                        label="Camera"
                        variant="outline"
                        disabled={isSavingMember}
                        onPress={() => void selectMemberCardFrom('camera')}
                        style={styles.flexButton}
                      />
                    </View>
                    <FieldError message={formErrors.member_insurance_card} />
                  </View>

                  <FieldError message={memberError} />

                  <View style={styles.actionRow}>
                    <AppButton
                      label={editingMemberId === null ? 'Add Family Member' : 'Save Changes'}
                      loading={isSavingMember}
                      onPress={() => void handleSubmitMember()}
                      style={styles.flexButton}
                    />

                    {editingMemberId !== null ? (
                      <AppButton
                        label="Cancel"
                        variant="outline"
                        onPress={resetMemberForm}
                        disabled={isSavingMember}
                        style={styles.flexButton}
                      />
                    ) : null}
                  </View>

                  {isLoadingMembers ? (
                    <View style={styles.loadingState}>
                      <ActivityIndicator color={colors.primary} />
                    </View>
                  ) : familyMembers.length === 0 ? (
                    <ThemedText style={styles.emptyText}>No family members saved yet.</ThemedText>
                  ) : (
                    <View style={styles.familyMemberList}>
                      {familyMembers.map((familyMember) => (
                        <View
                          key={familyMember.id}
                          style={[
                            styles.familyMemberCard,
                            { borderColor: colors.divider, backgroundColor: colors.background },
                          ]}>
                          <View style={styles.familyMemberInfo}>
                            <ThemedText style={styles.familyMemberName}>{familyMember.name}</ThemedText>
                            <ThemedText style={styles.familyMemberRelation}>{getFamilyRelationLabel(familyMember.relation)}</ThemedText>
                            <ThemedText style={styles.familyMemberRelation}>
                              {familyMember.insurance_card_number ?? 'No insurance card number'}
                            </ThemedText>
                            <ThemedText style={styles.familyMemberRelation}>
                              {familyMember.mobile_number ?? 'No mobile number'}
                            </ThemedText>
                            {!familyMember.profile_complete ? (
                              <ThemedText style={[styles.missingText, { color: colors.error }]}>
                                Missing: {familyMember.missing_profile_fields.join(', ')}
                              </ThemedText>
                            ) : null}
                          </View>
                          <View style={styles.memberActionButtons}>
                            <Pressable onPress={() => handleEditMember(familyMember)}>
                              <ThemedText style={{ color: colors.primary }}>Edit</ThemedText>
                            </Pressable>
                            <Pressable onPress={() => confirmDeleteMember(familyMember)}>
                              <ThemedText style={{ color: colors.error }}>Delete</ThemedText>
                            </Pressable>
                          </View>
                        </View>
                      ))}
                    </View>
                  )}
                </SurfaceCard>
              ) : (
                <AppButton
                  label="Update Profile"
                  onPress={() => Alert.alert('Update Profile', 'Profile update functionality coming soon')}
                />
              )}

              <AppButton label="Logout" tone="error" onPress={handleLogout} />
            </View>
          </ScrollView>
        </View>
      </KeyboardAvoidingView>

      <ImageViewer
        visible={viewerImageUri !== null}
        uri={viewerImageUri}
        title={viewerImageTitle}
        onClose={() => {
          setViewerImageUri(null);
          setViewerImageTitle('');
        }}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  actionRow: {
    flexDirection: 'row',
    gap: Spacing.md,
  },
  card: {
    gap: 10,
    marginTop: Spacing.xl,
  },
  cardPreview: {
    alignItems: 'center',
    borderRadius: Radii.md,
    borderWidth: 1,
    flexDirection: 'row',
    gap: Spacing.md,
    padding: Spacing.sm,
  },
  cardPreviewImage: {
    borderRadius: Radii.sm,
    height: 56,
    width: 80,
  },
  cardPreviewMeta: {
    opacity: Opacity.muted,
  },
  cardPreviewText: {
    flex: 1,
    gap: 2,
  },
  cardPreviewTitle: {
    ...Typography.bodyStrong,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  container: {
    flex: 1,
  },
  content: {
    flexGrow: 1,
    padding: Spacing.xl,
    paddingBottom: 120,
  },
  emptyText: {
    opacity: Opacity.muted,
  },
  familyMemberCard: {
    backgroundColor: 'transparent',
    borderRadius: Radii.md,
    borderWidth: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 14,
  },
  familyMemberInfo: {
    flex: 1,
    gap: 4,
  },
  familyMemberList: {
    gap: 10,
  },
  familyMemberName: {
    ...Typography.bodyStrong,
  },
  familyMemberRelation: {
    opacity: Opacity.subtle,
  },
  fieldGroup: {
    gap: 6,
  },
  flexButton: {
    flex: 1,
  },
  info: {
    fontSize: 16,
    opacity: Opacity.secondary,
  },
  keyboardContainer: {
    flex: 1,
  },
  loadingState: {
    paddingVertical: Spacing.lg,
  },
  memberActionButtons: {
    alignItems: 'flex-end',
    gap: Spacing.md,
    justifyContent: 'center',
  },
  missingText: {
    ...Typography.helper,
  },
  preferenceDescription: {
    opacity: Opacity.subtle,
  },
  preferenceRow: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: Spacing.lg,
    justifyContent: 'space-between',
  },
  preferenceTextBlock: {
    flex: 1,
    gap: 4,
  },
  preferenceTitle: {
    ...Typography.bodyStrong,
  },
  relationOption: {
    alignItems: 'center',
    borderRadius: Radii.md,
    borderWidth: 1,
    flex: 1,
    paddingHorizontal: Spacing.sm,
    paddingVertical: 12,
  },
  relationOptions: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  scrollContent: {
    flexGrow: 1,
  },
  screen: {
    flex: 1,
  },
  sectionDescription: {
    opacity: Opacity.subtle,
  },
  sectionHeaderRow: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statusPill: {
    ...Typography.helper,
    fontWeight: '700',
  },
});
