import { useCallback, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';

import { ThemedText } from '@/components/themed-text';
import { Colors } from '@/constants/theme';
import { useAuth } from '@/contexts/auth-context';
import { useThemePreference } from '@/contexts/theme-context';
import { useColorScheme } from '@/hooks/use-color-scheme';
import {
  createFamilyMember,
  deleteFamilyMember,
  getFamilyMembers,
  updateFamilyMember,
} from '@/services/api/family-members';
import type { ApiError, FamilyMember } from '@/services/api/types';

type ProfileScreenProps = {
  title: string;
};

export function ProfileScreen({ title }: ProfileScreenProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const { user, signOut } = useAuth();
  const { setPreference } = useThemePreference();
  const [familyMembers, setFamilyMembers] = useState<FamilyMember[]>([]);
  const [isLoadingMembers, setIsLoadingMembers] = useState(false);
  const [isSavingMember, setIsSavingMember] = useState(false);
  const [memberError, setMemberError] = useState('');
  const [editingMemberId, setEditingMemberId] = useState<number | null>(null);
  const [memberName, setMemberName] = useState('');
  const [memberRelation, setMemberRelation] = useState('');
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});

  const isUserProfile = user?.role === 'user';

  const resetMemberForm = () => {
    setEditingMemberId(null);
    setMemberName('');
    setMemberRelation('');
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
      void loadFamilyMembers();
    }, [loadFamilyMembers]),
  );

  const handleLogout = () => {
    Alert.alert('Logout', 'Are you sure you want to logout?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Logout', style: 'destructive', onPress: () => void signOut() },
    ]);
  };

  const validateMemberForm = () => {
    const nextErrors: Record<string, string> = {};

    if (!memberName.trim()) {
      nextErrors.name = 'Family member name is required.';
    }

    if (!memberRelation.trim()) {
      nextErrors.relation = 'Relation is required.';
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
      if (editingMemberId === null) {
        const createdFamilyMember = await createFamilyMember({
          name: memberName.trim(),
          relation: memberRelation.trim(),
        });
        setFamilyMembers((prev) => [...prev, createdFamilyMember]);
      } else {
        const updatedFamilyMember = await updateFamilyMember(editingMemberId, {
          name: memberName.trim(),
          relation: memberRelation.trim(),
        });
        setFamilyMembers((prev) =>
          prev.map((familyMember) => (familyMember.id === updatedFamilyMember.id ? updatedFamilyMember : familyMember)),
        );
      }

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
              <ThemedText type="title">{title}</ThemedText>

              <View style={[styles.card, { backgroundColor: colors.cardBackground, borderColor: colors.divider }]}>
                <ThemedText style={styles.cardTitle}>Profile Information</ThemedText>
                <ThemedText style={styles.info}>Name: {user?.full_name ?? 'Unknown'}</ThemedText>
                <ThemedText style={styles.info}>Email: {user?.email ?? 'Unknown'}</ThemedText>
                <ThemedText style={styles.info}>Role: {user?.role ?? 'Unknown'}</ThemedText>
              </View>

              <View style={[styles.card, { backgroundColor: colors.cardBackground, borderColor: colors.divider }]}>
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
              </View>

              {isUserProfile ? (
                <View style={[styles.card, { backgroundColor: colors.cardBackground, borderColor: colors.divider }]}>
                  <ThemedText style={styles.cardTitle}>Family Members</ThemedText>
                  <ThemedText style={styles.sectionDescription}>
                    Save your dependents here so claims and document scanning can use the correct patient name.
                  </ThemedText>

              <View style={styles.fieldGroup}>
                <ThemedText style={styles.fieldLabel}>Family Member Name</ThemedText>
                <TextInput
                  value={memberName}
                  onChangeText={(text) => {
                    setMemberName(text);
                    setFormErrors((prev) => ({ ...prev, name: '' }));
                  }}
                  placeholder="Enter family member name"
                  placeholderTextColor={colors.icon}
                  style={[styles.input, { borderColor: colors.divider, color: colors.text }]}
                />
                {formErrors.name ? <ThemedText style={[styles.errorText, { color: colors.error }]}>{formErrors.name}</ThemedText> : null}
              </View>

              <View style={styles.fieldGroup}>
                <ThemedText style={styles.fieldLabel}>Relation</ThemedText>
                <TextInput
                  value={memberRelation}
                  onChangeText={(text) => {
                    setMemberRelation(text);
                    setFormErrors((prev) => ({ ...prev, relation: '' }));
                  }}
                  placeholder="e.g., Spouse, Child, Parent"
                  placeholderTextColor={colors.icon}
                  style={[styles.input, { borderColor: colors.divider, color: colors.text }]}
                />
                {formErrors.relation ? (
                  <ThemedText style={[styles.errorText, { color: colors.error }]}>{formErrors.relation}</ThemedText>
                ) : null}
              </View>

              {memberError ? <ThemedText style={[styles.errorText, { color: colors.error }]}>{memberError}</ThemedText> : null}

              <View style={styles.actionRow}>
                <Pressable
                  style={[styles.primaryButton, styles.flexButton, { backgroundColor: colors.primary, opacity: isSavingMember ? 0.7 : 1 }]}
                  onPress={() => void handleSubmitMember()}
                  disabled={isSavingMember}>
                  {isSavingMember ? (
                    <ActivityIndicator color={colors.cardBackground} />
                  ) : (
                    <ThemedText style={[styles.buttonText, { color: colors.cardBackground }]}>
                      {editingMemberId === null ? 'Add Family Member' : 'Save Changes'}
                    </ThemedText>
                  )}
                </Pressable>

                {editingMemberId !== null ? (
                  <Pressable
                    style={[styles.secondaryButton, { borderColor: colors.divider }]}
                    onPress={resetMemberForm}
                    disabled={isSavingMember}>
                    <ThemedText style={[styles.secondaryButtonText, { color: colors.text }]}>Cancel</ThemedText>
                  </Pressable>
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
                      style={[styles.familyMemberCard, { borderColor: colors.divider, backgroundColor: colors.background }]}>
                      <View style={styles.familyMemberInfo}>
                        <ThemedText style={styles.familyMemberName}>{familyMember.name}</ThemedText>
                        <ThemedText style={styles.familyMemberRelation}>{familyMember.relation}</ThemedText>
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
                </View>
              ) : (
                <Pressable
                  style={[styles.primaryButton, { backgroundColor: colors.primary }]}
                  onPress={() => Alert.alert('Update Profile', 'Profile update functionality coming soon')}>
                  <ThemedText style={[styles.buttonText, { color: colors.cardBackground }]}>Update Profile</ThemedText>
                </Pressable>
              )}

              <Pressable style={[styles.primaryButton, { backgroundColor: colors.error }]} onPress={handleLogout}>
                <ThemedText style={[styles.buttonText, { color: colors.cardBackground }]}>Logout</ThemedText>
              </Pressable>
            </View>
          </ScrollView>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  actionRow: {
    flexDirection: 'row',
    gap: 12,
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  card: {
    borderRadius: 12,
    borderWidth: 1,
    gap: 10,
    marginTop: 20,
    padding: 16,
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
    padding: 20,
    paddingBottom: 120,
  },
  emptyText: {
    opacity: 0.7,
  },
  errorText: {
    fontSize: 12,
  },
  familyMemberCard: {
    borderRadius: 12,
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
    fontSize: 16,
    fontWeight: '600',
  },
  familyMemberRelation: {
    opacity: 0.75,
  },
  fieldGroup: {
    gap: 6,
  },
  fieldLabel: {
    fontSize: 14,
    fontWeight: '600',
  },
  flexButton: {
    flex: 1,
  },
  info: {
    fontSize: 16,
    opacity: 0.8,
  },
  input: {
    borderRadius: 12,
    borderWidth: 1,
    fontSize: 16,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  loadingState: {
    paddingVertical: 16,
  },
  keyboardContainer: {
    flex: 1,
  },
  memberActionButtons: {
    alignItems: 'flex-end',
    gap: 12,
    justifyContent: 'center',
  },
  preferenceDescription: {
    opacity: 0.75,
  },
  preferenceRow: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: 16,
    justifyContent: 'space-between',
  },
  preferenceTextBlock: {
    flex: 1,
    gap: 4,
  },
  preferenceTitle: {
    fontSize: 16,
    fontWeight: '600',
  },
  primaryButton: {
    alignItems: 'center',
    borderRadius: 12,
    marginTop: 16,
    paddingVertical: 14,
  },
  scrollContent: {
    flexGrow: 1,
  },
  screen: {
    flex: 1,
  },
  secondaryButton: {
    alignItems: 'center',
    borderRadius: 12,
    borderWidth: 1,
    flex: 1,
    justifyContent: 'center',
    marginTop: 16,
    paddingHorizontal: 18,
  },
  secondaryButtonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  sectionDescription: {
    opacity: 0.75,
  },
});
