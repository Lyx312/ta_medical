import { useCallback, useState } from 'react';
import {
  Alert,
  FlatList,
  Image,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import DateTimePicker, { type DateTimePickerEvent } from '@react-native-community/datetimepicker';
import { useFocusEffect } from '@react-navigation/native';

import { AppButton } from '@/components/ui/app-button';
import { FieldError, FieldLabel, SelectField, TextAreaField, TextField } from '@/components/ui/form-field';
import { ScreenHeader } from '@/components/ui/screen-header';
import { SurfaceCard } from '@/components/ui/surface-card';
import { ThemedText } from '@/components/themed-text';
import { Colors, Opacity, Radii, Spacing, Typography } from '@/constants/theme';
import { useAuth } from '@/contexts/auth-context';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { scanDocuments } from '@/services/api/ai';
import { createClaim } from '@/services/api/claims';
import { getFamilyMembers } from '@/services/api/family-members';
import type { AIMedicalScanResult, ApiError, FamilyMember } from '@/services/api/types';

const BENEFIT_TYPES = [
  { value: 'rawat_jalan', label: 'Rawat Jalan' },
  { value: 'rawat_inap', label: 'Rawat Inap' },
  { value: 'emergency', label: 'Emergency' },
  { value: 'maternity', label: 'Maternity' },
];

type PatientOption = {
  key: string;
  name: string;
  relation: string;
};

function normalizePatientName(name: string) {
  return name.trim().toLocaleLowerCase();
}

function sanitizeClaimAmountInput(value: string) {
  return value.replace(/\D/g, '');
}

function formatClaimAmountForDisplay(value: string) {
  if (!value) {
    return '';
  }

  return new Intl.NumberFormat('id-ID').format(Number(value));
}

export function NewClaimScreen() {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const { user } = useAuth();
  const [imageUris, setImageUris] = useState<string[]>([]);
  const [familyMembers, setFamilyMembers] = useState<FamilyMember[]>([]);
  const [isLoadingFamilyMembers, setIsLoadingFamilyMembers] = useState(false);
  const [familyMemberError, setFamilyMemberError] = useState('');
  const [selectedPatientKey, setSelectedPatientKey] = useState('');
  const [showPatientPicker, setShowPatientPicker] = useState(false);
  const [serviceDate, setServiceDate] = useState(new Date());
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [healthFacility, setHealthFacility] = useState('');
  const [benefitType, setBenefitType] = useState('');
  const [showBenefitPicker, setShowBenefitPicker] = useState(false);
  const [claimAmount, setClaimAmount] = useState('');
  const [diseaseDescription, setDiseaseDescription] = useState('');
  const [chronology, setChronology] = useState('');
  const [sourceScanLogId, setSourceScanLogId] = useState<number | undefined>(undefined);
  const [isScanning, setIsScanning] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const patientOptions: PatientOption[] = [];
  if (user?.full_name) {
    patientOptions.push({
      key: `self:${user.id}`,
      name: user.full_name,
      relation: 'Saya',
    });
  }
  for (const familyMember of familyMembers) {
    patientOptions.push({
      key: `family:${familyMember.id}`,
      name: familyMember.name,
      relation: familyMember.relation,
    });
  }

  const getSelectedPatientOption = () => patientOptions.find((option) => option.key === selectedPatientKey);

  const findMatchingPatientOption = (name: string) => {
    const normalizedName = normalizePatientName(name);
    if (!normalizedName) {
      return undefined;
    }

    return patientOptions.find((option) => normalizePatientName(option.name) === normalizedName);
  };

  const loadFamilyMembers = useCallback(async () => {
    setIsLoadingFamilyMembers(true);
    setFamilyMemberError('');

    try {
      setFamilyMembers(await getFamilyMembers());
    } catch (err) {
      const apiError = err as ApiError;
      setFamilyMemberError(apiError.message ?? 'Unable to load family members.');
    } finally {
      setIsLoadingFamilyMembers(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void loadFamilyMembers();
    }, [loadFamilyMembers]),
  );

  const pickImages = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      allowsMultipleSelection: true,
      mediaTypes: ['images'],
      quality: 0.8,
    });

    if (!result.canceled) {
      const newUris = result.assets.map((asset) => asset.uri);
      setImageUris((prev) => [...prev, ...newUris]);
      setSourceScanLogId(undefined);
      setErrors((prev) => ({ ...prev, images: '' }));
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission Denied', 'Camera permission is required to take photos.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      quality: 0.8,
    });

    if (!result.canceled) {
      const newUris = result.assets.map((asset) => asset.uri);
      setImageUris((prev) => [...prev, ...newUris]);
      setSourceScanLogId(undefined);
      setErrors((prev) => ({ ...prev, images: '' }));
    }
  };

  const removeImage = (index: number) => {
    setSourceScanLogId(undefined);
    setImageUris((prev) => prev.filter((_, currentIndex) => currentIndex !== index));
  };

  const handleDateChange = (_event: DateTimePickerEvent, selectedDate?: Date) => {
    setShowDatePicker(Platform.OS === 'ios');

    if (selectedDate) {
      setServiceDate(selectedDate);
      setErrors((prev) => ({ ...prev, tanggal_pelayanan: '' }));
    }
  };

  const formatDateForDisplay = (date: Date) =>
    new Intl.DateTimeFormat('en-GB', {
      day: 'numeric',
      month: 'long',
      year: 'numeric',
    }).format(date);

  const formatDateForSubmission = (date: Date) => {
    const day = `${date.getDate()}`.padStart(2, '0');
    const month = `${date.getMonth() + 1}`.padStart(2, '0');
    const year = date.getFullYear();
    return `${day}-${month}-${year}`;
  };

  const parseExtractedDate = (value: string): Date | null => {
    const ymdMatch = value.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
    if (ymdMatch) {
      const [, yearRaw, monthRaw, dayRaw] = ymdMatch;
      const day = Number(dayRaw);
      const month = Number(monthRaw);
      const year = Number(yearRaw);
      const parsed = new Date(year, month - 1, day);
      if (
        parsed.getFullYear() === year &&
        parsed.getMonth() === month - 1 &&
        parsed.getDate() === day
      ) {
        return parsed;
      }
      return null;
    }
    return null;
  };

  const displayedClaimAmount = formatClaimAmountForDisplay(claimAmount);

  const applyAIScanData = (data: AIMedicalScanResult) => {
    if (data.patient_info.name) {
      const matchedPatientOption = findMatchingPatientOption(data.patient_info.name);
      if (matchedPatientOption) {
        setSelectedPatientKey(matchedPatientOption.key);
        setErrors((prev) => ({ ...prev, nama_pasien: '' }));
      }
    }

    if (data.summary.medical_facility_name) {
      setHealthFacility(data.summary.medical_facility_name);
    }

    const extractedServiceDate = data.summary.service_date ?? data.summary.transaction_date;
    if (extractedServiceDate) {
      const parsedDate = parseExtractedDate(extractedServiceDate);
      if (parsedDate) {
        setServiceDate(parsedDate);
      }
    }

    if (data.summary.total_amount) {
      setClaimAmount(sanitizeClaimAmountInput(data.summary.total_amount.toString()));
    }

    const diagnosisDescriptions = data.medical_context.diagnoses
      .map((diagnosis) => diagnosis.description?.trim())
      .filter((description): description is string => Boolean(description));

    if (diagnosisDescriptions.length > 0) {
      setDiseaseDescription(diagnosisDescriptions.join(', '));
    }

    if (data.medical_context.anamnesis) {
      setChronology(data.medical_context.anamnesis);
    }
  };

  const handleAIScan = async () => {
    if (imageUris.length === 0) {
      Alert.alert('No Images', 'Please upload or take photos of your medical documents first.');
      return;
    }

    setIsScanning(true);
    setSourceScanLogId(undefined);
    setErrors({});

    try {
      const result = await scanDocuments(imageUris);
      applyAIScanData(result);
      setSourceScanLogId(result.scan_log_id);
      Alert.alert('Success', 'Documents scanned and fields auto-filled. Please review the data.');
    } catch (err) {
      const apiError = err as ApiError;
      Alert.alert('Scan Failed', apiError.message ?? 'Unable to scan documents.');
    } finally {
      setIsScanning(false);
    }
  };

  const validate = () => {
    const nextErrors: Record<string, string> = {};
    const selectedPatient = getSelectedPatientOption();

    if (imageUris.length === 0) {
      nextErrors.images = 'At least one document image is required.';
    }

    if (!selectedPatient) {
      nextErrors.nama_pasien = 'Patient selection is required.';
    }

    if (!healthFacility.trim()) {
      nextErrors.fasilitas_kesehatan = 'Healthcare facility is required.';
    }

    if (!benefitType) {
      nextErrors.jenis_manfaat = 'Benefit type is required.';
    }

    if (!claimAmount || Number.isNaN(parseFloat(claimAmount)) || parseFloat(claimAmount) <= 0) {
      nextErrors.nominal_klaim = 'Valid claim amount is required.';
    }

    if (!diseaseDescription.trim()) {
      nextErrors.keterangan_penyakit = 'Disease description is required.';
    }

    if (!chronology.trim()) {
      nextErrors.kronologi = 'Chronology is required.';
    }

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const resetForm = () => {
    setImageUris([]);
    setSelectedPatientKey('');
    setServiceDate(new Date());
    setHealthFacility('');
    setBenefitType('');
    setClaimAmount('');
    setDiseaseDescription('');
    setChronology('');
    setSourceScanLogId(undefined);
    setErrors({});
  };

  const handleSubmit = async () => {
    const selectedPatient = getSelectedPatientOption();
    if (!validate()) {
      return;
    }

    if (!selectedPatient) {
      return;
    }

    setIsSubmitting(true);
    setErrors({});

    try {
      await createClaim({
        nama_pasien: selectedPatient.name,
        tanggal_pelayanan: formatDateForSubmission(serviceDate),
        fasilitas_kesehatan: healthFacility.trim(),
        jenis_manfaat: benefitType,
        nominal_klaim: parseFloat(claimAmount),
        keterangan_penyakit: diseaseDescription.trim(),
        kronologi: chronology.trim(),
        fileUris: imageUris,
        source_scan_log_id: sourceScanLogId,
      });

      Alert.alert('Success', 'Claim submitted successfully.', [{ text: 'OK', onPress: resetForm }]);
    } catch (err) {
      const apiError = err as ApiError;
      Alert.alert('Submission Failed', apiError.message ?? 'Unable to submit claim.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <SafeAreaView style={[viewStyles.container, { backgroundColor: colors.background }]}>
      <KeyboardAvoidingView
        style={viewStyles.keyboardContainer}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
        <View style={viewStyles.screen}>
          <ScrollView
            contentContainerStyle={viewStyles.scrollContent}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}>
            <View style={viewStyles.content}>
              <ScreenHeader title="New Claim" subtitle="Submit a new insurance claim" />

              <SurfaceCard style={viewStyles.card}>
                <ThemedText style={[textStyles.sectionTitle, { color: colors.primary }]}>Document Upload</ThemedText>
                <ThemedText style={textStyles.sectionDescription}>
                  Upload or take photos of your medical documents
                </ThemedText>

                <View style={viewStyles.imageActions}>
                  <Pressable style={[viewStyles.imageButton, { borderColor: colors.divider }]} onPress={pickImages}>
                    <Ionicons name="image-outline" size={20} color={colors.primary} />
                    <ThemedText style={textStyles.imageButtonText}>Gallery</ThemedText>
                  </Pressable>
                  <Pressable style={[viewStyles.imageButton, { borderColor: colors.divider }]} onPress={takePhoto}>
                    <Ionicons name="camera-outline" size={20} color={colors.primary} />
                    <ThemedText style={textStyles.imageButtonText}>Camera</ThemedText>
                  </Pressable>
                </View>

                <FieldError message={errors.images} />

                {imageUris.length > 0 ? (
                  <View style={viewStyles.imageGrid}>
                    {imageUris.map((uri, index) => (
                      <View key={`${uri}-${index}`} style={viewStyles.imageThumbContainer}>
                        <Image source={{ uri }} style={imageStyles.imageThumb} />
                        <Pressable style={viewStyles.removeImageButton} onPress={() => removeImage(index)}>
                          <Ionicons name="close-circle" size={22} color={colors.error} />
                        </Pressable>
                      </View>
                    ))}
                  </View>
                ) : null}

                {imageUris.length > 0 ? (
                  <AppButton
                    label="Scan with AI"
                    tone="secondary"
                    loading={isScanning}
                    leftIcon={<Ionicons name="scan-outline" size={20} color={colors.cardBackground} />}
                    onPress={handleAIScan}
                  />
                ) : null}
              </SurfaceCard>

              <SurfaceCard style={viewStyles.card}>
                <ThemedText style={[textStyles.sectionTitle, { color: colors.primary }]}>Claim Details</ThemedText>

                <View style={viewStyles.fieldGroup}>
                  <FieldLabel>Nama Pasien</FieldLabel>
                  <SelectField
                    onPress={() => setShowPatientPicker(true)}
                    value={
                      getSelectedPatientOption()
                        ? `${getSelectedPatientOption()?.name} (${getSelectedPatientOption()?.relation})`
                        : 'Pilih pasien'
                    }
                    helperText={isLoadingFamilyMembers ? 'Memuat anggota keluarga...' : undefined}
                    accessory={<Ionicons name="chevron-down" size={20} color={colors.icon} />}
                    isPlaceholder={!getSelectedPatientOption()}
                  />
                  <FieldError message={familyMemberError} />
                  <FieldError message={errors.nama_pasien} />
                </View>

                <View style={viewStyles.fieldGroup}>
                  <FieldLabel>Tanggal Pelayanan</FieldLabel>
                  <SelectField
                    onPress={() => setShowDatePicker(true)}
                    value={formatDateForDisplay(serviceDate)}
                    accessory={<Ionicons name="calendar-outline" size={20} color={colors.icon} />}
                  />
                  {showDatePicker ? (
                    Platform.OS === 'ios' ? (
                      <Modal transparent animationType="slide" visible={showDatePicker}>
                        <View style={viewStyles.modalOverlay}>
                          <View style={[viewStyles.modalContent, { backgroundColor: colors.cardBackground }]}>
                            <View style={viewStyles.modalHeader}>
                              <Pressable onPress={() => setShowDatePicker(false)}>
                                <ThemedText style={{ color: colors.secondary }}>Done</ThemedText>
                              </Pressable>
                            </View>
                            <DateTimePicker value={serviceDate} mode="date" display="spinner" onChange={handleDateChange} />
                          </View>
                        </View>
                      </Modal>
                    ) : (
                      <DateTimePicker value={serviceDate} mode="date" display="default" onChange={handleDateChange} />
                    )
                  ) : null}
                  <FieldError message={errors.tanggal_pelayanan} />
                </View>

                <View style={viewStyles.fieldGroup}>
                  <FieldLabel>Fasilitas Kesehatan</FieldLabel>
                  <TextField
                    value={healthFacility}
                    onChangeText={(text) => {
                      setHealthFacility(text);
                      setErrors((prev) => ({ ...prev, fasilitas_kesehatan: '' }));
                    }}
                    placeholder="e.g., RS IA KEBON JERUK"
                  />
                  <FieldError message={errors.fasilitas_kesehatan} />
                </View>

                <View style={viewStyles.fieldGroup}>
                  <FieldLabel>Jenis Manfaat</FieldLabel>
                  <SelectField
                    onPress={() => setShowBenefitPicker(true)}
                    value={benefitType ? BENEFIT_TYPES.find((benefit) => benefit.value === benefitType)?.label ?? '' : 'Select benefit type'}
                    accessory={<Ionicons name="chevron-down" size={20} color={colors.icon} />}
                    isPlaceholder={!benefitType}
                  />
                  <FieldError message={errors.jenis_manfaat} />
                </View>

                <View style={viewStyles.fieldGroup}>
                  <FieldLabel>Nominal Klaim (IDR)</FieldLabel>
                  <TextField
                    value={displayedClaimAmount}
                    onChangeText={(text) => {
                      setClaimAmount(sanitizeClaimAmountInput(text));
                      setErrors((prev) => ({ ...prev, nominal_klaim: '' }));
                    }}
                    placeholder="0"
                    keyboardType="numeric"
                  />
                  <FieldError message={errors.nominal_klaim} />
                </View>

                <View style={viewStyles.fieldGroup}>
                  <FieldLabel>Keterangan Penyakit (Diagnosis)</FieldLabel>
                  <TextAreaField
                    value={diseaseDescription}
                    onChangeText={(text) => {
                      setDiseaseDescription(text);
                      setErrors((prev) => ({ ...prev, keterangan_penyakit: '' }));
                    }}
                    placeholder="Describe the illness or condition"
                    numberOfLines={2}
                  />
                  <FieldError message={errors.keterangan_penyakit} />
                </View>

                <View style={viewStyles.fieldGroup}>
                  <FieldLabel>Kronologi (Anamnesis)</FieldLabel>
                  <TextAreaField
                    value={chronology}
                    onChangeText={(text) => {
                      setChronology(text);
                      setErrors((prev) => ({ ...prev, kronologi: '' }));
                    }}
                    placeholder="Describe the sequence of events"
                    numberOfLines={3}
                  />
                  <FieldError message={errors.kronologi} />
                </View>

                <AppButton
                  label="Submit Claim"
                  tone="cta"
                  loading={isSubmitting}
                  onPress={handleSubmit}
                />
              </SurfaceCard>
            </View>
          </ScrollView>
        </View>
      </KeyboardAvoidingView>

      <Modal transparent animationType="slide" visible={showBenefitPicker}>
        <View style={viewStyles.modalOverlay}>
          <View style={[viewStyles.modalContent, { backgroundColor: colors.cardBackground }]}>
            <View style={viewStyles.modalHeader}>
              <ThemedText style={textStyles.modalTitle}>Select Benefit Type</ThemedText>
              <Pressable onPress={() => setShowBenefitPicker(false)}>
                <ThemedText style={{ color: colors.secondary }}>Cancel</ThemedText>
              </Pressable>
            </View>
            <FlatList
              data={BENEFIT_TYPES}
              keyExtractor={(item) => item.value}
              renderItem={({ item }) => (
                <Pressable
                  style={[
                    viewStyles.benefitOption,
                    { borderBottomColor: colors.divider },
                    benefitType === item.value && { backgroundColor: `${colors.secondary}20` },
                  ]}
                  onPress={() => {
                    setBenefitType(item.value);
                    setErrors((prev) => ({ ...prev, jenis_manfaat: '' }));
                    setShowBenefitPicker(false);
                  }}>
                  <ThemedText
                    style={[
                      textStyles.benefitOptionText,
                      { color: benefitType === item.value ? colors.secondary : colors.text },
                    ]}>
                    {item.label}
                  </ThemedText>
                  {benefitType === item.value ? <Ionicons name="checkmark" size={22} color={colors.secondary} /> : null}
                </Pressable>
              )}
            />
          </View>
        </View>
      </Modal>

      <Modal transparent animationType="slide" visible={showPatientPicker}>
        <View style={viewStyles.modalOverlay}>
          <View style={[viewStyles.modalContent, { backgroundColor: colors.cardBackground }]}>
            <View style={viewStyles.modalHeader}>
              <ThemedText style={textStyles.modalTitle}>Pilih Pasien</ThemedText>
              <Pressable onPress={() => setShowPatientPicker(false)}>
                <ThemedText style={{ color: colors.secondary }}>Cancel</ThemedText>
              </Pressable>
            </View>
            <FlatList
              data={patientOptions}
              keyExtractor={(item) => item.key}
              renderItem={({ item }) => (
                <Pressable
                  style={[
                    viewStyles.benefitOption,
                    { borderBottomColor: colors.divider },
                    selectedPatientKey === item.key && { backgroundColor: `${colors.secondary}20` },
                  ]}
                  onPress={() => {
                    setSelectedPatientKey(item.key);
                    setErrors((prev) => ({ ...prev, nama_pasien: '' }));
                    setShowPatientPicker(false);
                  }}>
                  <View style={viewStyles.patientOptionTextContainer}>
                    <ThemedText
                      style={[
                        textStyles.benefitOptionText,
                        { color: selectedPatientKey === item.key ? colors.secondary : colors.text },
                      ]}>
                      {item.name}
                    </ThemedText>
                    <ThemedText style={textStyles.patientOptionMeta}>{item.relation}</ThemedText>
                  </View>
                  {selectedPatientKey === item.key ? <Ionicons name="checkmark" size={22} color={colors.secondary} /> : null}
                </Pressable>
              )}
            />
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const viewStyles = StyleSheet.create({
  benefitOption: {
    alignItems: 'center',
    borderBottomWidth: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.sm,
    paddingVertical: 14,
  },
  card: {
    gap: Spacing.md,
  },
  container: {
    flex: 1,
  },
  content: {
    flexGrow: 1,
    gap: Spacing.lg,
    padding: Spacing.xl,
    paddingBottom: 120,
  },
  fieldGroup: {
    gap: 6,
  },
  imageActions: {
    flexDirection: 'row',
    gap: 10,
  },
  imageButton: {
    alignItems: 'center',
    borderRadius: Radii.md,
    borderWidth: 1,
    flex: 1,
    flexDirection: 'row',
    gap: Spacing.sm,
    justifyContent: 'center',
    paddingHorizontal: Spacing.lg,
    paddingVertical: 14,
  },
  imageGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.sm,
  },
  imageThumbContainer: {
    position: 'relative',
  },
  modalContent: {
    borderTopLeftRadius: Spacing.xl,
    borderTopRightRadius: Spacing.xl,
    maxHeight: '50%',
    padding: Spacing.lg,
  },
  modalHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: Spacing.lg,
  },
  keyboardContainer: {
    flex: 1,
  },
  modalOverlay: {
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    flex: 1,
    justifyContent: 'flex-end',
  },
  patientOptionTextContainer: {
    flex: 1,
    gap: 2,
  },
  removeImageButton: {
    position: 'absolute',
    right: -6,
    top: -6,
  },
  scrollContent: {
    flexGrow: 1,
  },
  screen: {
    flex: 1,
  },
});

const textStyles = StyleSheet.create({
  benefitOptionText: {
    fontSize: 16,
  },
  imageButtonText: {
    fontSize: 14,
    fontWeight: '600',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  patientOptionMeta: {
    fontSize: 13,
    opacity: Opacity.muted,
  },
  sectionDescription: {
    fontSize: 14,
    opacity: Opacity.muted,
  },
  sectionTitle: {
    ...Typography.sectionTitle,
  },
});

const imageStyles = StyleSheet.create({
  imageThumb: {
    borderRadius: 8,
    height: 80,
    width: 80,
  },
});
