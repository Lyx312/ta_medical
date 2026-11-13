import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
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
import { router, type Href } from 'expo-router';
import EventSource, { type EventSourceEvent, type EventSourceListener } from 'react-native-sse';

import { AppButton } from '@/components/ui/app-button';
import { FieldError, FieldLabel, SelectField, TextAreaField, TextField } from '@/components/ui/form-field';
import { ImageViewer } from '@/components/ui/image-viewer';
import { ScreenHeader } from '@/components/ui/screen-header';
import { SurfaceCard } from '@/components/ui/surface-card';
import { ThemedText } from '@/components/themed-text';
import { Colors, Opacity, Radii, Spacing, Typography } from '@/constants/theme';
import { useAuth } from '@/contexts/auth-context';
import { ClaimScanProgressCard } from '@/features/claims/components/claim-scan-progress-card';
import { useColorScheme } from '@/hooks/use-color-scheme';
import {
  cancelScanDocumentsJob,
  getScanDocumentsJob,
  openScanDocumentsJobEventStream,
  startScanDocumentsJob,
} from '@/services/api/ai';
import { createClaimWithProgress } from '@/services/api/claims';
import { getFamilyMembers } from '@/services/api/family-members';
import type {
  AIMedicalScanResult,
  ApiError,
  ClaimMedicineInput,
  FamilyMember,
  ScanJobEventName,
  ScanProgressPayload,
} from '@/services/api/types';

const BENEFIT_TYPES = [
  { value: 'rawat_jalan', label: 'Rawat Jalan', disabled: false },
  { value: 'rawat_inap', label: 'Rawat Inap', disabled: true },
  { value: 'emergency', label: 'Emergency', disabled: true },
  { value: 'maternity', label: 'Maternity', disabled: true },
];

type PatientOption = {
  key: string;
  name: string;
  relation: string;
};

type ClaimMedicineDraft = {
  localId: string;
  name: string;
  source: ClaimMedicineInput['source'];
};

type ScanUiState = 'idle' | 'uploading' | 'processing' | 'succeeded' | 'failed';
type CaptureGuideAction = 'view' | 'gallery' | 'camera';

const DOCUMENT_CAPTURE_GUIDE_ITEMS = [
  'Capture one document per photo, not a collage.',
  'Keep the whole document inside the frame.',
  'Use bright, even lighting and avoid glare or shadows.',
  'Place the document flat and keep the camera parallel to it.',
  'Make sure text is sharp before uploading.',
  'Include every relevant receipt, prescription, diagnosis note, and bill detail.',
];

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

function createMedicineDraft(
  source: ClaimMedicineInput['source'] = 'user_input',
  overrides: Partial<ClaimMedicineDraft> = {},
): ClaimMedicineDraft {
  return {
    localId: `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
    name: '',
    source,
    ...overrides,
  };
}

function mapAIMedicinesToDrafts(medicines: AIMedicalScanResult['medicines']): ClaimMedicineDraft[] {
  return medicines.map((medicine) =>
    createMedicineDraft('ai_extracted', {
      name: typeof medicine.name === 'string' ? medicine.name.trim() : '',
    }),
  );
}

function toClaimMedicineInputList(medicines: ClaimMedicineDraft[]): ClaimMedicineInput[] {
  const result: ClaimMedicineInput[] = [];

  for (const medicine of medicines) {
    const name = medicine.name.trim();
    if (!name) {
      continue;
    }

    result.push({
      name,
      source: medicine.source,
    });
  }

  return result;
}

export function NewClaimScreen() {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const { user, refreshUser } = useAuth();
  const [profileGateUser, setProfileGateUser] = useState(user);
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
  const [medicines, setMedicines] = useState<ClaimMedicineDraft[]>([]);
  const [chronology, setChronology] = useState('');
  const [sourceScanLogId, setSourceScanLogId] = useState<number | undefined>(undefined);
  const [scanState, setScanState] = useState<ScanUiState>('idle');
  const [scanProgress, setScanProgress] = useState<ScanProgressPayload | null>(null);
  const [scanErrorMessage, setScanErrorMessage] = useState('');
  const [currentScanLogId, setCurrentScanLogId] = useState<number | undefined>(undefined);
  const [isCancellingScan, setIsCancellingScan] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatusText, setSubmitStatusText] = useState('');
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [captureGuideAction, setCaptureGuideAction] = useState<CaptureGuideAction | null>(null);
  const [viewerImageUri, setViewerImageUri] = useState<string | null>(null);
  const [viewerImageTitle, setViewerImageTitle] = useState('');
  const scanEventSourceRef = useRef<EventSource<ScanJobEventName> | null>(null);
  const scanStreamOpenedRef = useRef(false);
  const scanStreamSettledRef = useRef(false);
  const scanFallbackInFlightRef = useRef(false);

  const closeScanStream = useCallback(() => {
    if (scanEventSourceRef.current) {
      scanEventSourceRef.current.removeAllEventListeners();
      scanEventSourceRef.current.close();
      scanEventSourceRef.current = null;
    }
    scanStreamOpenedRef.current = false;
    scanStreamSettledRef.current = false;
    scanFallbackInFlightRef.current = false;
  }, []);

  const resetScanFeedback = useCallback(() => {
    closeScanStream();
    setScanState('idle');
    setScanProgress(null);
    setScanErrorMessage('');
    setCurrentScanLogId(undefined);
    setIsCancellingScan(false);
  }, [closeScanStream]);

  useEffect(() => () => closeScanStream(), [closeScanStream]);

  const activeUser = profileGateUser ?? user;

  const incompleteFamilyMembers = useMemo(
    () => familyMembers.filter((familyMember) => !familyMember.profile_complete),
    [familyMembers],
  );
  const claimGateMissingItems = useMemo(() => {
    const missingItems: string[] = [];

    if (activeUser && !activeUser.profile_complete) {
      missingItems.push(`Your profile: ${activeUser.missing_profile_fields.join(', ')}`);
    }

    for (const familyMember of incompleteFamilyMembers) {
      missingItems.push(`${familyMember.name}: ${familyMember.missing_profile_fields.join(', ')}`);
    }

    return missingItems;
  }, [activeUser, incompleteFamilyMembers]);
  const claimProfileBlocked = claimGateMissingItems.length > 0;

  const patientOptions: PatientOption[] = useMemo(() => {
    const options: PatientOption[] = [];

    if (activeUser?.full_name) {
      options.push({
        key: `self:${activeUser.id}`,
        name: activeUser.full_name,
        relation: 'Saya',
      });
    }

    for (const familyMember of familyMembers) {
      options.push({
        key: `family:${familyMember.id}`,
        name: familyMember.name,
        relation: familyMember.relation,
      });
    }

    return options;
  }, [activeUser?.full_name, activeUser?.id, familyMembers]);

  const getSelectedPatientOption = () => patientOptions.find((option) => option.key === selectedPatientKey);

  const findMatchingPatientOption = useCallback(
    (name: string) => {
      const normalizedName = normalizePatientName(name);
      if (!normalizedName) {
        return undefined;
      }

      return patientOptions.find((option) => normalizePatientName(option.name) === normalizedName);
    },
    [patientOptions],
  );

  const loadClaimReadiness = useCallback(async () => {
    setIsLoadingFamilyMembers(true);
    setFamilyMemberError('');

    try {
      const [refreshedUser, members] = await Promise.all([
        refreshUser(),
        getFamilyMembers(),
      ]);
      setProfileGateUser(refreshedUser);
      setFamilyMembers(members);
    } catch (err) {
      const apiError = err as ApiError;
      setFamilyMemberError(apiError.message ?? 'Unable to load claim profile data.');
    } finally {
      setIsLoadingFamilyMembers(false);
    }
  }, [refreshUser]);

  useFocusEffect(
    useCallback(() => {
      void loadClaimReadiness();
    }, [loadClaimReadiness]),
  );

  const launchImagePicker = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      allowsMultipleSelection: true,
      mediaTypes: ['images'],
      quality: 0.8,
    });

    if (!result.canceled) {
      const newUris = result.assets.map((asset) => asset.uri);
      setImageUris((prev) => [...prev, ...newUris]);
      setSourceScanLogId(undefined);
      resetScanFeedback();
      setErrors((prev) => ({ ...prev, images: '' }));
    }
  };

  const launchCamera = async () => {
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
      resetScanFeedback();
      setErrors((prev) => ({ ...prev, images: '' }));
    }
  };

  const openCaptureGuide = (action: CaptureGuideAction) => {
    setCaptureGuideAction(action);
  };

  const continueFromCaptureGuide = async () => {
    const action = captureGuideAction;
    setCaptureGuideAction(null);

    if (action === 'gallery') {
      await launchImagePicker();
      return;
    }

    if (action === 'camera') {
      await launchCamera();
    }
  };

  const removeImage = (index: number) => {
    setSourceScanLogId(undefined);
    resetScanFeedback();
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
  const scanBusy = scanState === 'uploading' || scanState === 'processing';

  const buildUploadProgressPayload = (uploadPercent: number): ScanProgressPayload => ({
    stage: 'queued',
    title: 'Uploading files',
    detail: 'Sending document images to server.',
    percent: Math.max(0, Math.min(15, Math.round(uploadPercent * 0.15))),
    status: 'pending',
    image_count: imageUris.length,
    total_runs: null,
    completed_runs: null,
    successful_runs: null,
    medicine_count: null,
  });

  const applyAIScanData = useCallback((data: AIMedicalScanResult) => {
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

    setMedicines(mapAIMedicinesToDrafts(data.medicines));
  }, [findMatchingPatientOption]);

  const updateMedicineDraft = (
    localId: string,
    field: keyof Omit<ClaimMedicineDraft, 'localId' | 'source'>,
    value: string,
  ) => {
    setMedicines((prev) =>
      prev.map((medicine) =>
        medicine.localId === localId
          ? {
              ...medicine,
              [field]: value,
            }
          : medicine,
      ),
    );
  };

  const addMedicineDraft = () => {
    setMedicines((prev) => [...prev, createMedicineDraft('user_input')]);
  };

  const removeMedicineDraft = (localId: string) => {
    setMedicines((prev) => prev.filter((medicine) => medicine.localId !== localId));
  };

  const reconcileScanJob = useCallback(
    async (scanLogId: number) => {
      const status = await getScanDocumentsJob(scanLogId);
      setScanProgress(status.progress);

      if (status.result) {
        applyAIScanData(status.result);
        setSourceScanLogId(status.scan_log_id);
        setScanErrorMessage('');
        setScanState('succeeded');
        closeScanStream();
        return status;
      }

      if (status.progress.status === 'failed') {
        setScanErrorMessage(status.error_message ?? 'Unable to scan documents.');
        setScanState('failed');
        closeScanStream();
        return status;
      }

      setScanErrorMessage('');
      setScanState('processing');
      return status;
    },
    [applyAIScanData, closeScanStream],
  );

  const openScanStream = useCallback(
    async (scanLogId: number) => {
      closeScanStream();

      const eventSource = await openScanDocumentsJobEventStream(scanLogId);
      scanEventSourceRef.current = eventSource;
      scanStreamOpenedRef.current = false;
      scanStreamSettledRef.current = false;
      scanFallbackInFlightRef.current = false;

      const applyProgressPayload = (event: EventSourceEvent<'progress', ScanJobEventName>) => {
        if (!event.data) {
          return;
        }
        const status = JSON.parse(event.data) as Awaited<ReturnType<typeof getScanDocumentsJob>>;
        setScanProgress(status.progress);
        setScanErrorMessage('');
        setScanState('processing');
      };

      const applyCompletedPayload = (event: EventSourceEvent<'completed', ScanJobEventName>) => {
        if (!event.data) {
          return;
        }
        const status = JSON.parse(event.data) as Awaited<ReturnType<typeof getScanDocumentsJob>>;
        if (status.result) {
          applyAIScanData(status.result);
          setSourceScanLogId(status.scan_log_id);
        }
        setScanProgress(status.progress);
        setScanErrorMessage('');
        setScanState('succeeded');
        scanStreamSettledRef.current = true;
        closeScanStream();
      };

      const applyFailedPayload = (
        event: EventSourceEvent<'failed' | 'cancelled', ScanJobEventName>,
        fallbackMessage: string,
      ) => {
        if (!event.data) {
          return;
        }
        const status = JSON.parse(event.data) as Awaited<ReturnType<typeof getScanDocumentsJob>>;
        setScanProgress(status.progress);
        setScanErrorMessage(status.error_message ?? fallbackMessage);
        setScanState('failed');
        scanStreamSettledRef.current = true;
        closeScanStream();
      };

      const reconcileAfterStreamError = async () => {
        if (scanFallbackInFlightRef.current || scanStreamSettledRef.current) {
          return;
        }

        scanFallbackInFlightRef.current = true;
        try {
          await reconcileScanJob(scanLogId);
        } catch (err) {
          if (!scanStreamOpenedRef.current) {
            const apiError = err as ApiError;
            setScanErrorMessage(apiError.message ?? 'Unable to refresh scan progress.');
            setScanState('failed');
            closeScanStream();
          }
        } finally {
          scanFallbackInFlightRef.current = false;
        }
      };

      const openListener: EventSourceListener<ScanJobEventName, 'open'> = () => {
        scanStreamOpenedRef.current = true;
      };
      const progressListener: EventSourceListener<ScanJobEventName, 'progress'> = (event) => {
        applyProgressPayload(event);
      };
      const completedListener: EventSourceListener<ScanJobEventName, 'completed'> = (event) => {
        applyCompletedPayload(event);
      };
      const failedListener: EventSourceListener<ScanJobEventName, 'failed'> = (event) => {
        applyFailedPayload(event, 'Unable to scan documents.');
      };
      const cancelledListener: EventSourceListener<ScanJobEventName, 'cancelled'> = (event) => {
        applyFailedPayload(event, 'Scan was cancelled by the user.');
      };
      const errorListener: EventSourceListener<ScanJobEventName, 'error'> = () => {
        void reconcileAfterStreamError();
      };

      eventSource.addEventListener('open', openListener);
      eventSource.addEventListener('progress', progressListener);
      eventSource.addEventListener('completed', completedListener);
      eventSource.addEventListener('failed', failedListener);
      eventSource.addEventListener('cancelled', cancelledListener);
      eventSource.addEventListener('error', errorListener);
    },
    [applyAIScanData, closeScanStream, reconcileScanJob],
  );

  const handleAIScan = async () => {
    if (imageUris.length === 0) {
      Alert.alert('No Images', 'Please upload or take photos of your medical documents first.');
      return;
    }

    closeScanStream();
    setScanState('uploading');
    setScanErrorMessage('');
    setScanProgress(buildUploadProgressPayload(0));
    setSourceScanLogId(undefined);
    setErrors({});

    try {
      const job = await startScanDocumentsJob(imageUris, (uploadPercent) => {
        setScanProgress(buildUploadProgressPayload(uploadPercent));
      });
      setCurrentScanLogId(job.scan_log_id);
      setScanProgress(job.progress);
      setScanState('processing');
      await openScanStream(job.scan_log_id);
    } catch (err) {
      const apiError = err as ApiError;
      setScanErrorMessage(apiError.message ?? 'Unable to scan documents.');
      setScanState('failed');
    }
  };

  const handleCancelScan = async () => {
    if (!currentScanLogId) return;
    
    setIsCancellingScan(true);
    try {
      const response = await cancelScanDocumentsJob(currentScanLogId);
      setScanProgress(response.progress);
      setScanState('failed');
      setScanErrorMessage('Scan was cancelled by the user.');
      closeScanStream();
    } catch (err) {
      const apiError = err as ApiError;
      setScanErrorMessage(apiError.message ?? 'Failed to cancel scan.');
    } finally {
      setIsCancellingScan(false);
      setCurrentScanLogId(undefined);
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
    setMedicines([]);
    setChronology('');
    setSourceScanLogId(undefined);
    resetScanFeedback();
    setSubmitStatusText('');
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
    setSubmitStatusText('Uploading claim files');

    try {
      const medicinesInput = toClaimMedicineInputList(medicines);
      const [selectedPatientType, selectedPatientId] = selectedPatient.key.split(':');
      const createdClaim = await createClaimWithProgress(
        {
          nama_pasien: selectedPatient.name,
          tanggal_pelayanan: formatDateForSubmission(serviceDate),
          fasilitas_kesehatan: healthFacility.trim(),
          jenis_manfaat: benefitType,
          nominal_klaim: parseFloat(claimAmount),
          keterangan_penyakit: diseaseDescription.trim(),
          kronologi: chronology.trim(),
          medicines: medicinesInput,
          fileUris: imageUris,
          source_scan_log_id: sourceScanLogId,
          selected_patient_type: selectedPatientType === 'family' ? 'family' : 'self',
          family_member_id: selectedPatientType === 'family' ? Number(selectedPatientId) : undefined,
        },
        (progressPercent) => {
          if (progressPercent >= 100) {
            setSubmitStatusText('Saving claim');
            return;
          }
          setSubmitStatusText(`Uploading claim files (${progressPercent}%)`);
        },
      );

      resetForm();
      router.replace(`/user/claims/${createdClaim.id}` as Href);
    } catch (err) {
      const apiError = err as ApiError;
      Alert.alert('Submission Failed', apiError.message ?? 'Unable to submit claim.');
    } finally {
      setIsSubmitting(false);
      setSubmitStatusText('');
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

              {isLoadingFamilyMembers ? (
                <SurfaceCard style={viewStyles.card}>
                  <ActivityIndicator color={colors.primary} />
                  <ThemedText style={textStyles.sectionDescription}>Checking profile completeness...</ThemedText>
                </SurfaceCard>
              ) : claimProfileBlocked ? (
                <SurfaceCard style={viewStyles.card}>
                  <ThemedText style={[textStyles.sectionTitle, { color: colors.primary }]}>Complete your profile first</ThemedText>
                  <ThemedText style={textStyles.sectionDescription}>
                    Please complete your profile and family member data before submitting a new claim.
                  </ThemedText>
                  <View style={viewStyles.missingProfileList}>
                    {claimGateMissingItems.map((item) => (
                      <View key={item} style={viewStyles.missingProfileRow}>
                        <Ionicons name="alert-circle-outline" size={18} color={colors.error} />
                        <ThemedText style={[textStyles.missingProfileText, { color: colors.error }]}>{item}</ThemedText>
                      </View>
                    ))}
                  </View>
                  <FieldError message={familyMemberError} />
                  <AppButton
                    label="Go to Profile"
                    tone="cta"
                    onPress={() => router.push('/user/profile' as Href)}
                  />
                </SurfaceCard>
              ) : (
                <>
              <SurfaceCard style={viewStyles.card}>
                <ThemedText style={[textStyles.sectionTitle, { color: colors.primary }]}>Document Upload</ThemedText>
                <ThemedText style={textStyles.sectionDescription}>
                  Upload or take photos of your medical documents
                </ThemedText>
                <View style={[viewStyles.guidePanel, { backgroundColor: `${colors.primary}0D` }]}>
                  <View style={viewStyles.guideHeaderRow}>
                    <View style={viewStyles.guideTitleRow}>
                      <Ionicons name="bulb-outline" size={18} color={colors.primary} />
                      <ThemedText style={[textStyles.guideTitle, { color: colors.primary }]}>
                        Document photo guide
                      </ThemedText>
                    </View>
                    <Pressable style={viewStyles.guideAction} onPress={() => openCaptureGuide('view')}>
                      <ThemedText style={[textStyles.guideActionText, { color: colors.secondary }]}>View guide</ThemedText>
                      <Ionicons name="chevron-forward" size={16} color={colors.secondary} />
                    </Pressable>
                  </View>
                  <View style={viewStyles.guideList}>
                    {DOCUMENT_CAPTURE_GUIDE_ITEMS.slice(0, 4).map((item) => (
                      <View key={item} style={viewStyles.guideRow}>
                        <Ionicons name="checkmark-circle-outline" size={16} color={colors.primary} />
                        <ThemedText style={textStyles.guideText}>{item}</ThemedText>
                      </View>
                    ))}
                  </View>
                </View>

                <View style={viewStyles.imageActions}>
                  <Pressable
                    style={[viewStyles.imageButton, { borderColor: colors.divider }, scanBusy && viewStyles.disabledBlock]}
                    onPress={() => openCaptureGuide('gallery')}
                    disabled={scanBusy}>
                    <Ionicons name="image-outline" size={20} color={colors.primary} />
                    <ThemedText style={textStyles.imageButtonText}>Gallery</ThemedText>
                  </Pressable>
                  <Pressable
                    style={[viewStyles.imageButton, { borderColor: colors.divider }, scanBusy && viewStyles.disabledBlock]}
                    onPress={() => openCaptureGuide('camera')}
                    disabled={scanBusy}>
                    <Ionicons name="camera-outline" size={20} color={colors.primary} />
                    <ThemedText style={textStyles.imageButtonText}>Camera</ThemedText>
                  </Pressable>
                </View>

                <FieldError message={errors.images} />

                {imageUris.length > 0 ? (
                  <View style={viewStyles.imageGrid}>
                    {imageUris.map((uri, index) => (
                      <View key={`${uri}-${index}`} style={viewStyles.imageThumbContainer}>
                        <Pressable
                          onPress={() => {
                            setViewerImageUri(uri);
                            setViewerImageTitle(`Document ${index + 1}`);
                          }}>
                          <Image source={{ uri }} style={imageStyles.imageThumb} />
                        </Pressable>
                        <Pressable style={viewStyles.removeImageButton} onPress={() => removeImage(index)} disabled={scanBusy}>
                          <Ionicons name="close-circle" size={22} color={colors.error} />
                        </Pressable>
                      </View>
                    ))}
                  </View>
                ) : null}

                {scanProgress && scanBusy ? (
                  <ClaimScanProgressCard 
                    progress={scanProgress}
                    onCancel={handleCancelScan}
                    isCancelling={isCancellingScan}
                  />
                ) : null}

                {scanState === 'succeeded' ? (
                  <SurfaceCard style={[viewStyles.feedbackCard, { borderColor: colors.divider }]}>
                    <ThemedText style={[textStyles.successTitle, { color: colors.cta }]}>AI review ready</ThemedText>
                    <ThemedText style={textStyles.feedbackText}>
                      AI review ready. Please check extracted fields before submitting claim.
                    </ThemedText>
                  </SurfaceCard>
                ) : null}

                {scanState === 'failed' ? (
                  <SurfaceCard style={[viewStyles.feedbackCard, { borderColor: colors.error }]}>
                    <ThemedText style={[textStyles.errorTitle, { color: colors.error }]}>Scan failed</ThemedText>
                    <ThemedText style={textStyles.feedbackText}>
                      {scanErrorMessage || 'Unable to scan documents.'}
                    </ThemedText>
                    <AppButton label="Try scan again" tone="error" variant="outline" onPress={handleAIScan} />
                  </SurfaceCard>
                ) : null}

                {imageUris.length > 0 ? (
                  <AppButton
                    label={scanBusy ? 'Scanning in progress' : 'Scan with AI'}
                    tone="secondary"
                    disabled={scanBusy}
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

                <View style={viewStyles.fieldGroup}>
                  <FieldLabel>Medicines (Optional)</FieldLabel>
                  {medicines.length === 0 ? (
                    <ThemedText style={textStyles.optionalHint}>No medicines added yet.</ThemedText>
                  ) : null}

                  {medicines.map((medicine, index) => (
                    <View
                      key={medicine.localId}
                      style={[viewStyles.medicineCard, { borderColor: colors.divider, backgroundColor: colors.background }]}> 
                      <View style={viewStyles.medicineHeaderRow}>
                        <ThemedText style={textStyles.medicineTitle}>Medicine #{index + 1}</ThemedText>
                        <Pressable onPress={() => removeMedicineDraft(medicine.localId)}>
                          <Ionicons name="trash-outline" size={16} color={colors.error} />
                        </Pressable>
                      </View>

                      <TextField
                        value={medicine.name}
                        onChangeText={(text) => updateMedicineDraft(medicine.localId, 'name', text)}
                        placeholder="Medicine name"
                      />
                    </View>
                  ))}

                  <Pressable
                    style={[viewStyles.addMedicineButton, { borderColor: colors.divider }]}
                    onPress={addMedicineDraft}>
                    <Ionicons name="add-circle-outline" size={18} color={colors.primary} />
                    <ThemedText style={textStyles.addMedicineButtonText}>Add Medicine</ThemedText>
                  </Pressable>
                </View>

                <AppButton
                  label="Submit Claim"
                  tone="cta"
                  loading={isSubmitting}
                  disabled={scanBusy}
                  onPress={handleSubmit}
                />
                {isSubmitting && submitStatusText ? (
                  <ThemedText style={textStyles.submitStatusText}>{submitStatusText}</ThemedText>
                ) : null}
              </SurfaceCard>
                </>
              )}
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
              renderItem={({ item }) => {
                const isSelected = benefitType === item.value;
                const isDisabled = item.disabled;

                return (
                  <Pressable
                    disabled={isDisabled}
                    style={[
                      viewStyles.benefitOption,
                      { borderBottomColor: colors.divider },
                      isSelected && !isDisabled && { backgroundColor: `${colors.secondary}20` },
                      isDisabled && viewStyles.disabledBlock,
                    ]}
                    onPress={() => {
                      setBenefitType(item.value);
                      setErrors((prev) => ({ ...prev, jenis_manfaat: '' }));
                      setShowBenefitPicker(false);
                    }}>
                    <ThemedText
                      style={[
                        textStyles.benefitOptionText,
                        { color: isSelected ? colors.secondary : colors.text },
                        isDisabled && { color: colors.icon },
                      ]}>
                      {item.label}
                    </ThemedText>
                    {isSelected && !isDisabled ? <Ionicons name="checkmark" size={22} color={colors.secondary} /> : null}
                  </Pressable>
                );
              }}
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

      <Modal
        transparent
        animationType="slide"
        visible={captureGuideAction !== null}
        onRequestClose={() => setCaptureGuideAction(null)}>
        <View style={viewStyles.modalOverlay}>
          <View style={[viewStyles.modalContent, viewStyles.guideModalContent, { backgroundColor: colors.cardBackground }]}>
            <View style={viewStyles.modalHeader}>
              <ThemedText style={textStyles.modalTitle}>Before scanning your documents</ThemedText>
              <Pressable onPress={() => setCaptureGuideAction(null)}>
                <ThemedText style={{ color: colors.secondary }}>Cancel</ThemedText>
              </Pressable>
            </View>
            <View style={viewStyles.guideModalList}>
              {DOCUMENT_CAPTURE_GUIDE_ITEMS.map((item) => (
                <View key={item} style={viewStyles.guideModalRow}>
                  <Ionicons name="checkmark-circle-outline" size={18} color={colors.primary} />
                  <ThemedText style={textStyles.guideModalText}>{item}</ThemedText>
                </View>
              ))}
            </View>
            {captureGuideAction === 'view' ? (
              <AppButton label="Close" tone="secondary" variant="outline" onPress={() => setCaptureGuideAction(null)} />
            ) : (
              <AppButton
                label={captureGuideAction === 'camera' ? 'Continue to Camera' : 'Continue to Gallery'}
                tone="cta"
                onPress={() => void continueFromCaptureGuide()}
              />
            )}
          </View>
        </View>
      </Modal>

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

const viewStyles = StyleSheet.create({
  addMedicineButton: {
    alignItems: 'center',
    borderRadius: Radii.md,
    borderStyle: 'dashed',
    borderWidth: 1,
    flexDirection: 'row',
    gap: Spacing.xs,
    justifyContent: 'center',
    paddingHorizontal: Spacing.md,
    paddingVertical: 12,
  },
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
  disabledBlock: {
    opacity: 0.5,
  },
  fieldGroup: {
    gap: 6,
  },
  feedbackCard: {
    borderWidth: 1,
    gap: Spacing.sm,
  },
  guideAction: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: 2,
    paddingVertical: 2,
  },
  guideHeaderRow: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: Spacing.sm,
    justifyContent: 'space-between',
  },
  guideList: {
    gap: 7,
  },
  guideModalContent: {
    maxHeight: '78%',
  },
  guideModalList: {
    gap: Spacing.sm,
    marginBottom: Spacing.lg,
  },
  guideModalRow: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  guidePanel: {
    borderRadius: Radii.md,
    gap: Spacing.sm,
    padding: Spacing.md,
  },
  guideRow: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: Spacing.xs,
  },
  guideTitleRow: {
    alignItems: 'center',
    flex: 1,
    flexDirection: 'row',
    gap: Spacing.xs,
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
  medicineCard: {
    borderRadius: Radii.md,
    borderWidth: 1,
    gap: Spacing.sm,
    padding: Spacing.sm,
  },
  medicineHeaderRow: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  missingProfileList: {
    gap: Spacing.sm,
  },
  missingProfileRow: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: Spacing.sm,
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
  addMedicineButtonText: {
    ...Typography.button,
    fontSize: 14,
  },
  benefitOptionText: {
    fontSize: 16,
  },
  errorTitle: {
    ...Typography.sectionTitle,
  },
  feedbackText: {
    ...Typography.helper,
  },
  guideActionText: {
    ...Typography.helper,
    fontWeight: '600',
  },
  guideModalText: {
    flex: 1,
    fontSize: 14,
  },
  guideText: {
    ...Typography.helper,
    flex: 1,
  },
  guideTitle: {
    ...Typography.bodyStrong,
    fontSize: 14,
  },
  imageButtonText: {
    fontSize: 14,
    fontWeight: '600',
  },
  medicineTitle: {
    ...Typography.bodyStrong,
    fontSize: 14,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  missingProfileText: {
    ...Typography.helper,
    flex: 1,
  },
  optionalHint: {
    ...Typography.helper,
    opacity: Opacity.subtle,
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
  submitStatusText: {
    ...Typography.helper,
    textAlign: 'center',
  },
  successTitle: {
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
