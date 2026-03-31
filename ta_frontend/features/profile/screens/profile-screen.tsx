import { Alert, Pressable, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { Colors } from '@/constants/theme';
import { useAuth } from '@/contexts/auth-context';
import { useColorScheme } from '@/hooks/use-color-scheme';

type ProfileScreenProps = {
  title: string;
};

export function ProfileScreen({ title }: ProfileScreenProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const { user, signOut } = useAuth();

  const handleLogout = () => {
    Alert.alert('Logout', 'Are you sure you want to logout?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Logout', style: 'destructive', onPress: () => void signOut() },
    ]);
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={styles.content}>
        <ThemedText type="title">{title}</ThemedText>

        <View style={[styles.card, { backgroundColor: colors.cardBackground, borderColor: colors.divider }]}>
          <ThemedText style={styles.cardTitle}>Profile Information</ThemedText>
          <ThemedText style={styles.info}>Name: {user?.full_name ?? 'Unknown'}</ThemedText>
          <ThemedText style={styles.info}>Email: {user?.email ?? 'Unknown'}</ThemedText>
          <ThemedText style={styles.info}>Role: {user?.role ?? 'Unknown'}</ThemedText>
        </View>

        <Pressable
          style={[styles.button, { backgroundColor: colors.primary }]}
          onPress={() => Alert.alert('Update Profile', 'Profile update functionality coming soon')}>
          <ThemedText style={[styles.buttonText, { color: colors.cardBackground }]}>Update Profile</ThemedText>
        </Pressable>

        <Pressable style={[styles.button, { backgroundColor: colors.error }]} onPress={handleLogout}>
          <ThemedText style={[styles.buttonText, { color: colors.cardBackground }]}>Logout</ThemedText>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  button: {
    alignItems: 'center',
    borderRadius: 12,
    marginTop: 16,
    paddingVertical: 14,
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  card: {
    borderRadius: 12,
    borderWidth: 1,
    gap: 8,
    marginTop: 20,
    padding: 16,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 8,
  },
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  info: {
    fontSize: 16,
    opacity: 0.8,
  },
});
