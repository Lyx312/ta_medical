import { Pressable, StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Colors } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';

type AuthBottomNavProps = {
  activeTab: 'login' | 'register';
  onTabPress: (tab: 'login' | 'register') => void;
  disabled?: boolean;
};

export function AuthBottomNav({ activeTab, onTabPress, disabled = false }: AuthBottomNavProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: colors.cardBackground,
          borderColor: colors.divider,
        },
      ]}>
      <Pressable
        onPress={() => onTabPress('login')}
        disabled={disabled}
        style={[
          styles.tab,
          activeTab === 'login' && { backgroundColor: colors.secondary },
          disabled && styles.disabledTab,
        ]}>
        <ThemedText
          style={[
            styles.label,
            { color: activeTab === 'login' ? colors.cardBackground : colors.text },
          ]}>
          Login
        </ThemedText>
      </Pressable>

      <Pressable
        onPress={() => onTabPress('register')}
        disabled={disabled}
        style={[
          styles.tab,
          activeTab === 'register' && { backgroundColor: colors.secondary },
          disabled && styles.disabledTab,
        ]}>
        <ThemedText
          style={[
            styles.label,
            { color: activeTab === 'register' ? colors.cardBackground : colors.text },
          ]}>
          Register
        </ThemedText>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    borderTopWidth: 1,
    paddingHorizontal: 16,
    paddingVertical: 12,
    gap: 10,
  },
  tab: {
    flex: 1,
    borderRadius: 14,
    paddingVertical: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  disabledTab: {
    opacity: 0.7,
  },
  label: {
    fontSize: 15,
    fontWeight: '700',
  },
});
