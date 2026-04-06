import {
  Pressable,
  StyleSheet,
  TextInput,
  View,
  type PressableProps,
  type StyleProp,
  type TextInputProps,
  type TextStyle,
  type ViewStyle,
} from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Colors, Opacity, Radii, Spacing, Typography } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';

export type TextFieldProps = TextInputProps & {
  style?: StyleProp<TextStyle>;
};

export type SelectFieldProps = Omit<PressableProps, 'style'> & {
  value: React.ReactNode;
  helperText?: string;
  accessory?: React.ReactNode;
  isPlaceholder?: boolean;
  style?: StyleProp<ViewStyle>;
};

export function FieldLabel({ children, style }: { children: React.ReactNode; style?: StyleProp<TextStyle> }) {
  return <ThemedText style={[styles.label, style]}>{children}</ThemedText>;
}

export function FieldError({ message, style }: { message?: string; style?: StyleProp<TextStyle> }) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];

  if (!message) {
    return null;
  }

  return <ThemedText style={[styles.error, { color: colors.error }, style]}>{message}</ThemedText>;
}

export function TextField({ style, placeholderTextColor, ...rest }: TextFieldProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];

  return (
    <TextInput
      placeholderTextColor={placeholderTextColor ?? colors.icon}
      style={[
        styles.input,
        {
          borderColor: colors.divider,
          color: colors.text,
        },
        style,
      ]}
      {...rest}
    />
  );
}

export function TextAreaField({ style, multiline = true, numberOfLines = 3, ...rest }: TextFieldProps) {
  return <TextField multiline={multiline} numberOfLines={numberOfLines} style={[styles.textArea, style]} {...rest} />;
}

export function SelectField({
  value,
  helperText,
  accessory,
  isPlaceholder = false,
  style,
  ...rest
}: SelectFieldProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];

  return (
    <Pressable
      style={[
        styles.selectField,
        {
          borderColor: colors.divider,
        },
        style,
      ]}
      {...rest}>
      <View style={styles.valueContainer}>
        {typeof value === 'string' ? (
          <ThemedText style={[styles.selectValue, { color: isPlaceholder ? colors.icon : colors.text }]}>{value}</ThemedText>
        ) : (
          value
        )}
        {helperText ? <ThemedText style={styles.helper}>{helperText}</ThemedText> : null}
      </View>
      {accessory ? accessory : null}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  error: {
    ...Typography.error,
  },
  helper: {
    ...Typography.helper,
    opacity: Opacity.subtle,
  },
  input: {
    borderRadius: Radii.md,
    borderWidth: 1,
    fontSize: 16,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  label: {
    ...Typography.fieldLabel,
  },
  selectField: {
    alignItems: 'center',
    borderRadius: Radii.md,
    borderWidth: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  selectValue: {
    fontSize: 16,
  },
  textArea: {
    minHeight: 80,
    textAlignVertical: 'top',
  },
  valueContainer: {
    flex: 1,
    gap: Spacing.xs,
  },
});
