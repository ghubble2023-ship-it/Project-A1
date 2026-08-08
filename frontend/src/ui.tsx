import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import * as Haptics from 'expo-haptics';
import { theme } from './theme';

// ---------------------------------------------------------------------------
// PrimaryButton
// ---------------------------------------------------------------------------
export function PrimaryButton({
  label,
  onPress,
  testID,
  disabled,
  variant = 'primary',
}: {
  label: string;
  onPress: () => void;
  testID?: string;
  disabled?: boolean;
  variant?: 'primary' | 'outline';
}) {
  return (
    <Pressable
      testID={testID}
      onPress={() => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
        onPress();
      }}
      disabled={disabled}
      style={({ pressed }) => [
        styles.btnBase,
        variant === 'primary' ? styles.btnPrimary : styles.btnOutline,
        pressed && !disabled && styles.btnPressed,
        disabled && styles.btnDisabled,
      ]}
    >
      <Text
        style={[
          styles.btnLabel,
          variant === 'primary' ? styles.btnLabelPrimary : styles.btnLabelOutline,
        ]}
      >
        {label}
      </Text>
    </Pressable>
  );
}

// ---------------------------------------------------------------------------
// SectionHeader
// ---------------------------------------------------------------------------
export function SectionHeader({ label, testID }: { label: string; testID?: string }) {
  return (
    <View style={styles.sectionHeader} testID={testID}>
      <Text style={styles.sectionHeaderText}>{label}</Text>
      <View style={styles.sectionRule} />
    </View>
  );
}

// ---------------------------------------------------------------------------
// FieldLabel
// ---------------------------------------------------------------------------
export function FieldLabel({ children, testID }: { children: string; testID?: string }) {
  return (
    <Text style={styles.fieldLabel} testID={testID}>
      {children.toUpperCase()}
    </Text>
  );
}

// ---------------------------------------------------------------------------
// Segmented control
// ---------------------------------------------------------------------------
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  testID,
}: {
  value: T;
  options: readonly T[];
  onChange: (v: T) => void;
  testID?: string;
}) {
  return (
    <View style={styles.segmented} testID={testID}>
      {options.map((opt, i) => {
        const active = opt === value;
        return (
          <Pressable
            key={opt}
            testID={`${testID}-opt-${opt}`}
            onPress={() => {
              Haptics.selectionAsync().catch(() => {});
              onChange(opt);
            }}
            style={[
              styles.segmentedItem,
              i > 0 && styles.segmentedItemBordered,
              active && styles.segmentedItemActive,
            ]}
          >
            <Text style={[styles.segmentedLabel, active && styles.segmentedLabelActive]}>
              {opt.toUpperCase()}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

// ---------------------------------------------------------------------------
// ToggleRow
// ---------------------------------------------------------------------------
export function ToggleRow({
  label,
  value,
  onChange,
  testID,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
  testID?: string;
}) {
  return (
    <Pressable
      testID={testID}
      style={styles.toggleRow}
      onPress={() => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
        onChange(!value);
      }}
    >
      <Text style={styles.toggleLabel} numberOfLines={2}>
        {label}
      </Text>
      <View style={[styles.toggleBox, value && styles.toggleBoxActive]}>
        {value ? <View style={styles.toggleDot} /> : null}
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  btnBase: {
    paddingVertical: 16,
    paddingHorizontal: 20,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: theme.border.thick,
    borderRadius: 0,
  },
  btnPrimary: {
    backgroundColor: theme.color.brandPrimary,
    borderColor: theme.color.borderStrong,
  },
  btnOutline: {
    backgroundColor: theme.color.surface,
    borderColor: theme.color.borderStrong,
  },
  btnPressed: { opacity: 0.85 },
  btnDisabled: { opacity: 0.4 },
  btnLabel: {
    fontFamily: theme.font.monoBold,
    fontSize: theme.size.base,
    letterSpacing: 1,
  },
  btnLabelPrimary: { color: theme.color.onBrandPrimary },
  btnLabelOutline: { color: theme.color.onSurface },

  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: theme.space.xl,
    marginBottom: theme.space.md,
  },
  sectionHeaderText: {
    fontFamily: theme.font.monoBold,
    fontSize: theme.size.sm,
    letterSpacing: 2,
    color: theme.color.onSurface,
  },
  sectionRule: {
    flex: 1,
    height: 2,
    backgroundColor: theme.color.borderStrong,
    marginLeft: theme.space.md,
  },

  fieldLabel: {
    fontFamily: theme.font.monoBold,
    fontSize: theme.size.sm,
    letterSpacing: 1.5,
    color: theme.color.onSurface,
    marginBottom: theme.space.sm,
  },

  segmented: {
    flexDirection: 'row',
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
  },
  segmentedItem: {
    flex: 1,
    paddingVertical: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.color.surface,
  },
  segmentedItemBordered: {
    borderLeftWidth: theme.border.thick,
    borderLeftColor: theme.color.borderStrong,
  },
  segmentedItemActive: {
    backgroundColor: theme.color.brandPrimary,
  },
  segmentedLabel: {
    fontFamily: theme.font.monoBold,
    fontSize: theme.size.sm,
    letterSpacing: 1.5,
    color: theme.color.onSurface,
  },
  segmentedLabelActive: {
    color: theme.color.onBrandPrimary,
  },

  toggleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 16,
    paddingHorizontal: theme.space.lg,
    borderBottomWidth: theme.border.thick,
    borderBottomColor: theme.color.borderStrong,
    backgroundColor: theme.color.surface,
  },
  toggleLabel: {
    flex: 1,
    fontFamily: theme.font.mono,
    fontSize: theme.size.base,
    color: theme.color.onSurface,
    marginRight: theme.space.md,
  },
  toggleBox: {
    width: 32,
    height: 32,
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
    backgroundColor: theme.color.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  toggleBoxActive: {
    backgroundColor: theme.color.brandPrimary,
  },
  toggleDot: {
    width: 14,
    height: 14,
    backgroundColor: theme.color.onBrandPrimary,
  },
});
