import React, { useState } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TextInput, View, Pressable } from 'react-native';
import { useRouter } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';

import { theme } from '@/src/theme';
import { api } from '@/src/api';
import { PrimaryButton, FieldLabel, Segmented } from '@/src/ui';

const REGIONS = ['US', 'GB', 'CA', 'AU', 'DE'] as const;

export default function PhoneScanScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const [phone, setPhone] = useState('');
  const [region, setRegion] = useState<(typeof REGIONS)[number]>('US');
  const [claimed, setClaimed] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!phone.trim()) {
      setError('Enter a phone number to run this scan.');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const result = await api.scanPhone({
        phone_number: phone,
        default_region: region,
        claimed_location: claimed || null,
      });
      await api.saveScan({
        scan_type: 'phone',
        input_summary: `${result.digits.slice(-4).padStart(result.digits.length, '•')} · ${region}`,
        result,
        overall_risk: result.flags.length > 1 ? 'MEDIUM' : result.flags.length === 1 ? 'MEDIUM' : result.valid ? 'LOW' : 'HIGH',
      }).catch(() => {});
      router.push({ pathname: '/results', params: { data: JSON.stringify(result), kind: 'phone' } });
    } catch (e: any) {
      setError(e.message ?? 'Scan failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <View style={[styles.root, { paddingTop: insets.top }]}>
        <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
          <Pressable testID="phone-scan-back" onPress={() => router.back()} style={styles.backBtn}>
            <MaterialCommunityIcons name="arrow-left" size={22} color={theme.color.onSurface} />
            <Text style={styles.backText}>BACK</Text>
          </Pressable>

          <Text style={styles.code}>MODULE / 04</Text>
          <Text style={styles.title}>PHONE LOOKUP</Text>
          <Text style={styles.body}>
            Validate a phone number and flag mismatches between area code and claimed location.
          </Text>

          <View style={{ height: theme.space.xl }} />

          <FieldLabel>Phone Number</FieldLabel>
          <TextInput
            testID="phone-input-number"
            style={styles.input}
            value={phone}
            onChangeText={setPhone}
            keyboardType="phone-pad"
            placeholder="+1 555 555 0100"
            placeholderTextColor={theme.color.muted}
          />

          <View style={{ height: theme.space.lg }} />

          <FieldLabel>Default Region</FieldLabel>
          <Segmented testID="phone-region" value={region} options={REGIONS} onChange={setRegion} />

          <View style={{ height: theme.space.lg }} />

          <FieldLabel>Claimed Location (optional)</FieldLabel>
          <TextInput
            testID="phone-input-claimed"
            style={styles.input}
            value={claimed}
            onChangeText={setClaimed}
            placeholder="e.g. Chicago"
            placeholderTextColor={theme.color.muted}
          />

          {error ? (
            <View style={styles.errorBox} testID="phone-scan-error">
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}
        </ScrollView>

        <View style={[styles.footer, { paddingBottom: Math.max(insets.bottom, theme.space.md) }]}>
          <PrimaryButton
            testID="phone-scan-run"
            label={loading ? 'LOOKING UP...' : 'CHECK NUMBER'}
            onPress={run}
            disabled={loading}
          />
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: theme.color.surface },
  scroll: { paddingHorizontal: theme.space.lg, paddingBottom: theme.space.xxxl },
  backBtn: { flexDirection: 'row', alignItems: 'center', paddingVertical: theme.space.md },
  backText: { fontFamily: theme.font.monoBold, fontSize: 10, letterSpacing: 2, color: theme.color.onSurface, marginLeft: theme.space.xs },
  code: { fontFamily: theme.font.monoBold, fontSize: 10, letterSpacing: 2, color: theme.color.muted, marginTop: theme.space.sm },
  title: { fontFamily: theme.font.display, fontSize: 32, letterSpacing: -1, color: theme.color.onSurface, marginTop: theme.space.xs, marginBottom: theme.space.md },
  body: { fontFamily: theme.font.mono, fontSize: theme.size.sm, lineHeight: 20, color: theme.color.onSurface },
  input: {
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
    backgroundColor: theme.color.surfaceSecondary,
    paddingHorizontal: theme.space.md,
    paddingVertical: 14,
    fontFamily: theme.font.mono,
    fontSize: theme.size.base,
    color: theme.color.onSurface,
  },
  errorBox: {
    marginTop: theme.space.md,
    borderWidth: theme.border.thick,
    borderColor: theme.color.error,
    padding: theme.space.md,
  },
  errorText: { fontFamily: theme.font.monoBold, color: theme.color.error },
  footer: {
    borderTopWidth: theme.border.thick,
    borderTopColor: theme.color.borderStrong,
    paddingHorizontal: theme.space.lg,
    paddingTop: theme.space.md,
    backgroundColor: theme.color.surface,
  },
});
