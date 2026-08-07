import React, { useState } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TextInput, View, Pressable } from 'react-native';
import { useRouter } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';

import { theme } from '@/src/theme';
import { api } from '@/src/api';
import { PrimaryButton, FieldLabel, ToggleRow } from '@/src/ui';

type Field = {
  key: 'follower_count' | 'following_count' | 'post_count' | 'photo_count' | 'account_age_days';
  label: string;
};

const NUMBER_FIELDS: Field[] = [
  { key: 'follower_count', label: 'Follower Count' },
  { key: 'following_count', label: 'Following Count' },
  { key: 'post_count', label: 'Post Count' },
  { key: 'photo_count', label: 'Photo Count' },
  { key: 'account_age_days', label: 'Account Age (days)' },
];

export default function ProfileScanScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const [nums, setNums] = useState<Record<string, string>>({});
  const [bio, setBio] = useState('');
  const [contactedFirst, setContactedFirst] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setError(null);
    setLoading(true);
    try {
      const body: Record<string, any> = { bio: bio || null, they_contacted_first: contactedFirst };
      for (const f of NUMBER_FIELDS) {
        const raw = nums[f.key];
        body[f.key] = raw && raw.trim() ? parseInt(raw, 10) : null;
      }
      const result = await api.scanProfile(body);
      await api.saveScan({
        scan_type: 'profile',
        input_summary: `${result.signal_count} signal(s), concern=${result.profile_concern}`,
        result,
        overall_risk: result.profile_concern,
      }).catch(() => {});
      router.push({ pathname: '/results', params: { data: JSON.stringify(result), kind: 'profile' } });
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
          <Pressable testID="profile-scan-back" onPress={() => router.back()} style={styles.backBtn}>
            <MaterialCommunityIcons name="arrow-left" size={22} color={theme.color.onSurface} />
            <Text style={styles.backText}>BACK</Text>
          </Pressable>

          <Text style={styles.code}>MODULE / 03</Text>
          <Text style={styles.title}>PROFILE CHECK</Text>
          <Text style={styles.body}>
            Enter observable profile metrics. Leave a field blank to skip it. Signals include
            thin bios, low follower counts, high follow ratios, and new accounts.
          </Text>

          <View style={{ height: theme.space.xl }} />

          {NUMBER_FIELDS.map((f) => (
            <View key={f.key} style={styles.field}>
              <FieldLabel>{f.label}</FieldLabel>
              <TextInput
                testID={`profile-input-${f.key}`}
                style={styles.input}
                value={nums[f.key] ?? ''}
                onChangeText={(v) => setNums((s) => ({ ...s, [f.key]: v.replace(/[^0-9]/g, '') }))}
                keyboardType="number-pad"
                placeholder="—"
                placeholderTextColor={theme.color.muted}
              />
            </View>
          ))}

          <View style={styles.field}>
            <FieldLabel>Bio Text</FieldLabel>
            <TextInput
              testID="profile-input-bio"
              style={[styles.input, { minHeight: 90 }]}
              value={bio}
              onChangeText={setBio}
              multiline
              placeholder="Paste bio text..."
              placeholderTextColor={theme.color.muted}
              textAlignVertical="top"
            />
          </View>

          <View style={styles.toggleWrap}>
            <ToggleRow
              testID="profile-contacted-first"
              label="They contacted me first"
              value={contactedFirst}
              onChange={setContactedFirst}
            />
          </View>

          {error ? (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}
        </ScrollView>

        <View style={[styles.footer, { paddingBottom: Math.max(insets.bottom, theme.space.md) }]}>
          <PrimaryButton
            testID="profile-scan-run"
            label={loading ? 'CALCULATING...' : 'RUN PROFILE CHECK'}
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
  field: { marginBottom: theme.space.lg },
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
  toggleWrap: {
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
    marginTop: theme.space.sm,
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
