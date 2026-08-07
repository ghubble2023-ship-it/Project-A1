import React, { useState } from 'react';
import { ScrollView, StyleSheet, Text, View, Pressable } from 'react-native';
import { useRouter } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';

import { theme } from '@/src/theme';
import { api } from '@/src/api';
import { PrimaryButton, FieldLabel, ToggleRow, Segmented } from '@/src/ui';

const FLAGS = [
  { key: 'missing_or_wrong_contact_shadow', label: 'Missing or inconsistent contact shadow' },
  { key: 'glasses_frame_issue', label: 'Glasses frame distortion / asymmetry' },
  { key: 'lighting_direction_mismatch', label: 'Lighting direction mismatch' },
  { key: 'unnatural_skin_texture', label: 'Unnaturally smooth / plastic skin texture' },
  { key: 'edge_or_hair_blending_failure', label: 'Edge or hair blending failure' },
  { key: 'mismatched_eye_catchlights', label: 'Mismatched eye catchlights' },
  { key: 'detached_or_too_clean_background', label: 'Detached or too-clean background' },
] as const;

const QUALITIES = ['good', 'partial', 'poor'] as const;

type FlagKey = typeof FLAGS[number]['key'];

export default function PhotoScanScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const [flags, setFlags] = useState<Record<FlagKey, boolean>>(
    Object.fromEntries(FLAGS.map((f) => [f.key, false])) as Record<FlagKey, boolean>
  );
  const [quality, setQuality] = useState<(typeof QUALITIES)[number]>('good');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setError(null);
    setLoading(true);
    try {
      const body = { ...flags, observation_quality: quality };
      const result = await api.scanPhoto(body);
      await api.saveScan({
        scan_type: 'photo',
        input_summary: `${result.flag_count} flag(s), quality=${quality}`,
        result,
        overall_risk: result.flag_count >= 3 && quality !== 'poor' ? 'HIGH' : result.flag_count > 0 ? 'MEDIUM' : 'LOW',
      }).catch(() => {});
      router.push({ pathname: '/results', params: { data: JSON.stringify(result), kind: 'photo' } });
    } catch (e: any) {
      setError(e.message ?? 'Scan failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <Pressable testID="photo-scan-back" onPress={() => router.back()} style={styles.backBtn}>
          <MaterialCommunityIcons name="arrow-left" size={22} color={theme.color.onSurface} />
          <Text style={styles.backText}>BACK</Text>
        </Pressable>

        <Text style={styles.code}>MODULE / 02</Text>
        <Text style={styles.title}>PHOTO SCAN</Text>
        <Text style={styles.body}>
          Toggle each visual manipulation flag observed in the subject photo. Set observation
          quality — poor image quality tempers the verdict.
        </Text>

        <View style={{ height: theme.space.xl }} />

        <FieldLabel>Observation Quality</FieldLabel>
        <Segmented
          testID="photo-scan-quality"
          value={quality}
          options={QUALITIES}
          onChange={setQuality}
        />

        <View style={{ height: theme.space.xl }} />

        <FieldLabel>Manipulation Flags</FieldLabel>

        <View style={styles.flagList}>
          {FLAGS.map((f) => (
            <ToggleRow
              key={f.key}
              testID={`photo-flag-${f.key}`}
              label={f.label}
              value={flags[f.key]}
              onChange={(v) => setFlags((s) => ({ ...s, [f.key]: v }))}
            />
          ))}
        </View>

        {error ? (
          <View style={styles.errorBox} testID="photo-scan-error">
            <Text style={styles.errorText}>{error}</Text>
          </View>
        ) : null}
      </ScrollView>

      <View style={[styles.footer, { paddingBottom: Math.max(insets.bottom, theme.space.md) }]}>
        <PrimaryButton
          testID="photo-scan-run"
          label={loading ? 'COMPILING...' : 'GENERATE VERDICT'}
          onPress={run}
          disabled={loading}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: theme.color.surface },
  scroll: { paddingHorizontal: theme.space.lg, paddingBottom: theme.space.xxxl },
  backBtn: { flexDirection: 'row', alignItems: 'center', paddingVertical: theme.space.md },
  backText: {
    fontFamily: theme.font.monoBold,
    fontSize: 10,
    letterSpacing: 2,
    color: theme.color.onSurface,
    marginLeft: theme.space.xs,
  },
  code: { fontFamily: theme.font.monoBold, fontSize: 10, letterSpacing: 2, color: theme.color.muted, marginTop: theme.space.sm },
  title: { fontFamily: theme.font.display, fontSize: 32, letterSpacing: -1, color: theme.color.onSurface, marginTop: theme.space.xs, marginBottom: theme.space.md },
  body: { fontFamily: theme.font.mono, fontSize: theme.size.sm, lineHeight: 20, color: theme.color.onSurface },

  flagList: {
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
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
