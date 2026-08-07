import React, { useState } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TextInput, View, Pressable } from 'react-native';
import { useRouter } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';

import { theme } from '@/src/theme';
import { api } from '@/src/api';
import { PrimaryButton, FieldLabel, ToggleRow, Segmented, SectionHeader } from '@/src/ui';

const FLAGS = [
  { key: 'missing_or_wrong_contact_shadow', label: 'Missing / inconsistent contact shadow' },
  { key: 'glasses_frame_issue', label: 'Glasses frame distortion' },
  { key: 'lighting_direction_mismatch', label: 'Lighting direction mismatch' },
  { key: 'unnatural_skin_texture', label: 'Unnatural skin texture' },
  { key: 'edge_or_hair_blending_failure', label: 'Edge / hair blending failure' },
  { key: 'mismatched_eye_catchlights', label: 'Mismatched eye catchlights' },
  { key: 'detached_or_too_clean_background', label: 'Detached / too-clean background' },
] as const;

const QUALITIES = ['good', 'partial', 'poor'] as const;
const REGIONS = ['US', 'GB', 'CA', 'AU', 'DE'] as const;

const NUM_FIELDS: { key: string; label: string }[] = [
  { key: 'follower_count', label: 'Followers' },
  { key: 'following_count', label: 'Following' },
  { key: 'post_count', label: 'Posts' },
  { key: 'photo_count', label: 'Photos' },
  { key: 'account_age_days', label: 'Age (days)' },
];

export default function FullScanScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();

  const [chat, setChat] = useState('');
  const [profileText, setProfileText] = useState('');
  const [flags, setFlags] = useState<Record<string, boolean>>({});
  const [quality, setQuality] = useState<(typeof QUALITIES)[number]>('good');
  const [nums, setNums] = useState<Record<string, string>>({});
  const [bio, setBio] = useState('');
  const [contactedFirst, setContactedFirst] = useState(false);
  const [phone, setPhone] = useState('');
  const [region, setRegion] = useState<(typeof REGIONS)[number]>('US');
  const [claimed, setClaimed] = useState('');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setError(null);
    setLoading(true);
    try {
      const body: Record<string, any> = {
        chat_messages: chat,
        profile_text: profileText || null,
        observation_quality: quality,
        bio: bio || null,
        they_contacted_first: contactedFirst,
        phone_number: phone || null,
        default_region: region,
        claimed_location: claimed || null,
      };
      for (const f of FLAGS) body[f.key] = !!flags[f.key];
      for (const f of NUM_FIELDS) {
        const raw = nums[f.key];
        body[f.key] = raw && raw.trim() ? parseInt(raw, 10) : null;
      }
      const result = await api.scanFull(body);
      await api.saveScan({
        scan_type: 'full',
        input_summary: `Overall=${result.overall_risk} · text=${result.text.risk_level} · profile=${result.profile.profile_concern}`,
        result,
        overall_risk: result.overall_risk,
      }).catch(() => {});
      router.push({ pathname: '/results', params: { data: JSON.stringify(result), kind: 'full' } });
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
          <Pressable testID="full-scan-back" onPress={() => router.back()} style={styles.backBtn}>
            <MaterialCommunityIcons name="arrow-left" size={22} color={theme.color.onSurface} />
            <Text style={styles.backText}>BACK</Text>
          </Pressable>

          <Text style={styles.code}>MODULE / 05</Text>
          <Text style={styles.title}>FULL SCAN</Text>
          <Text style={styles.body}>
            All modules in one pass. Fill in what you know — every field is optional except when
            you want a phone lookup.
          </Text>

          <SectionHeader label="TEXT" />
          <FieldLabel>Chat Messages</FieldLabel>
          <TextInput
            testID="full-chat"
            style={styles.textArea}
            value={chat}
            onChangeText={setChat}
            multiline
            placeholder="Paste chat text..."
            placeholderTextColor={theme.color.muted}
            textAlignVertical="top"
          />

          <SectionHeader label="PHOTO" />
          <FieldLabel>Observation Quality</FieldLabel>
          <Segmented testID="full-quality" value={quality} options={QUALITIES} onChange={setQuality} />
          <View style={{ height: theme.space.md }} />
          <View style={styles.flagList}>
            {FLAGS.map((f) => (
              <ToggleRow
                key={f.key}
                testID={`full-flag-${f.key}`}
                label={f.label}
                value={!!flags[f.key]}
                onChange={(v) => setFlags((s) => ({ ...s, [f.key]: v }))}
              />
            ))}
          </View>

          <SectionHeader label="PROFILE" />
          <View style={styles.numGrid}>
            {NUM_FIELDS.map((f) => (
              <View key={f.key} style={styles.numCell}>
                <FieldLabel>{f.label}</FieldLabel>
                <TextInput
                  testID={`full-num-${f.key}`}
                  style={styles.input}
                  value={nums[f.key] ?? ''}
                  onChangeText={(v) => setNums((s) => ({ ...s, [f.key]: v.replace(/[^0-9]/g, '') }))}
                  keyboardType="number-pad"
                  placeholder="—"
                  placeholderTextColor={theme.color.muted}
                />
              </View>
            ))}
          </View>
          <FieldLabel>Bio</FieldLabel>
          <TextInput
            testID="full-bio"
            style={[styles.input, { minHeight: 80 }]}
            value={bio}
            onChangeText={setBio}
            multiline
            placeholder="Bio text..."
            placeholderTextColor={theme.color.muted}
            textAlignVertical="top"
          />
          <View style={{ height: theme.space.md }} />
          <View style={{ borderWidth: theme.border.thick, borderColor: theme.color.borderStrong }}>
            <ToggleRow
              testID="full-contacted-first"
              label="They contacted me first"
              value={contactedFirst}
              onChange={setContactedFirst}
            />
          </View>

          <SectionHeader label="PHONE" />
          <FieldLabel>Phone Number (optional)</FieldLabel>
          <TextInput
            testID="full-phone"
            style={styles.input}
            value={phone}
            onChangeText={setPhone}
            keyboardType="phone-pad"
            placeholder="+1 ..."
            placeholderTextColor={theme.color.muted}
          />
          <View style={{ height: theme.space.md }} />
          <FieldLabel>Region</FieldLabel>
          <Segmented testID="full-region" value={region} options={REGIONS} onChange={setRegion} />
          <View style={{ height: theme.space.md }} />
          <FieldLabel>Claimed Location</FieldLabel>
          <TextInput
            testID="full-claimed"
            style={styles.input}
            value={claimed}
            onChangeText={setClaimed}
            placeholder="e.g. Miami"
            placeholderTextColor={theme.color.muted}
          />

          {error ? (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}
        </ScrollView>

        <View style={[styles.footer, { paddingBottom: Math.max(insets.bottom, theme.space.md) }]}>
          <PrimaryButton
            testID="full-scan-run"
            label={loading ? 'RUNNING FULL SCAN...' : 'RUN FULL SCAN'}
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

  textArea: {
    minHeight: 120,
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
    backgroundColor: theme.color.surfaceSecondary,
    padding: theme.space.md,
    fontFamily: theme.font.mono,
    fontSize: theme.size.base,
    color: theme.color.onSurface,
  },
  flagList: { borderWidth: theme.border.thick, borderColor: theme.color.borderStrong },
  numGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: theme.space.sm as unknown as number, marginBottom: theme.space.md },
  numCell: { width: '48%' },
  input: {
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
    backgroundColor: theme.color.surfaceSecondary,
    paddingHorizontal: theme.space.md,
    paddingVertical: 12,
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
