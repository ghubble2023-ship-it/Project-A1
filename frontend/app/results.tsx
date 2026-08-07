import React, { useMemo } from 'react';
import { ScrollView, StyleSheet, Text, View, Pressable } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';

import { theme, riskColor } from '@/src/theme';

type Kind = 'text' | 'photo' | 'profile' | 'phone' | 'full';

function fireHaptic(risk: string) {
  const r = (risk || '').toUpperCase();
  if (r === 'HIGH' || r === 'ELEVATED')
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => {});
  else if (r === 'MEDIUM')
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning).catch(() => {});
  else Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
}

function EvidenceBlock({ label, items }: { label: string; items: string[] }) {
  if (!items || items.length === 0) return null;
  return (
    <View style={styles.evidenceBlock}>
      <Text style={styles.evidenceLabel}>{label.toUpperCase()}</Text>
      {items.map((s, i) => (
        <View key={`${label}-${i}`} style={styles.evidenceRow}>
          <Text style={styles.evidenceBullet}>▪</Text>
          <Text style={styles.evidenceText}>{s}</Text>
        </View>
      ))}
    </View>
  );
}

function KeyValue({ k, v }: { k: string; v: string }) {
  return (
    <View style={styles.kvRow}>
      <Text style={styles.kvKey}>{k}</Text>
      <Text style={styles.kvVal}>{v}</Text>
    </View>
  );
}

function TextResultView({ result }: { result: any }) {
  const cats = result.fired_categories ?? [];
  const matches = result.matched_phrases ?? {};
  const catRows: string[] = [];
  for (const c of cats) {
    const phrases = matches[c] ?? [];
    catRows.push(`${c.replace(/_/g, ' ')} — ${phrases.join(', ')}`);
  }
  return (
    <>
      <KeyValue k="RISK LEVEL" v={result.risk_level} />
      <KeyValue k="CATEGORIES" v={String(result.category_count)} />
      <KeyValue k="MONEY REQUEST" v={result.money_request_present ? 'PRESENT' : 'ABSENT'} />
      <EvidenceBlock label="Fired categories" items={catRows} />
    </>
  );
}

function PhotoResultView({ result }: { result: any }) {
  return (
    <>
      <KeyValue k="FLAGS TRIGGERED" v={String(result.flag_count)} />
      <KeyValue k="OBSERVATION" v={String(result.observation_quality).toUpperCase()} />
      <KeyValue k="STATUS" v={result.photo_status} />
      <EvidenceBlock label="Visual anomalies" items={result.flags_triggered ?? []} />
    </>
  );
}

function ProfileResultView({ result }: { result: any }) {
  return (
    <>
      <KeyValue k="CONCERN" v={result.profile_concern} />
      <KeyValue k="SIGNAL COUNT" v={String(result.signal_count)} />
      <EvidenceBlock label="Signals fired" items={result.signals_fired ?? []} />
    </>
  );
}

function PhoneResultView({ result }: { result: any }) {
  return (
    <>
      <KeyValue k="VALID" v={result.valid ? 'YES' : 'NO'} />
      <KeyValue k="REGION" v={result.region} />
      <KeyValue k="AREA CODE" v={result.inferred_area_code ?? '—'} />
      <KeyValue k="SUMMARY" v={result.summary} />
      <EvidenceBlock label="Flags" items={result.flags ?? []} />
    </>
  );
}

function FullResultView({ result }: { result: any }) {
  return (
    <>
      <KeyValue k="ENGINE" v={result.engine} />
      <KeyValue k="TEXT RISK" v={result.text?.risk_level ?? '—'} />
      <KeyValue k="PROFILE CONCERN" v={result.profile?.profile_concern ?? '—'} />
      <KeyValue k="PHOTO FLAGS" v={String(result.photo?.flag_count ?? 0)} />
      <KeyValue k="PHONE" v={result.phone ? (result.phone.valid ? 'VALID' : 'INVALID') : '—'} />
      <EvidenceBlock label="Text categories" items={result.text?.fired_categories ?? []} />
      <EvidenceBlock label="Visual anomalies" items={result.photo?.flags_triggered ?? []} />
      <EvidenceBlock label="Profile signals" items={result.profile?.signals_fired ?? []} />
      <EvidenceBlock label="Phone flags" items={result.phone?.flags ?? []} />
    </>
  );
}

function primaryRisk(result: any, kind: Kind): string {
  if (kind === 'text') return result.risk_level ?? 'N/A';
  if (kind === 'profile') return result.profile_concern ?? 'N/A';
  if (kind === 'phone') return result.valid ? (result.flags?.length ? 'MEDIUM' : 'LOW') : 'HIGH';
  if (kind === 'photo') {
    if (result.flag_count === 0) return 'LOW';
    if (result.observation_quality === 'poor') return 'MEDIUM';
    if (result.flag_count >= 3) return 'HIGH';
    return 'MEDIUM';
  }
  return result.overall_risk ?? 'N/A';
}

export default function ResultsScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const params = useLocalSearchParams<{ data?: string; kind?: string }>();
  const result = useMemo(() => {
    try {
      return params.data ? JSON.parse(String(params.data)) : {};
    } catch {
      return {};
    }
  }, [params.data]);

  const kind = (params.kind ?? 'text') as Kind;
  const risk = primaryRisk(result, kind);
  const c = riskColor(risk);

  React.useEffect(() => {
    fireHaptic(risk);
  }, [risk]);

  const kindLabel = {
    text: 'TEXT ANALYSIS',
    photo: 'PHOTO SCAN',
    profile: 'PROFILE CHECK',
    phone: 'PHONE LOOKUP',
    full: 'FULL SCAN',
  }[kind];

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <ScrollView contentContainerStyle={{ paddingBottom: theme.space.xxxl }}>
        <Pressable testID="results-back" onPress={() => router.back()} style={styles.backBtn}>
          <MaterialCommunityIcons name="arrow-left" size={22} color={theme.color.onSurface} />
          <Text style={styles.backText}>BACK</Text>
        </Pressable>

        <View style={[styles.heroBlock, { backgroundColor: c.bg }]} testID="results-hero">
          <Text style={[styles.heroLabel, { color: c.fg }]}>REPORT / {kindLabel}</Text>
          <Text style={[styles.heroRisk, { color: c.fg }]}>{risk}</Text>
          <Text style={[styles.heroSub, { color: c.fg }]}>RISK LEVEL</Text>
        </View>

        <View style={styles.body}>
          {kind === 'text' && <TextResultView result={result} />}
          {kind === 'photo' && <PhotoResultView result={result} />}
          {kind === 'profile' && <ProfileResultView result={result} />}
          {kind === 'phone' && <PhoneResultView result={result} />}
          {kind === 'full' && <FullResultView result={result} />}

          <View style={styles.privacyNote}>
            <Text style={styles.privacyTitle}>PRIVACY NOTE</Text>
            <Text style={styles.privacyBody}>
              Raw content is not retained. Only the scan verdict and evidence categories are
              logged in your local archive.
            </Text>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: theme.color.surface },
  backBtn: { flexDirection: 'row', alignItems: 'center', padding: theme.space.lg, paddingBottom: theme.space.sm },
  backText: { fontFamily: theme.font.monoBold, fontSize: 10, letterSpacing: 2, color: theme.color.onSurface, marginLeft: theme.space.xs },

  heroBlock: {
    padding: theme.space.xl,
    borderBottomWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
    marginBottom: 0,
  },
  heroLabel: { fontFamily: theme.font.monoBold, fontSize: 10, letterSpacing: 2, marginBottom: theme.space.md },
  heroRisk: { fontFamily: theme.font.display, fontSize: 72, lineHeight: 74, letterSpacing: -3 },
  heroSub: { fontFamily: theme.font.monoBold, fontSize: 12, letterSpacing: 2, marginTop: theme.space.xs },

  body: { paddingHorizontal: theme.space.lg, paddingTop: theme.space.lg },

  kvRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    borderBottomWidth: theme.border.thin,
    borderBottomColor: theme.color.borderStrong,
    paddingVertical: theme.space.sm,
  },
  kvKey: {
    fontFamily: theme.font.monoBold,
    fontSize: 11,
    letterSpacing: 2,
    color: theme.color.muted,
  },
  kvVal: {
    fontFamily: theme.font.monoBold,
    fontSize: theme.size.sm,
    color: theme.color.onSurface,
    flexShrink: 1,
    textAlign: 'right',
    marginLeft: theme.space.md,
  },

  evidenceBlock: {
    marginTop: theme.space.lg,
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
    padding: theme.space.md,
    backgroundColor: theme.color.surfaceSecondary,
  },
  evidenceLabel: {
    fontFamily: theme.font.monoBold,
    fontSize: 10,
    letterSpacing: 2,
    color: theme.color.onSurface,
    marginBottom: theme.space.sm,
  },
  evidenceRow: { flexDirection: 'row', marginBottom: theme.space.xs },
  evidenceBullet: { fontFamily: theme.font.monoBold, color: theme.color.onSurface, marginRight: theme.space.sm },
  evidenceText: {
    fontFamily: theme.font.mono,
    fontSize: theme.size.sm,
    color: theme.color.onSurface,
    lineHeight: 20,
    flexShrink: 1,
  },

  privacyNote: {
    marginTop: theme.space.xl,
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
    padding: theme.space.md,
  },
  privacyTitle: {
    fontFamily: theme.font.monoBold,
    fontSize: 10,
    letterSpacing: 2,
    color: theme.color.onSurface,
    marginBottom: theme.space.xs,
  },
  privacyBody: {
    fontFamily: theme.font.mono,
    fontSize: theme.size.sm,
    color: theme.color.onSurface,
    lineHeight: 18,
  },
});
