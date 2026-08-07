import React from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useRouter } from 'expo-router';
import { Image } from 'expo-image';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import * as Haptics from 'expo-haptics';

import { theme } from '@/src/theme';

const HERO_IMAGE =
  'https://images.unsplash.com/photo-1644088379091-d574269d422f?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NDk1ODF8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMGRhdGElMjBhbmFseXNpcyUyMG1pbmltYWx8ZW58MHx8fHwxNzg2MDgzODQ2fDA&ixlib=rb-4.1.0&q=85';

type Tile = {
  key: string;
  code: string;
  label: string;
  route: string;
  icon: React.ComponentProps<typeof MaterialCommunityIcons>['name'];
  wide?: boolean;
};

const TILES: Tile[] = [
  { key: 'text', code: '01', label: 'Text\nAnalysis', route: '/text-scan', icon: 'message-text-outline' },
  { key: 'photo', code: '02', label: 'Photo\nScan', route: '/photo-scan', icon: 'image-search-outline' },
  { key: 'profile', code: '03', label: 'Profile\nCheck', route: '/profile-scan', icon: 'account-search-outline' },
  { key: 'phone', code: '04', label: 'Phone\nLookup', route: '/phone-scan', icon: 'phone-alert-outline' },
  { key: 'full', code: '05', label: 'Comprehensive Full Scan', route: '/full-scan', icon: 'shield-search', wide: true },
];

export default function Dashboard() {
  const router = useRouter();
  const insets = useSafeAreaInsets();

  return (
    <View style={styles.root}>
      <ScrollView
        contentContainerStyle={[
          styles.scroll,
          { paddingTop: insets.top + theme.space.md, paddingBottom: theme.space.xxxl },
        ]}
        showsVerticalScrollIndicator={false}
      >
        {/* Header stack */}
        <View style={styles.headerRow} testID="dashboard-header">
          <View style={styles.headerStatus}>
            <View style={styles.headerDot} />
            <Text style={styles.headerStatusText}>ENGINE ONLINE</Text>
          </View>
          <Text style={styles.headerVersion}>v1.0</Text>
        </View>

        <Text style={styles.titleTop}>PROJECT AI</Text>
        <Text style={styles.titleBottom}>CORE SCANNER</Text>

        <View style={styles.heroFrame}>
          <Image source={{ uri: HERO_IMAGE }} style={styles.heroImage} contentFit="cover" />
          <View style={styles.heroCaption}>
            <Text style={styles.heroCaptionText}>HARDENED CORE / RULE ENGINE</Text>
          </View>
        </View>

        <View style={styles.sectionHeader}>
          <Text style={styles.sectionText}>SCAN MODULES</Text>
          <View style={styles.rule} />
        </View>

        <View style={styles.grid}>
          {TILES.map((tile) => (
            <Pressable
              key={tile.key}
              testID={`tile-${tile.key}`}
              onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
                router.push(tile.route as any);
              }}
              style={({ pressed }) => [
                styles.tile,
                tile.wide && styles.tileWide,
                pressed && styles.tilePressed,
              ]}
            >
              <View style={styles.tileHeader}>
                <Text style={styles.tileCode}>{tile.code}</Text>
                <MaterialCommunityIcons name={tile.icon} size={20} color={theme.color.onSurface} />
              </View>
              <Text style={[styles.tileLabel, tile.wide && styles.tileLabelWide]}>{tile.label}</Text>
              <View style={styles.tileArrow}>
                <Text style={styles.tileArrowText}>→</Text>
              </View>
            </Pressable>
          ))}
        </View>

        <View style={styles.notice} testID="dashboard-notice">
          <Text style={styles.noticeTitle}>LOCAL / PRIVATE</Text>
          <Text style={styles.noticeBody}>
            Only scan hashes and verdicts are recorded. Raw chat text, photos, and metadata are never stored.
          </Text>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: theme.color.surface },
  scroll: { paddingHorizontal: theme.space.lg },

  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: theme.space.md,
  },
  headerStatus: { flexDirection: 'row', alignItems: 'center' },
  headerDot: {
    width: 10,
    height: 10,
    backgroundColor: theme.color.success,
    borderWidth: 1,
    borderColor: theme.color.borderStrong,
    marginRight: theme.space.sm,
  },
  headerStatusText: {
    fontFamily: theme.font.monoBold,
    fontSize: 10,
    letterSpacing: 2,
    color: theme.color.onSurface,
  },
  headerVersion: {
    fontFamily: theme.font.mono,
    fontSize: 10,
    letterSpacing: 2,
    color: theme.color.muted,
  },

  titleTop: {
    fontFamily: theme.font.display,
    fontSize: 34,
    letterSpacing: -1,
    color: theme.color.onSurface,
    lineHeight: 36,
  },
  titleBottom: {
    fontFamily: theme.font.display,
    fontSize: 34,
    letterSpacing: -1,
    color: theme.color.onSurface,
    lineHeight: 36,
    marginBottom: theme.space.lg,
  },

  heroFrame: {
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
    marginBottom: theme.space.md,
    backgroundColor: theme.color.surfaceSecondary,
  },
  heroImage: { width: '100%', height: 160 },
  heroCaption: {
    borderTopWidth: theme.border.thick,
    borderTopColor: theme.color.borderStrong,
    paddingVertical: theme.space.sm,
    paddingHorizontal: theme.space.md,
    backgroundColor: theme.color.surfaceInverse,
  },
  heroCaptionText: {
    fontFamily: theme.font.monoBold,
    fontSize: 10,
    letterSpacing: 2,
    color: theme.color.onSurfaceInverse,
  },

  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: theme.space.lg,
    marginBottom: theme.space.md,
  },
  sectionText: {
    fontFamily: theme.font.monoBold,
    fontSize: 10,
    letterSpacing: 2,
    color: theme.color.onSurface,
  },
  rule: { flex: 1, height: 2, backgroundColor: theme.color.borderStrong, marginLeft: theme.space.md },

  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: theme.space.md as unknown as number,
  },
  tile: {
    width: '48%',
    aspectRatio: 1,
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
    backgroundColor: theme.color.surface,
    padding: theme.space.md,
    justifyContent: 'space-between',
  },
  tileWide: {
    width: '100%',
    aspectRatio: undefined,
    minHeight: 120,
    backgroundColor: theme.color.brandPrimary,
  },
  tilePressed: { backgroundColor: theme.color.surfaceSecondary },
  tileHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  tileCode: {
    fontFamily: theme.font.monoBold,
    fontSize: 10,
    letterSpacing: 2,
    color: theme.color.muted,
  },
  tileLabel: {
    fontFamily: theme.font.display,
    fontSize: 20,
    lineHeight: 22,
    color: theme.color.onSurface,
    letterSpacing: -0.5,
  },
  tileLabelWide: { color: theme.color.onBrandPrimary, fontSize: 24, lineHeight: 26 },
  tileArrow: { alignSelf: 'flex-end' },
  tileArrowText: {
    fontFamily: theme.font.monoBold,
    fontSize: 20,
    color: theme.color.onSurface,
  },

  notice: {
    marginTop: theme.space.xl,
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
    padding: theme.space.md,
    backgroundColor: theme.color.surfaceSecondary,
  },
  noticeTitle: {
    fontFamily: theme.font.monoBold,
    fontSize: 10,
    letterSpacing: 2,
    color: theme.color.onSurface,
    marginBottom: theme.space.xs,
  },
  noticeBody: {
    fontFamily: theme.font.mono,
    fontSize: theme.size.sm,
    lineHeight: 18,
    color: theme.color.onSurface,
  },
});
