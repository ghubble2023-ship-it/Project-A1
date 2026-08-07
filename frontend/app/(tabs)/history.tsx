import React, { useCallback, useEffect, useState } from 'react';
import { FlatList, Pressable, StyleSheet, Text, View, RefreshControl } from 'react-native';
import { useRouter } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Image } from 'expo-image';
import { useFocusEffect } from 'expo-router';

import { theme, riskColor } from '@/src/theme';
import { api, ScanRecord } from '@/src/api';

const EMPTY_IMAGE =
  'https://images.unsplash.com/photo-1743796055664-3473eedab36e?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NDQ2NDN8MHwxfHNlYXJjaHwxfHxtYWduaWZ5aW5nJTIwZ2xhc3MlMjBtaW5pbWFsfGVufDB8fHx8MTc4NjA4Mzg0Nnww&ixlib=rb-4.1.0&q=85';

const KIND_LABEL: Record<string, string> = {
  text: 'TEXT',
  photo: 'PHOTO',
  profile: 'PROFILE',
  phone: 'PHONE',
  full: 'FULL',
};

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

export default function HistoryScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const [records, setRecords] = useState<ScanRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await api.listScans();
      setRecords(list);
    } catch (e: any) {
      setError(e.message ?? 'Failed to load history');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load])
  );

  const clearAll = async () => {
    try {
      await api.clearScans();
      setRecords([]);
    } catch (e: any) {
      setError(e.message ?? 'Clear failed');
    }
  };

  const renderItem = ({ item }: { item: ScanRecord }) => {
    const c = riskColor(item.overall_risk);
    return (
      <Pressable
        testID={`history-item-${item.id}`}
        style={styles.row}
        onPress={() =>
          router.push({
            pathname: '/results',
            params: { data: JSON.stringify(item.result), kind: item.scan_type },
          })
        }
      >
        <View style={[styles.riskStripe, { backgroundColor: c.bg }]} />
        <View style={styles.rowBody}>
          <View style={styles.rowTop}>
            <Text style={styles.rowKind}>{KIND_LABEL[item.scan_type] ?? item.scan_type.toUpperCase()}</Text>
            <Text style={styles.rowRisk}>{item.overall_risk}</Text>
          </View>
          <Text style={styles.rowSummary} numberOfLines={2}>
            {item.input_summary}
          </Text>
          <Text style={styles.rowDate}>{formatDate(item.created_at)}</Text>
        </View>
        <MaterialCommunityIcons name="chevron-right" size={22} color={theme.color.onSurface} />
      </Pressable>
    );
  };

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <View style={styles.header}>
        <View>
          <Text style={styles.headerCode}>ARCHIVE / 06</Text>
          <Text style={styles.headerTitle}>HISTORY</Text>
        </View>
        {records.length > 0 && (
          <Pressable testID="history-clear" onPress={clearAll} style={styles.clearBtn}>
            <Text style={styles.clearText}>CLEAR</Text>
          </Pressable>
        )}
      </View>

      {error ? (
        <View style={styles.errorBox} testID="history-error">
          <Text style={styles.errorText}>{error}</Text>
        </View>
      ) : null}

      <FlatList
        data={records}
        keyExtractor={(item) => item.id}
        renderItem={renderItem}
        contentContainerStyle={records.length === 0 ? styles.emptyWrap : { paddingBottom: theme.space.xxxl }}
        refreshControl={<RefreshControl refreshing={loading} onRefresh={load} tintColor={theme.color.onSurface} />}
        ListEmptyComponent={
          <View style={styles.emptyState} testID="history-empty">
            <Image source={{ uri: EMPTY_IMAGE }} style={styles.emptyImage} contentFit="cover" />
            <Text style={styles.emptyTitle}>NO PREVIOUS SCANS RECORDED</Text>
            <Text style={styles.emptyBody}>
              Run a scan from the Core dashboard. Your verdicts will archive here.
            </Text>
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: theme.color.surface },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    paddingHorizontal: theme.space.lg,
    paddingTop: theme.space.md,
    paddingBottom: theme.space.lg,
    borderBottomWidth: theme.border.thick,
    borderBottomColor: theme.color.borderStrong,
  },
  headerCode: { fontFamily: theme.font.monoBold, fontSize: 10, letterSpacing: 2, color: theme.color.muted },
  headerTitle: { fontFamily: theme.font.display, fontSize: 32, letterSpacing: -1, color: theme.color.onSurface },
  clearBtn: {
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
    paddingHorizontal: theme.space.md,
    paddingVertical: theme.space.sm,
  },
  clearText: {
    fontFamily: theme.font.monoBold,
    fontSize: 10,
    letterSpacing: 2,
    color: theme.color.onSurface,
  },

  row: {
    flexDirection: 'row',
    alignItems: 'center',
    borderBottomWidth: theme.border.thick,
    borderBottomColor: theme.color.borderStrong,
    backgroundColor: theme.color.surface,
  },
  riskStripe: { width: 8, alignSelf: 'stretch' },
  rowBody: { flex: 1, paddingVertical: theme.space.md, paddingHorizontal: theme.space.md },
  rowTop: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: theme.space.xs },
  rowKind: { fontFamily: theme.font.monoBold, fontSize: 10, letterSpacing: 2, color: theme.color.muted },
  rowRisk: { fontFamily: theme.font.monoBold, fontSize: 12, letterSpacing: 1, color: theme.color.onSurface },
  rowSummary: { fontFamily: theme.font.mono, fontSize: theme.size.sm, color: theme.color.onSurface, lineHeight: 18 },
  rowDate: { fontFamily: theme.font.mono, fontSize: 10, color: theme.color.muted, marginTop: theme.space.xs, letterSpacing: 1 },

  emptyWrap: { flex: 1, justifyContent: 'center', paddingHorizontal: theme.space.lg },
  emptyState: {
    alignItems: 'center',
    padding: theme.space.lg,
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
    backgroundColor: theme.color.surfaceSecondary,
  },
  emptyImage: {
    width: '100%',
    height: 160,
    borderBottomWidth: theme.border.thick,
    borderBottomColor: theme.color.borderStrong,
    marginBottom: theme.space.md,
  },
  emptyTitle: {
    fontFamily: theme.font.monoBold,
    fontSize: theme.size.sm,
    letterSpacing: 2,
    color: theme.color.onSurface,
    textAlign: 'center',
  },
  emptyBody: {
    fontFamily: theme.font.mono,
    fontSize: theme.size.sm,
    color: theme.color.onSurface,
    marginTop: theme.space.sm,
    textAlign: 'center',
    lineHeight: 18,
  },

  errorBox: {
    marginHorizontal: theme.space.lg,
    marginTop: theme.space.md,
    borderWidth: theme.border.thick,
    borderColor: theme.color.error,
    padding: theme.space.md,
  },
  errorText: { fontFamily: theme.font.monoBold, color: theme.color.error },
});
