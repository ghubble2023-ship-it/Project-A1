import React, { useState } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TextInput, View, Pressable } from 'react-native';
import { useRouter } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';

import { theme } from '@/src/theme';
import { api } from '@/src/api';
import { PrimaryButton, FieldLabel } from '@/src/ui';

export default function TextScanScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const [chat, setChat] = useState('');
  const [profileText, setProfileText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setError(null);
    setLoading(true);
    try {
      const result = await api.scanText({
        chat_messages: chat,
        profile_text: profileText || null,
      });
      const summary = chat ? chat.slice(0, 80) : profileText.slice(0, 80);
      await api.saveScan({
        scan_type: 'text',
        input_summary: summary || 'Empty input',
        result,
        overall_risk: result.risk_level,
      }).catch(() => {});
      router.push({ pathname: '/results', params: { data: JSON.stringify(result), kind: 'text' } });
    } catch (e: any) {
      setError(e.message ?? 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <View style={[styles.root, { paddingTop: insets.top }]}>
        <ScrollView
          style={{ flex: 1 }}
          contentContainerStyle={styles.scroll}
          keyboardShouldPersistTaps="handled"
        >
          <Pressable
            testID="text-scan-back"
            onPress={() => router.back()}
            style={styles.backBtn}
          >
            <MaterialCommunityIcons name="arrow-left" size={22} color={theme.color.onSurface} />
            <Text style={styles.backText}>BACK</Text>
          </Pressable>

          <Text style={styles.code}>MODULE / 01</Text>
          <Text style={styles.title}>TEXT ANALYSIS</Text>
          <Text style={styles.body}>
            Paste chat transcripts and profile text. The engine looks for love-bombing,
            off-platform migration, unverifiable identity, pressure, and isolation signals.
          </Text>

          <View style={{ height: theme.space.xl }} />

          <FieldLabel>Chat Messages</FieldLabel>
          <TextInput
            testID="text-scan-chat-input"
            style={styles.textArea}
            value={chat}
            onChangeText={setChat}
            multiline
            placeholder="Paste chat text here..."
            placeholderTextColor={theme.color.muted}
            textAlignVertical="top"
          />

          <View style={{ height: theme.space.lg }} />

          <FieldLabel>Profile Text (optional)</FieldLabel>
          <TextInput
            testID="text-scan-profile-input"
            style={[styles.textArea, { minHeight: 100 }]}
            value={profileText}
            onChangeText={setProfileText}
            multiline
            placeholder="Bio, headline, description..."
            placeholderTextColor={theme.color.muted}
            textAlignVertical="top"
          />

          {error ? (
            <View style={styles.errorBox} testID="text-scan-error">
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}
        </ScrollView>

        <View style={[styles.footer, { paddingBottom: Math.max(insets.bottom, theme.space.md) }]}>
          <PrimaryButton
            label={loading ? 'ANALYZING...' : 'ANALYZE TEXT'}
            onPress={run}
            disabled={loading}
            testID="text-scan-run"
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
  backText: {
    fontFamily: theme.font.monoBold,
    fontSize: 10,
    letterSpacing: 2,
    color: theme.color.onSurface,
    marginLeft: theme.space.xs,
  },
  code: {
    fontFamily: theme.font.monoBold,
    fontSize: 10,
    letterSpacing: 2,
    color: theme.color.muted,
    marginTop: theme.space.sm,
  },
  title: {
    fontFamily: theme.font.display,
    fontSize: 32,
    letterSpacing: -1,
    color: theme.color.onSurface,
    marginTop: theme.space.xs,
    marginBottom: theme.space.md,
  },
  body: {
    fontFamily: theme.font.mono,
    fontSize: theme.size.sm,
    lineHeight: 20,
    color: theme.color.onSurface,
  },
  textArea: {
    minHeight: 160,
    borderWidth: theme.border.thick,
    borderColor: theme.color.borderStrong,
    backgroundColor: theme.color.surfaceSecondary,
    padding: theme.space.md,
    fontFamily: theme.font.mono,
    fontSize: theme.size.base,
    color: theme.color.onSurface,
  },
  errorBox: {
    marginTop: theme.space.md,
    borderWidth: theme.border.thick,
    borderColor: theme.color.error,
    padding: theme.space.md,
    backgroundColor: theme.color.surface,
  },
  errorText: {
    fontFamily: theme.font.monoBold,
    color: theme.color.error,
    fontSize: theme.size.sm,
  },
  footer: {
    borderTopWidth: theme.border.thick,
    borderTopColor: theme.color.borderStrong,
    paddingHorizontal: theme.space.lg,
    paddingTop: theme.space.md,
    backgroundColor: theme.color.surface,
  },
});
