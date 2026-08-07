import { Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { useEffect } from 'react';
import { LogBox } from 'react-native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { useFonts } from 'expo-font';
import { StatusBar } from 'expo-status-bar';

import { useIconFonts } from '@/src/hooks/use-icon-fonts';

LogBox.ignoreAllLogs(true);

// Keep the native splash visible from cold start until icon fonts register.
// (Required for Android Expo Go icon prewarm — do not remove.)
SplashScreen.preventAutoHideAsync();

// Brand fonts pulled from public CDN URLs to avoid banned @expo-google-fonts pkgs.
const BRAND_FONTS = {
  SpaceGrotesk_500Medium:
    'https://fonts.gstatic.com/s/spacegrotesk/v16/V8mDoQDjQSkFtoMM3T6r8E7mF71Q-gOoraIAEj7oUUxjaVs.ttf',
  SpaceGrotesk_700Bold:
    'https://fonts.gstatic.com/s/spacegrotesk/v16/V8mDoQDjQSkFtoMM3T6r8E7mF71Q-gOoraIAEj62WExjaVs.ttf',
  IBMPlexMono_400Regular:
    'https://fonts.gstatic.com/s/ibmplexmono/v19/-F63fjptAgt5VM-kVkqdyU8n5igg1l9kn-s.ttf',
  IBMPlexMono_500Medium:
    'https://fonts.gstatic.com/s/ibmplexmono/v19/-F6qfjptAgt5VM-kVkqdyU8n5jsL0R11r1YE.ttf',
  IBMPlexMono_700Bold:
    'https://fonts.gstatic.com/s/ibmplexmono/v19/-F6qfjptAgt5VM-kVkqdyU8n5jsL2eB1r1YE.ttf',
};

export default function RootLayout() {
  const [iconsLoaded, iconError] = useIconFonts();
  const [fontsLoaded, fontError] = useFonts(BRAND_FONTS);

  useEffect(() => {
    if ((iconsLoaded || iconError) && (fontsLoaded || fontError)) {
      SplashScreen.hideAsync();
    }
  }, [iconsLoaded, iconError, fontsLoaded, fontError]);

  if (!iconsLoaded && !iconError) return null;
  // Do not block on brand font error — system font will fall back.

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <StatusBar style="dark" />
        <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: '#FFFFFF' } }} />
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}
