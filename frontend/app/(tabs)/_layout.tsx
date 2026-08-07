import { Tabs } from 'expo-router';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { theme } from '@/src/theme';

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: theme.color.onSurfaceInverse,
        tabBarInactiveTintColor: theme.color.muted,
        tabBarStyle: {
          backgroundColor: theme.color.surfaceInverse,
          borderTopWidth: theme.border.thick,
          borderTopColor: theme.color.borderStrong,
          height: 68,
          paddingTop: 8,
          paddingBottom: 12,
        },
        tabBarLabelStyle: {
          fontFamily: theme.font.monoBold,
          fontSize: 10,
          letterSpacing: 1.5,
        },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'CORE',
          tabBarIcon: ({ color }) => (
            <MaterialCommunityIcons name="crosshairs-gps" size={22} color={color} />
          ),
          tabBarButtonTestID: 'tab-core',
        }}
      />
      <Tabs.Screen
        name="history"
        options={{
          title: 'ARCHIVE',
          tabBarIcon: ({ color }) => (
            <MaterialCommunityIcons name="archive-outline" size={22} color={color} />
          ),
          tabBarButtonTestID: 'tab-archive',
        }}
      />
    </Tabs>
  );
}
