/**
 * Point d'entrée de l'application KrigiFi.
 */

import React from 'react';
import { StatusBar } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import AppNavigateur from './src/navigation/AppNavigateur';

export default function App() {
  return (
    <SafeAreaProvider>
      <StatusBar barStyle="light-content" backgroundColor="#0d0d1a" />
      <AppNavigateur />
    </SafeAreaProvider>
  );
}
