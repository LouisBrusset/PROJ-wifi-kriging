/**
 * Navigateur principal de l'application.
 * Stack navigation : Accueil → Plan / Mesure / Carte
 */

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import AccueilScreen from '../screens/AccueilScreen';
import PlanScreen from '../screens/PlanScreen';
import MesureScreen from '../screens/MesureScreen';
import CarteScreen from '../screens/CarteScreen';
import type { ParamListeRacine } from '../types';

const Stack = createNativeStackNavigator<ParamListeRacine>();

const optionsEntete = {
  headerStyle: { backgroundColor: '#1a1a2e' },
  headerTintColor: '#4fc3f7',
  headerTitleStyle: { color: '#e0e0ff', fontWeight: '600' as const },
};

export default function AppNavigateur() {
  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={optionsEntete}>
        <Stack.Screen
          name="Accueil"
          component={AccueilScreen}
          options={{ headerShown: false }}
        />
        <Stack.Screen
          name="Plan"
          component={PlanScreen}
          options={({ route }) => ({ title: `Plan – ${route.params.batimentNom}` })}
        />
        <Stack.Screen
          name="Mesure"
          component={MesureScreen}
          options={({ route }) => ({ title: `Mesure – ${route.params.batimentNom}` })}
        />
        <Stack.Screen
          name="Carte"
          component={CarteScreen}
          options={({ route }) => ({ title: `Heatmap – ${route.params.batimentNom}` })}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
