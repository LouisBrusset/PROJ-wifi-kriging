/**
 * Abstraction des capteurs IMU (accéléromètre + gyroscope).
 * Utilise react-native-sensors pour accéder à la centrale inertielle du téléphone.
 */

import {
  accelerometer,
  gyroscope,
  setUpdateIntervalForType,
  SensorTypes,
} from 'react-native-sensors';
import type { Subscription } from 'rxjs';
import type { DonneesImu } from '../types';

// Intervalle de mise à jour en millisecondes
const INTERVALLE_MS = 50;

setUpdateIntervalForType(SensorTypes.accelerometer, INTERVALLE_MS);
setUpdateIntervalForType(SensorTypes.gyroscope, INTERVALLE_MS);

export function abonnerAccelerometre(
  rappel: (donnees: DonneesImu) => void,
): Subscription {
  return accelerometer.subscribe(({ x, y, z, timestamp }) => {
    rappel({ x, y, z, timestamp });
  });
}

export function abonnerGyroscope(
  rappel: (donnees: DonneesImu) => void,
): Subscription {
  return gyroscope.subscribe(({ x, y, z, timestamp }) => {
    rappel({ x, y, z, timestamp });
  });
}

/**
 * Calcule la magnitude du vecteur d'accélération.
 * Utilisé pour la détection de pas.
 */
export function calculerMagnitude(x: number, y: number, z: number): number {
  return Math.sqrt(x * x + y * y + z * z);
}
