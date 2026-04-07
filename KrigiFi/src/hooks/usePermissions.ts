/**
 * Hook de gestion des permissions Android.
 * À appeler au démarrage de chaque écran nécessitant capteurs ou WiFi.
 */

import { useEffect, useState } from 'react';
import { PermissionsAndroid, Platform } from 'react-native';

export type EtatPermissions = 'en_attente' | 'accordees' | 'refusees';

const PERMISSIONS_REQUISES = [
  PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
  PermissionsAndroid.PERMISSIONS.BODY_SENSORS,
];

export function usePermissions(): EtatPermissions {
  const [etat, setEtat] = useState<EtatPermissions>('en_attente');

  useEffect(() => {
    if (Platform.OS !== 'android') {
      setEtat('accordees');
      return;
    }

    PermissionsAndroid.requestMultiple(PERMISSIONS_REQUISES).then(resultats => {
      const toutesAccordees = PERMISSIONS_REQUISES.every(
        p => resultats[p] === PermissionsAndroid.RESULTS.GRANTED,
      );
      setEtat(toutesAccordees ? 'accordees' : 'refusees');
    });
  }, []);

  return etat;
}
