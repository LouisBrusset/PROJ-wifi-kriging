/**
 * Hook de mesure WiFi combiné à la position IMU.
 * Prend des mesures à intervalles réguliers et les associe à la position courante.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { prendreMessureMediane } from '../services/wifi';
import type { Point } from '../types';

export interface PointMesure {
  x: number;
  y: number;
  rssi: number;
  ssid: string;
  typeReseau: 'wifi' | 'mobile';
  horodatage: number;
}

export interface EtatMesure {
  mesures: PointMesure[];
  enCours: boolean;
  dernierRssi: number | null;
}

const INTERVALLE_MESURE_MS = 2000; // Mesure toutes les 2 secondes

export function useWifi(obtenirPosition: () => Point) {
  const [etat, setEtat] = useState<EtatMesure>({
    mesures: [],
    enCours: false,
    dernierRssi: null,
  });

  const refIntervalle = useRef<ReturnType<typeof setInterval> | null>(null);
  const refMesures = useRef<PointMesure[]>([]);

  const demarrer = useCallback(() => {
    if (refIntervalle.current) { return; }
    setEtat(prev => ({ ...prev, enCours: true }));

    refIntervalle.current = setInterval(async () => {
      const signal = await prendreMessureMediane(3, 250);
      if (!signal) { return; }

      const pos = obtenirPosition();
      const point: PointMesure = {
        x: pos.x,
        y: pos.y,
        rssi: signal.rssi,
        ssid: signal.ssid,
        typeReseau: signal.typeReseau,
        horodatage: signal.horodatage,
      };

      refMesures.current = [...refMesures.current, point];
      setEtat(prev => ({
        ...prev,
        mesures: [...refMesures.current],
        dernierRssi: signal.rssi,
      }));
    }, INTERVALLE_MESURE_MS);
  }, [obtenirPosition]);

  const arreter = useCallback(() => {
    if (refIntervalle.current) {
      clearInterval(refIntervalle.current);
      refIntervalle.current = null;
    }
    setEtat(prev => ({ ...prev, enCours: false }));
  }, []);

  const reinitialiser = useCallback(() => {
    arreter();
    refMesures.current = [];
    setEtat({ mesures: [], enCours: false, dernierRssi: null });
  }, [arreter]);

  useEffect(() => {
    return () => { arreter(); };
  }, [arreter]);

  return { etat, demarrer, arreter, reinitialiser };
}
