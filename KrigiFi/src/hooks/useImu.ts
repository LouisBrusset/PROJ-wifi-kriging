/**
 * Hook de suivi de position par déduction inertielle (PDR).
 *
 * Algorithme PDR (Pedestrian Dead Reckoning) :
 *  - Détection de pas via les pics de magnitude de l'accéléromètre
 *  - Suivi du cap via l'intégration du gyroscope Z
 *  - Longueur de pas fixe (configurable, ~0.75 m par défaut)
 *  - Un nouveau sommet est enregistré à chaque virage > SEUIL_VIRAGE degrés
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { abonnerAccelerometre, abonnerGyroscope, calculerMagnitude } from '../services/capteurs';
import type { Point } from '../types';
import type { Subscription } from 'rxjs';

const LONGUEUR_PAS_M = 0.75;           // mètres par pas
const SEUIL_PIC_ACCEL = 11.5;          // m/s² – seuil de détection de pas
const SEUIL_REFRACTAIRE_MS = 300;      // min. temps entre deux pas
const SEUIL_VIRAGE_DEG = 60;           // angle de virage pour créer un sommet

export interface EtatImu {
  position: Point;          // position courante (mètres)
  cap: number;              // cap courant (degrés, 0 = Nord)
  nbPas: number;
  sommets: Point[];         // coins enregistrés lors des virages
  tracé: Point[];           // historique complet de la trajectoire
  actif: boolean;
}

export function useImu() {
  const [etat, setEtat] = useState<EtatImu>({
    position: { x: 0, y: 0 },
    cap: 0,
    nbPas: 0,
    sommets: [],
    tracé: [{ x: 0, y: 0 }],
    actif: false,
  });

  const refDernierpas = useRef<number>(0);
  const refMagnitudePrecedente = useRef<number>(9.8);
  const refCapAccumule = useRef<number>(0);
  const refCapDernierSommet = useRef<number>(0);
  const refPosition = useRef<Point>({ x: 0, y: 0 });
  const refCap = useRef<number>(0);
  const refSommets = useRef<Point[]>([]);
  const refTracé = useRef<Point[]>([{ x: 0, y: 0 }]);

  const subAccel = useRef<Subscription | null>(null);
  const subGyro = useRef<Subscription | null>(null);

  const demarrer = useCallback(() => {
    if (subAccel.current) { return; }

    // Réinitialisation
    refPosition.current = { x: 0, y: 0 };
    refCap.current = 0;
    refSommets.current = [];
    refTracé.current = [{ x: 0, y: 0 }];
    refCapAccumule.current = 0;
    refCapDernierSommet.current = 0;

    setEtat(prev => ({
      ...prev,
      position: { x: 0, y: 0 },
      cap: 0,
      nbPas: 0,
      sommets: [],
      tracé: [{ x: 0, y: 0 }],
      actif: true,
    }));

    // Gyroscope → intégration du cap
    subGyro.current = abonnerGyroscope(({ z, timestamp }) => {
      const dt = 0.05; // 50 ms
      const deltaDeg = (z * dt * 180) / Math.PI;
      refCap.current = (refCap.current + deltaDeg + 360) % 360;
      refCapAccumule.current += Math.abs(deltaDeg);
    });

    // Accéléromètre → détection de pas
    subAccel.current = abonnerAccelerometre(({ x, y, z, timestamp }) => {
      const mag = calculerMagnitude(x, y, z);
      const maintenant = timestamp;

      // Détection du pic (montée puis descente)
      const estPic =
        mag > SEUIL_PIC_ACCEL &&
        refMagnitudePrecedente.current <= SEUIL_PIC_ACCEL &&
        maintenant - refDernierpas.current > SEUIL_REFRACTAIRE_MS;

      refMagnitudePrecedente.current = mag;

      if (!estPic) { return; }
      refDernierpas.current = maintenant;

      // Avancer d'un pas dans la direction du cap
      const capRad = (refCap.current * Math.PI) / 180;
      const nouvX = refPosition.current.x + LONGUEUR_PAS_M * Math.sin(capRad);
      const nouvY = refPosition.current.y + LONGUEUR_PAS_M * Math.cos(capRad);
      refPosition.current = { x: nouvX, y: nouvY };
      refTracé.current = [...refTracé.current, { ...refPosition.current }];

      // Détection de virage → enregistrement d'un sommet
      const deltaCapDepuisSommet = Math.abs(refCapAccumule.current - refCapDernierSommet.current);
      if (deltaCapDepuisSommet > SEUIL_VIRAGE_DEG) {
        refSommets.current = [...refSommets.current, { ...refPosition.current }];
        refCapDernierSommet.current = refCapAccumule.current;
      }

      setEtat(prev => ({
        ...prev,
        position: { ...refPosition.current },
        cap: refCap.current,
        nbPas: prev.nbPas + 1,
        sommets: [...refSommets.current],
        tracé: [...refTracé.current],
      }));
    });
  }, []);

  const arreter = useCallback(() => {
    subAccel.current?.unsubscribe();
    subGyro.current?.unsubscribe();
    subAccel.current = null;
    subGyro.current = null;
    setEtat(prev => ({ ...prev, actif: false }));
  }, []);

  const reinitialiser = useCallback(() => {
    arreter();
    refPosition.current = { x: 0, y: 0 };
    refCap.current = 0;
    refSommets.current = [];
    refTracé.current = [{ x: 0, y: 0 }];
    setEtat({
      position: { x: 0, y: 0 },
      cap: 0,
      nbPas: 0,
      sommets: [],
      tracé: [{ x: 0, y: 0 }],
      actif: false,
    });
  }, [arreter]);

  useEffect(() => {
    return () => { arreter(); };
  }, [arreter]);

  return { etat, demarrer, arreter, reinitialiser };
}
