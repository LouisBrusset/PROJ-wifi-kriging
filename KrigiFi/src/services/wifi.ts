/**
 * Service de mesure du signal WiFi.
 * Utilise react-native-wifi-reborn (v4) + NetInfo pour le RSSI.
 */

import WifiManager from 'react-native-wifi-reborn';
import NetInfo from '@react-native-community/netinfo';

export interface MesureSignal {
  rssi: number;
  ssid: string;
  typeReseau: 'wifi' | 'mobile';
  horodatage: number;
}

/**
 * Prend une mesure de signal unique (WiFi ou mobile).
 * Retourne null si aucun réseau disponible.
 */
export async function prendreMessure(): Promise<MesureSignal | null> {
  try {
    const etatReseau = await NetInfo.fetch();

    if (etatReseau.type === 'wifi' && etatReseau.isConnected) {
      // v4 : getCurrentWifiSSID() + le RSSI vient de NetInfo details
      const ssid = await WifiManager.getCurrentWifiSSID().catch((_e: unknown) => 'Inconnu');
      const details = etatReseau.details as any;
      // NetInfo expose strength (0-100) sur Android, on le convertit en dBm approx.
      const force = details?.strength ?? null;
      const rssi: number = force != null ? -100 + force * 0.7 : -70;
      return {
        rssi: Math.round(rssi),
        ssid: ssid ?? 'Inconnu',
        typeReseau: 'wifi',
        horodatage: Date.now(),
      };
    }

    if (etatReseau.type === 'cellular' && etatReseau.isConnected) {
      const details = etatReseau.details as any;
      const rssi = details?.strength != null ? -100 + details.strength * 0.4 : -80;
      return {
        rssi: Math.round(rssi),
        ssid: `Mobile ${details?.cellularGeneration ?? '4G'}`,
        typeReseau: 'mobile',
        horodatage: Date.now(),
      };
    }

    return null;
  } catch (e) {
    console.warn('[WiFi] Erreur de mesure :', e);
    return null;
  }
}

/**
 * Prend N mesures espacées de `delaiMs` ms et retourne la médiane.
 * Réduit le bruit de mesure.
 */
export async function prendreMessureMediane(
  nbMesures: number = 3,
  delaiMs: number = 300,
): Promise<MesureSignal | null> {
  const resultats: number[] = [];
  let dernierResultat: MesureSignal | null = null;

  for (let i = 0; i < nbMesures; i++) {
    const m = await prendreMessure();
    if (m) {
      resultats.push(m.rssi);
      dernierResultat = m;
    }
    if (i < nbMesures - 1) {
      await new Promise<void>(r => setTimeout(r, delaiMs));
    }
  }

  if (!dernierResultat || resultats.length === 0) {
    return null;
  }

  resultats.sort((a, b) => a - b);
  const mediane = resultats[Math.floor(resultats.length / 2)];
  return { ...dernierResultat, rssi: mediane };
}
