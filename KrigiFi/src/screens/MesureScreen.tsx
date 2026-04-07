/**
 * Écran de prise de mesures WiFi.
 *
 * L'utilisateur marche dans la pièce pendant que l'app :
 *  - Suit sa position via le PDR (IMU)
 *  - Enregistre le RSSI WiFi toutes les 2 secondes
 *  - Affiche les points de mesure sur le plan
 */

import React, { useCallback, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Dimensions,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { ajouterMesures, effacerMesures, listerPieces } from '../services/api';
import { useImu } from '../hooks/useImu';
import { useWifi } from '../hooks/useWifi';
import { usePermissions } from '../hooks/usePermissions';
import DessinateurPlan from '../components/DessinateurPlan';
import type { ParamListeRacine, Piece } from '../types';

type Props = NativeStackScreenProps<ParamListeRacine, 'Mesure'>;

const { width: LARGEUR_ECRAN } = Dimensions.get('window');
const TAILLE_CANVAS = LARGEUR_ECRAN - 32;

function indicateurRssi(rssi: number): { couleur: string; label: string } {
  if (rssi >= -50) { return { couleur: '#00e676', label: 'Excellent' }; }
  if (rssi >= -60) { return { couleur: '#76ff03', label: 'Bon' }; }
  if (rssi >= -70) { return { couleur: '#ffea00', label: 'Moyen' }; }
  if (rssi >= -80) { return { couleur: '#ff9100', label: 'Faible' }; }
  return { couleur: '#f44336', label: 'Très faible' };
}

export default function MesureScreen({ route }: Props) {
  const { batimentId, batimentNom } = route.params;
  const permissions = usePermissions();
  const { etat: etatImu, demarrer: demarrerImu, arreter: arreterImu, reinitialiser: reinitImu } = useImu();

  if (permissions === 'refusees') {
    return (
      <View style={styles.conteneur}>
        <Text style={[styles.titre, { margin: 24 }]}>Permissions refusées</Text>
        <Text style={{ color: '#888', margin: 24 }}>
          L'accès à la localisation et aux capteurs est requis.{'\n'}
          Activez-les dans les paramètres de l'application.
        </Text>
      </View>
    );
  }
  const positionRef = useRef(etatImu.position);

  // Synchroniser la position pour le hook wifi
  React.useEffect(() => {
    positionRef.current = etatImu.position;
  }, [etatImu.position]);

  const obtenirPosition = useCallback(() => positionRef.current, []);
  const { etat: etatWifi, demarrer: demarrerWifi, arreter: arreterWifi, reinitialiser: reinitWifi } = useWifi(obtenirPosition);

  const [pieces, setPieces] = useState<Piece[]>([]);
  const [enSauvegarde, setEnSauvegarde] = useState(false);

  React.useEffect(() => {
    listerPieces(batimentId).then(setPieces).catch(() => {});
  }, [batimentId]);

  const demarrer = () => {
    demarrerImu();
    demarrerWifi();
  };

  const arreter = () => {
    arreterImu();
    arreterWifi();
  };

  const reinitialiser = () => {
    reinitImu();
    reinitWifi();
  };

  const sauvegarder = async () => {
    if (etatWifi.mesures.length === 0) {
      Alert.alert('Aucune mesure', 'Déplacez-vous pour enregistrer des mesures WiFi.');
      return;
    }
    setEnSauvegarde(true);
    try {
      const donnees = etatWifi.mesures.map(m => ({
        x: m.x,
        y: m.y,
        rssi: m.rssi,
        ssid: m.ssid,
        type_reseau: m.typeReseau,
      }));
      await ajouterMesures(batimentId, donnees);
      Alert.alert(
        'Succès',
        `${etatWifi.mesures.length} mesure(s) envoyées au serveur.\nVous pouvez maintenant générer la heatmap.`,
      );
      reinitialiser();
    } catch {
      Alert.alert('Erreur', 'Impossible d\'envoyer les mesures.');
    } finally {
      setEnSauvegarde(false);
    }
  };

  const confirmerEffacement = () => {
    Alert.alert(
      'Effacer les mesures',
      'Supprimer toutes les mesures WiFi de ce bâtiment ?',
      [
        { text: 'Annuler', style: 'cancel' },
        {
          text: 'Effacer',
          style: 'destructive',
          onPress: async () => {
            await effacerMesures(batimentId);
            reinitialiser();
            Alert.alert('Fait', 'Mesures effacées.');
          },
        },
      ],
    );
  };

  const rssiInfo = etatWifi.dernierRssi != null ? indicateurRssi(etatWifi.dernierRssi) : null;
  const enCours = etatImu.actif && etatWifi.enCours;

  return (
    <ScrollView style={styles.conteneur} contentContainerStyle={styles.scrollContent}>
      <Text style={styles.titre}>{batimentNom}</Text>
      <Text style={styles.sousTitre}>Marchez dans le bâtiment pour cartographier le signal</Text>

      {/* Canvas */}
      <DessinateurPlan
        largeur={TAILLE_CANVAS}
        hauteur={TAILLE_CANVAS}
        tracé={etatImu.tracé}
        sommets={etatImu.sommets}
        positionCourante={etatImu.position}
        pieces={pieces}
      />

      {/* Signal actuel */}
      <View style={styles.panneauSignal}>
        <View style={styles.signalGauche}>
          <Text style={styles.signalLabel}>Signal actuel</Text>
          {rssiInfo ? (
            <>
              <Text style={[styles.signalValeur, { color: rssiInfo.couleur }]}>
                {etatWifi.dernierRssi} dBm
              </Text>
              <Text style={[styles.signalQualite, { color: rssiInfo.couleur }]}>
                {rssiInfo.label}
              </Text>
            </>
          ) : (
            <Text style={styles.signalAucun}>—</Text>
          )}
        </View>

        <View style={styles.signalStats}>
          <StatItem label="Mesures" valeur={String(etatWifi.mesures.length)} />
          <StatItem label="Pas" valeur={String(etatImu.nbPas)} />
          <StatItem
            label="Distance"
            valeur={`${(etatImu.nbPas * 0.75).toFixed(0)} m`}
          />
        </View>
      </View>

      {/* Boutons de contrôle */}
      <View style={styles.boutons}>
        {!enCours ? (
          <TouchableOpacity style={[styles.btn, styles.btnVert]} onPress={demarrer}>
            <Text style={styles.btnTexte}>
              {etatWifi.mesures.length > 0 ? 'Reprendre' : 'Démarrer la mesure'}
            </Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity style={[styles.btn, styles.btnOrange]} onPress={arreter}>
            <Text style={styles.btnTexte}>Pause</Text>
          </TouchableOpacity>
        )}

        <TouchableOpacity
          style={[
            styles.btn,
            styles.btnBleu,
            (etatWifi.mesures.length === 0 || enSauvegarde) && styles.btnDesactive,
          ]}
          onPress={sauvegarder}
          disabled={etatWifi.mesures.length === 0 || enSauvegarde}
        >
          {enSauvegarde ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.btnTexte}>Envoyer ({etatWifi.mesures.length})</Text>
          )}
        </TouchableOpacity>

        <View style={styles.boutonsSec}>
          <TouchableOpacity style={[styles.btn, styles.btnGris, { flex: 1 }]} onPress={reinitialiser}>
            <Text style={styles.btnTexte}>Réinitialiser</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.btn, styles.btnRouge, { flex: 1 }]} onPress={confirmerEffacement}>
            <Text style={styles.btnTexte}>Effacer DB</Text>
          </TouchableOpacity>
        </View>
      </View>

      <Text style={styles.info}>
        Une mesure est prise automatiquement toutes les 2 secondes lorsque vous vous déplacez.
      </Text>
    </ScrollView>
  );
}

function StatItem({ label, valeur }: { label: string; valeur: string }) {
  return (
    <View style={{ alignItems: 'center' }}>
      <Text style={{ color: '#4fc3f7', fontSize: 20, fontWeight: 'bold' }}>{valeur}</Text>
      <Text style={{ color: '#888', fontSize: 11 }}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  conteneur: { flex: 1, backgroundColor: '#0d0d1a' },
  scrollContent: { padding: 16, gap: 16 },
  titre: { fontSize: 22, fontWeight: 'bold', color: '#e0e0ff' },
  sousTitre: { fontSize: 13, color: '#888' },
  panneauSignal: {
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#2a2a4e',
  },
  signalGauche: { flex: 1 },
  signalLabel: { color: '#888', fontSize: 12, marginBottom: 4 },
  signalValeur: { fontSize: 28, fontWeight: 'bold' },
  signalQualite: { fontSize: 13, marginTop: 2 },
  signalAucun: { color: '#555', fontSize: 24 },
  signalStats: {
    flexDirection: 'row',
    gap: 20,
    paddingLeft: 16,
    borderLeftWidth: 1,
    borderLeftColor: '#2a2a4e',
  },
  boutons: { gap: 10 },
  boutonsSec: { flexDirection: 'row', gap: 10 },
  btn: { paddingVertical: 14, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  btnTexte: { color: '#fff', fontSize: 15, fontWeight: '600' },
  btnVert: { backgroundColor: '#2e7d32' },
  btnOrange: { backgroundColor: '#e65100' },
  btnBleu: { backgroundColor: '#1565c0' },
  btnGris: { backgroundColor: '#37474f' },
  btnRouge: { backgroundColor: '#b71c1c' },
  btnDesactive: { opacity: 0.4 },
  info: { color: '#555', fontSize: 12, textAlign: 'center', lineHeight: 18 },
});
