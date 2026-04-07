/**
 * Écran de visualisation de la heatmap WiFi.
 * Lance le krigeage sur le serveur et affiche le résultat interpolé.
 */

import React, { useState } from 'react';
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
import { calculerHeatmap, listerMesures, listerPieces } from '../services/api';
import CarteThermique from '../components/CarteThermique';
import type { MesureWifi, Piece, ResultatKrigeage, ParamListeRacine } from '../types';

type Props = NativeStackScreenProps<ParamListeRacine, 'Carte'>;

const { width: LARGEUR_ECRAN } = Dimensions.get('window');
const TAILLE_CANVAS = LARGEUR_ECRAN - 32;

type Variogramme = 'spherical' | 'gaussian' | 'exponential';
const VARIOGRAMMES: Variogramme[] = ['spherical', 'gaussian', 'exponential'];
const RESOLUTIONS = [30, 50, 80];

export default function CarteScreen({ route }: Props) {
  const { batimentId, batimentNom } = route.params;

  const [resultat, setResultat] = useState<ResultatKrigeage | null>(null);
  const [mesures, setMesures] = useState<MesureWifi[]>([]);
  const [pieces, setPieces] = useState<Piece[]>([]);
  const [enCalcul, setEnCalcul] = useState(false);
  const [variogramme, setVariogramme] = useState<Variogramme>('spherical');
  const [resolution, setResolution] = useState(50);
  const [afficherMesures, setAfficherMesures] = useState(true);

  React.useEffect(() => {
    listerPieces(batimentId).then(setPieces).catch(() => {});
    listerMesures(batimentId).then(setMesures).catch(() => {});
  }, [batimentId]);

  const lancerKrigeage = async () => {
    if (mesures.length < 4) {
      Alert.alert(
        'Mesures insuffisantes',
        `Le krigeage nécessite au moins 4 mesures.\nActuellement : ${mesures.length} mesure(s).`,
      );
      return;
    }
    setEnCalcul(true);
    try {
      const res = await calculerHeatmap(batimentId, resolution, variogramme);
      setResultat(res);
    } catch (e: any) {
      Alert.alert(
        'Erreur de krigeage',
        e?.response?.data?.detail ?? 'Une erreur est survenue lors du calcul.',
      );
    } finally {
      setEnCalcul(false);
    }
  };

  return (
    <ScrollView style={styles.conteneur} contentContainerStyle={styles.scrollContent}>
      <Text style={styles.titre}>{batimentNom}</Text>
      <Text style={styles.sousTitre}>
        {mesures.length} mesure(s) disponible(s) · {pieces.length} pièce(s)
      </Text>

      {/* Paramètres */}
      <View style={styles.section}>
        <Text style={styles.sectionTitre}>Modèle de variogramme</Text>
        <View style={styles.chips}>
          {VARIOGRAMMES.map(v => (
            <TouchableOpacity
              key={v}
              style={[styles.chip, variogramme === v && styles.chipActif]}
              onPress={() => setVariogramme(v)}
            >
              <Text style={[styles.chipTexte, variogramme === v && styles.chipTexteActif]}>
                {v}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <Text style={[styles.sectionTitre, { marginTop: 12 }]}>Résolution de la grille</Text>
        <View style={styles.chips}>
          {RESOLUTIONS.map(r => (
            <TouchableOpacity
              key={r}
              style={[styles.chip, resolution === r && styles.chipActif]}
              onPress={() => setResolution(r)}
            >
              <Text style={[styles.chipTexte, resolution === r && styles.chipTexteActif]}>
                {r}×{r}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Bouton krigeage */}
      <TouchableOpacity
        style={[styles.btn, styles.btnViola, enCalcul && styles.btnDesactive]}
        onPress={lancerKrigeage}
        disabled={enCalcul}
      >
        {enCalcul ? (
          <View style={styles.btnContenu}>
            <ActivityIndicator color="#fff" style={{ marginRight: 8 }} />
            <Text style={styles.btnTexte}>Krigeage en cours…</Text>
          </View>
        ) : (
          <Text style={styles.btnTexte}>Calculer la heatmap</Text>
        )}
      </TouchableOpacity>

      {/* Heatmap */}
      {resultat ? (
        <>
          <CarteThermique
            largeur={TAILLE_CANVAS}
            hauteur={TAILLE_CANVAS}
            resultat={resultat}
            mesures={afficherMesures ? mesures : []}
            pieces={pieces}
          />

          {/* Statistiques */}
          <View style={styles.statsGrid}>
            <StatCard label="RSSI min" valeur={`${resultat.rssi_min.toFixed(1)} dBm`} />
            <StatCard label="RSSI max" valeur={`${resultat.rssi_max.toFixed(1)} dBm`} />
            <StatCard label="Grille" valeur={`${resultat.resolution}²`} />
            <StatCard label="Modèle" valeur={variogramme} />
          </View>

          <TouchableOpacity
            style={styles.toggleMesures}
            onPress={() => setAfficherMesures(p => !p)}
          >
            <Text style={styles.toggleTexte}>
              {afficherMesures ? 'Masquer' : 'Afficher'} les points de mesure
            </Text>
          </TouchableOpacity>

          <Text style={styles.info}>
            La heatmap est générée par krigeage ordinaire.{'\n'}
            Les zones bleues indiquent un signal faible, les zones rouges un signal fort.
          </Text>
        </>
      ) : (
        <View style={styles.vide}>
          <Text style={styles.videIcone}>📡</Text>
          <Text style={styles.videTexte}>
            Appuyez sur "Calculer" pour générer la heatmap à partir des mesures WiFi.
          </Text>
        </View>
      )}
    </ScrollView>
  );
}

function StatCard({ label, valeur }: { label: string; valeur: string }) {
  return (
    <View style={styles.statCard}>
      <Text style={styles.statValeur}>{valeur}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  conteneur: { flex: 1, backgroundColor: '#0d0d1a' },
  scrollContent: { padding: 16, gap: 16 },
  titre: { fontSize: 22, fontWeight: 'bold', color: '#e0e0ff' },
  sousTitre: { fontSize: 13, color: '#888' },
  section: {
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: '#2a2a4e',
  },
  sectionTitre: { color: '#aaa', fontSize: 12, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5 },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 20,
    backgroundColor: '#0d0d1a',
    borderWidth: 1,
    borderColor: '#2a2a4e',
  },
  chipActif: { backgroundColor: '#6a1b9a', borderColor: '#9c27b0' },
  chipTexte: { color: '#888', fontSize: 13 },
  chipTexteActif: { color: '#fff', fontWeight: '600' },
  btn: { paddingVertical: 14, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  btnContenu: { flexDirection: 'row', alignItems: 'center' },
  btnTexte: { color: '#fff', fontSize: 15, fontWeight: '600' },
  btnViola: { backgroundColor: '#6a1b9a' },
  btnDesactive: { opacity: 0.5 },
  statsGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  statCard: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: '#1a1a2e',
    borderRadius: 10,
    padding: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#2a2a4e',
  },
  statValeur: { color: '#ce93d8', fontSize: 16, fontWeight: 'bold' },
  statLabel: { color: '#888', fontSize: 11, marginTop: 2 },
  toggleMesures: { alignItems: 'center', paddingVertical: 8 },
  toggleTexte: { color: '#4fc3f7', fontSize: 13 },
  vide: { alignItems: 'center', padding: 40, gap: 16 },
  videIcone: { fontSize: 48 },
  videTexte: { color: '#555', fontSize: 14, textAlign: 'center', lineHeight: 22 },
  info: { color: '#555', fontSize: 12, textAlign: 'center', lineHeight: 18 },
});
