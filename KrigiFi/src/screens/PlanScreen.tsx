/**
 * Écran de construction du plan.
 *
 * Mode 1 – PDR (IMU) : l'utilisateur marche le long des murs, l'app trace la pièce.
 * Mode 2 – Image : l'utilisateur uploade un plan existant et calibre l'échelle.
 */

import React, { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Dimensions,
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { launchImageLibrary } from 'react-native-image-picker';
import { creerPiece, uploaderPlan, definirEchelle, listerPieces } from '../services/api';
import { useImu } from '../hooks/useImu';
import { usePermissions } from '../hooks/usePermissions';
import DessinateurPlan from '../components/DessinateurPlan';
import type { ParamListeRacine, Piece } from '../types';

type Props = NativeStackScreenProps<ParamListeRacine, 'Plan'>;

const { width: LARGEUR_ECRAN } = Dimensions.get('window');
const TAILLE_CANVAS = LARGEUR_ECRAN - 32;

type Mode = 'choix' | 'imu' | 'image';

export default function PlanScreen({ route }: Props) {
  const { batimentId, batimentNom } = route.params;
  const permissions = usePermissions();
  const { etat: etatImu, demarrer, arreter, reinitialiser } = useImu();

  const [mode, setMode] = useState<Mode>('choix');

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
  const [nomPiece, setNomPiece] = useState('Salon');
  const [modalNom, setModalNom] = useState(false);
  const [pieces, setPieces] = useState<Piece[]>([]);
  const [chargement, setChargement] = useState(false);

  // Charger les pièces existantes au montage
  React.useEffect(() => {
    listerPieces(batimentId).then(setPieces).catch(() => {});
  }, [batimentId]);

  // ── Mode IMU ───────────────────────────────────────────────────────────────

  const sauvegarderPieceImu = async (nom: string) => {
    if (etatImu.sommets.length < 3) {
      Alert.alert('Insuffisant', 'Marchez autour de la pièce pour détecter au moins 3 coins.');
      return;
    }
    setChargement(true);
    try {
      // On ferme le polygone en ajoutant le premier sommet
      const pointsFermes = [...etatImu.sommets, etatImu.sommets[0]];
      const piece = await creerPiece(batimentId, nom, pointsFermes);
      setPieces(prev => [...prev, piece]);
      reinitialiser();
      Alert.alert('Succès', `Pièce "${nom}" sauvegardée !`);
    } catch {
      Alert.alert('Erreur', 'Impossible de sauvegarder la pièce.');
    } finally {
      setChargement(false);
    }
  };

  // ── Mode Image ─────────────────────────────────────────────────────────────

  const [imageUri, setImageUri] = useState<string | null>(null);
  const [metresParPixel, setMetresParPixel] = useState('');

  const choisirImage = async () => {
    const res = await launchImageLibrary({ mediaType: 'photo', quality: 0.8 });
    if (res.assets?.[0]?.uri) {
      setImageUri(res.assets[0].uri);
    }
  };

  const uploaderImage = async () => {
    if (!imageUri) { return; }
    setChargement(true);
    try {
      await uploaderPlan(batimentId, imageUri);
      if (metresParPixel) {
        await definirEchelle(batimentId, parseFloat(metresParPixel));
      }
      Alert.alert('Succès', 'Plan importé avec succès !');
      setImageUri(null);
    } catch {
      Alert.alert('Erreur', "Impossible d'uploader le plan.");
    } finally {
      setChargement(false);
    }
  };

  // ── Rendu ──────────────────────────────────────────────────────────────────

  if (mode === 'choix') {
    return (
      <View style={styles.conteneur}>
        <Text style={styles.titre}>{batimentNom}</Text>
        <Text style={styles.sousTitre}>Comment souhaitez-vous construire le plan ?</Text>

        <TouchableOpacity style={styles.carteMode} onPress={() => setMode('imu')}>
          <Text style={styles.carteModeIcone}>🚶</Text>
          <View>
            <Text style={styles.carteModeNom}>Marcher dans la pièce</Text>
            <Text style={styles.carteModeDesc}>
              Utilisez l'accéléromètre et le gyroscope pour tracer les contours automatiquement.
            </Text>
          </View>
        </TouchableOpacity>

        <TouchableOpacity style={styles.carteMode} onPress={() => setMode('image')}>
          <Text style={styles.carteModeIcone}>🗺️</Text>
          <View>
            <Text style={styles.carteModeNom}>Importer un plan</Text>
            <Text style={styles.carteModeDesc}>
              Importez une image existante et calibrez l'échelle en mètres.
            </Text>
          </View>
        </TouchableOpacity>

        {pieces.length > 0 && (
          <View style={styles.piecesSauvees}>
            <Text style={styles.piecesSaveesTexte}>
              {pieces.length} pièce(s) enregistrée(s)
            </Text>
          </View>
        )}
      </View>
    );
  }

  if (mode === 'imu') {
    return (
      <ScrollView style={styles.conteneur} contentContainerStyle={styles.scrollContent}>
        <Text style={styles.titre}>Tracer la pièce</Text>
        <Text style={styles.instruction}>
          Marchez le long des murs en tenant le téléphone à la verticale.{'\n'}
          Les virages supérieurs à 60° créent automatiquement un coin.
        </Text>

        <DessinateurPlan
          largeur={TAILLE_CANVAS}
          hauteur={TAILLE_CANVAS}
          tracé={etatImu.tracé}
          sommets={etatImu.sommets}
          positionCourante={etatImu.position}
          pieces={pieces}
        />

        <View style={styles.stats}>
          <StatItem label="Pas" valeur={String(etatImu.nbPas)} />
          <StatItem label="Coins" valeur={String(etatImu.sommets.length)} />
          <StatItem
            label="Distance"
            valeur={`${(etatImu.nbPas * 0.75).toFixed(1)} m`}
          />
        </View>

        <View style={styles.boutons}>
          {!etatImu.actif ? (
            <TouchableOpacity style={[styles.btn, styles.btnVert]} onPress={demarrer}>
              <Text style={styles.btnTexte}>Démarrer</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity style={[styles.btn, styles.btnOrange]} onPress={arreter}>
              <Text style={styles.btnTexte}>Pause</Text>
            </TouchableOpacity>
          )}

          <TouchableOpacity
            style={[styles.btn, styles.btnBleu, etatImu.sommets.length < 3 && styles.btnDesactive]}
            onPress={() => setModalNom(true)}
            disabled={etatImu.sommets.length < 3}
          >
            {chargement ? <ActivityIndicator color="#fff" /> : <Text style={styles.btnTexte}>Enregistrer la pièce</Text>}
          </TouchableOpacity>

          <TouchableOpacity style={[styles.btn, styles.btnGris]} onPress={reinitialiser}>
            <Text style={styles.btnTexte}>Réinitialiser</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity onPress={() => setMode('choix')} style={styles.retour}>
          <Text style={styles.retourTexte}>← Retour</Text>
        </TouchableOpacity>

        {/* Modal nom de pièce */}
        <Modal visible={modalNom} transparent animationType="fade">
          <View style={styles.overlay}>
            <View style={styles.modal}>
              <Text style={styles.modalTitre}>Nom de la pièce</Text>
              <TextInput
                style={styles.input}
                value={nomPiece}
                onChangeText={setNomPiece}
                placeholder="ex : Salon, Chambre…"
                placeholderTextColor="#666"
                autoFocus
              />
              <View style={styles.modalBoutons}>
                <TouchableOpacity style={[styles.btn, styles.btnGris]} onPress={() => setModalNom(false)}>
                  <Text style={styles.btnTexte}>Annuler</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.btn, styles.btnBleu]}
                  onPress={() => { setModalNom(false); sauvegarderPieceImu(nomPiece); }}
                >
                  <Text style={styles.btnTexte}>Enregistrer</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>
      </ScrollView>
    );
  }

  // Mode image
  return (
    <ScrollView style={styles.conteneur} contentContainerStyle={styles.scrollContent}>
      <Text style={styles.titre}>Importer un plan</Text>

      <TouchableOpacity style={styles.zoneDrop} onPress={choisirImage}>
        <Text style={styles.zoneDropTexte}>
          {imageUri ? '✓ Image sélectionnée' : 'Appuyer pour choisir une image'}
        </Text>
      </TouchableOpacity>

      <View style={styles.champEchelle}>
        <Text style={styles.champEchelleLabel}>Échelle (mètres / pixel) :</Text>
        <TextInput
          style={styles.input}
          value={metresParPixel}
          onChangeText={setMetresParPixel}
          placeholder="ex : 0.05"
          placeholderTextColor="#666"
          keyboardType="decimal-pad"
        />
        <Text style={styles.champEchelleAide}>
          Mesurez une distance connue sur l'image et divisez par le nombre de pixels.
        </Text>
      </View>

      <TouchableOpacity
        style={[styles.btn, styles.btnBleu, (!imageUri || chargement) && styles.btnDesactive]}
        onPress={uploaderImage}
        disabled={!imageUri || chargement}
      >
        {chargement ? <ActivityIndicator color="#fff" /> : <Text style={styles.btnTexte}>Importer</Text>}
      </TouchableOpacity>

      <TouchableOpacity onPress={() => setMode('choix')} style={styles.retour}>
        <Text style={styles.retourTexte}>← Retour</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

function StatItem({ label, valeur }: { label: string; valeur: string }) {
  return (
    <View style={{ alignItems: 'center' }}>
      <Text style={{ color: '#4fc3f7', fontSize: 20, fontWeight: 'bold' }}>{valeur}</Text>
      <Text style={{ color: '#888', fontSize: 12 }}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  conteneur: { flex: 1, backgroundColor: '#0d0d1a' },
  scrollContent: { padding: 16, gap: 16 },
  titre: { fontSize: 22, fontWeight: 'bold', color: '#e0e0ff', marginBottom: 4 },
  sousTitre: { fontSize: 14, color: '#888', marginBottom: 20 },
  instruction: { fontSize: 13, color: '#aaa', lineHeight: 20, marginBottom: 8 },
  carteMode: {
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    borderWidth: 1,
    borderColor: '#2a2a4e',
    marginBottom: 12,
  },
  carteModeIcone: { fontSize: 36 },
  carteModeNom: { fontSize: 16, fontWeight: '600', color: '#e0e0ff', marginBottom: 4 },
  carteModeDesc: { fontSize: 13, color: '#888', maxWidth: '85%' },
  piecesSauvees: { marginTop: 20, alignItems: 'center' },
  piecesSaveesTexte: { color: '#4fc3f7', fontSize: 13 },
  stats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: '#1a1a2e',
    borderRadius: 10,
    padding: 16,
  },
  boutons: { gap: 10 },
  btn: { paddingVertical: 14, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  btnTexte: { color: '#fff', fontSize: 15, fontWeight: '600' },
  btnVert: { backgroundColor: '#2e7d32' },
  btnOrange: { backgroundColor: '#e65100' },
  btnBleu: { backgroundColor: '#1565c0' },
  btnGris: { backgroundColor: '#37474f' },
  btnDesactive: { opacity: 0.4 },
  retour: { alignItems: 'center', paddingVertical: 8 },
  retourTexte: { color: '#4fc3f7', fontSize: 14 },
  overlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.7)', justifyContent: 'center', padding: 24 },
  modal: { backgroundColor: '#1a1a2e', borderRadius: 16, padding: 24, gap: 16 },
  modalTitre: { fontSize: 18, fontWeight: '600', color: '#e0e0ff' },
  modalBoutons: { flexDirection: 'row', gap: 12 },
  input: {
    backgroundColor: '#0d0d1a',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2a2a4e',
    padding: 12,
    color: '#e0e0ff',
    fontSize: 15,
  },
  zoneDrop: {
    borderWidth: 2,
    borderColor: '#2a2a4e',
    borderStyle: 'dashed',
    borderRadius: 12,
    padding: 40,
    alignItems: 'center',
  },
  zoneDropTexte: { color: '#4fc3f7', fontSize: 15 },
  champEchelle: { gap: 8 },
  champEchelleLabel: { color: '#ccc', fontSize: 14 },
  champEchelleAide: { color: '#666', fontSize: 12, lineHeight: 18 },
});
