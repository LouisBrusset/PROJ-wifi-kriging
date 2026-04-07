/**
 * Écran d'accueil : liste des bâtiments cartographiés.
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Modal,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { creerBatiment, listerBatiments, supprimerBatiment } from '../services/api';
import type { Batiment, ParamListeRacine } from '../types';

type NavProp = NativeStackNavigationProp<ParamListeRacine, 'Accueil'>;

export default function AccueilScreen() {
  const navigation = useNavigation<NavProp>();
  const [batiments, setBatiments] = useState<Batiment[]>([]);
  const [chargement, setChargement] = useState(true);
  const [modalVisible, setModalVisible] = useState(false);
  const [nouveauNom, setNouveauNom] = useState('');
  const [nouvelleDesc, setNouvelleDesc] = useState('');
  const [enCreation, setEnCreation] = useState(false);

  const chargerBatiments = useCallback(async () => {
    try {
      setChargement(true);
      const liste = await listerBatiments();
      setBatiments(liste);
    } catch {
      Alert.alert('Erreur', 'Impossible de contacter le serveur.\nVérifiez que le backend est lancé.');
    } finally {
      setChargement(false);
    }
  }, []);

  useEffect(() => {
    const unsub = navigation.addListener('focus', chargerBatiments);
    return unsub;
  }, [navigation, chargerBatiments]);

  const creer = async () => {
    if (!nouveauNom.trim()) { return; }
    setEnCreation(true);
    try {
      const b = await creerBatiment(nouveauNom.trim(), nouvelleDesc.trim() || undefined);
      setBatiments(prev => [b, ...prev]);
      setModalVisible(false);
      setNouveauNom('');
      setNouvelleDesc('');
    } catch {
      Alert.alert('Erreur', 'Création impossible.');
    } finally {
      setEnCreation(false);
    }
  };

  const confirmerSuppression = (b: Batiment) => {
    Alert.alert(
      'Supprimer',
      `Supprimer "${b.nom}" et toutes ses données ?`,
      [
        { text: 'Annuler', style: 'cancel' },
        {
          text: 'Supprimer',
          style: 'destructive',
          onPress: async () => {
            await supprimerBatiment(b.id);
            setBatiments(prev => prev.filter(x => x.id !== b.id));
          },
        },
      ],
    );
  };

  const renderItem = ({ item }: { item: Batiment }) => (
    <View style={styles.carte}>
      <View style={styles.carteInfo}>
        <Text style={styles.carteNom}>{item.nom}</Text>
        {item.description ? (
          <Text style={styles.carteDesc}>{item.description}</Text>
        ) : null}
        <Text style={styles.carteDate}>
          {new Date(item.cree_le).toLocaleDateString('fr-FR')}
        </Text>
      </View>
      <View style={styles.carteBoutons}>
        <TouchableOpacity
          style={[styles.btn, styles.btnPlan]}
          onPress={() => navigation.navigate('Plan', { batimentId: item.id, batimentNom: item.nom })}
        >
          <Text style={styles.btnTexte}>Plan</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.btn, styles.btnMesure]}
          onPress={() => navigation.navigate('Mesure', { batimentId: item.id, batimentNom: item.nom })}
        >
          <Text style={styles.btnTexte}>Mesure</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.btn, styles.btnCarte]}
          onPress={() => navigation.navigate('Carte', { batimentId: item.id, batimentNom: item.nom })}
        >
          <Text style={styles.btnTexte}>Carte</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.btn, styles.btnSuppr]}
          onPress={() => confirmerSuppression(item)}
        >
          <Text style={styles.btnTexte}>✕</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  return (
    <View style={styles.conteneur}>
      <View style={styles.entete}>
        <Text style={styles.titre}>KrigiFi</Text>
        <Text style={styles.sousTitre}>Cartographie WiFi par krigeage</Text>
      </View>

      {chargement ? (
        <ActivityIndicator size="large" color="#4fc3f7" style={{ marginTop: 40 }} />
      ) : (
        <FlatList
          data={batiments}
          keyExtractor={item => String(item.id)}
          renderItem={renderItem}
          contentContainerStyle={styles.liste}
          ListEmptyComponent={
            <View style={styles.vide}>
              <Text style={styles.videTexte}>Aucun bâtiment.</Text>
              <Text style={styles.videTexte}>Appuyez sur + pour commencer.</Text>
            </View>
          }
        />
      )}

      {/* Bouton + */}
      <TouchableOpacity style={styles.fab} onPress={() => setModalVisible(true)}>
        <Text style={styles.fabTexte}>+</Text>
      </TouchableOpacity>

      {/* Modal de création */}
      <Modal visible={modalVisible} transparent animationType="fade">
        <View style={styles.overlay}>
          <View style={styles.modal}>
            <Text style={styles.modalTitre}>Nouveau bâtiment</Text>
            <TextInput
              style={styles.input}
              placeholder="Nom (ex : Appartement Paris)"
              placeholderTextColor="#666"
              value={nouveauNom}
              onChangeText={setNouveauNom}
              autoFocus
            />
            <TextInput
              style={styles.input}
              placeholder="Description (optionnel)"
              placeholderTextColor="#666"
              value={nouvelleDesc}
              onChangeText={setNouvelleDesc}
            />
            <View style={styles.modalBoutons}>
              <TouchableOpacity
                style={[styles.btn, styles.btnAnnuler]}
                onPress={() => setModalVisible(false)}
              >
                <Text style={styles.btnTexte}>Annuler</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.btn, styles.btnCreer, !nouveauNom.trim() && styles.btnDesactive]}
                onPress={creer}
                disabled={!nouveauNom.trim() || enCreation}
              >
                {enCreation ? (
                  <ActivityIndicator size="small" color="#fff" />
                ) : (
                  <Text style={styles.btnTexte}>Créer</Text>
                )}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  conteneur: { flex: 1, backgroundColor: '#0d0d1a' },
  entete: {
    paddingTop: 20,
    paddingBottom: 16,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#1e1e3a',
  },
  titre: { fontSize: 28, fontWeight: 'bold', color: '#4fc3f7' },
  sousTitre: { fontSize: 13, color: '#888', marginTop: 2 },
  liste: { padding: 16, gap: 12 },
  carte: {
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: '#2a2a4e',
  },
  carteInfo: { marginBottom: 12 },
  carteNom: { fontSize: 17, fontWeight: '600', color: '#e0e0ff' },
  carteDesc: { fontSize: 13, color: '#888', marginTop: 3 },
  carteDate: { fontSize: 11, color: '#555', marginTop: 4 },
  carteBoutons: { flexDirection: 'row', gap: 8 },
  btn: { paddingHorizontal: 14, paddingVertical: 8, borderRadius: 8, alignItems: 'center', justifyContent: 'center' },
  btnTexte: { color: '#fff', fontSize: 13, fontWeight: '500' },
  btnPlan: { backgroundColor: '#1565c0', flex: 1 },
  btnMesure: { backgroundColor: '#2e7d32', flex: 1 },
  btnCarte: { backgroundColor: '#6a1b9a', flex: 1 },
  btnSuppr: { backgroundColor: '#b71c1c', paddingHorizontal: 10 },
  btnAnnuler: { backgroundColor: '#333', flex: 1 },
  btnCreer: { backgroundColor: '#1565c0', flex: 1 },
  btnDesactive: { opacity: 0.5 },
  fab: {
    position: 'absolute',
    bottom: 24,
    right: 24,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#4fc3f7',
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 6,
  },
  fabTexte: { fontSize: 28, color: '#0d0d1a', lineHeight: 32 },
  vide: { alignItems: 'center', marginTop: 80, gap: 8 },
  videTexte: { color: '#555', fontSize: 15 },
  overlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.7)', justifyContent: 'center', padding: 24 },
  modal: { backgroundColor: '#1a1a2e', borderRadius: 16, padding: 24, gap: 16 },
  modalTitre: { fontSize: 20, fontWeight: '600', color: '#e0e0ff' },
  input: {
    backgroundColor: '#0d0d1a',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2a2a4e',
    padding: 12,
    color: '#e0e0ff',
    fontSize: 15,
  },
  modalBoutons: { flexDirection: 'row', gap: 12 },
});
