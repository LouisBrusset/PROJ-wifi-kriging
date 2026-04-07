/**
 * Composant d'affichage de la heatmap WiFi.
 * Superpose une grille colorée sur le plan du bâtiment.
 */

import React, { useMemo } from 'react';
import { View, StyleSheet, Text } from 'react-native';
import Svg, { Rect, Circle, Text as SvgText, Defs, LinearGradient, Stop } from 'react-native-svg';
import type { ResultatKrigeage, MesureWifi, Piece } from '../types';

interface Props {
  largeur: number;
  hauteur: number;
  resultat: ResultatKrigeage;
  mesures?: MesureWifi[];
  pieces?: Piece[];
  opacite?: number;
}

/**
 * Convertit une valeur RSSI (dBm) en couleur RGBA.
 * Rouge = signal fort, Bleu = signal faible.
 * Plage typique : -90 dBm (faible) à -30 dBm (fort)
 */
function rssiVersCouleur(rssi: number, rssiMin: number, rssiMax: number): string {
  const t = Math.max(0, Math.min(1, (rssi - rssiMin) / (rssiMax - rssiMin || 1)));

  // Gradient : Bleu → Cyan → Vert → Jaune → Rouge
  let r: number, g: number, b: number;
  if (t < 0.25) {
    const s = t / 0.25;
    r = 0; g = Math.round(255 * s); b = 255;
  } else if (t < 0.5) {
    const s = (t - 0.25) / 0.25;
    r = 0; g = 255; b = Math.round(255 * (1 - s));
  } else if (t < 0.75) {
    const s = (t - 0.5) / 0.25;
    r = Math.round(255 * s); g = 255; b = 0;
  } else {
    const s = (t - 0.75) / 0.25;
    r = 255; g = Math.round(255 * (1 - s)); b = 0;
  }

  return `rgb(${r},${g},${b})`;
}

export default function CarteThermique({
  largeur,
  hauteur,
  resultat,
  mesures = [],
  pieces = [],
  opacite = 0.75,
}: Props) {
  const { x_min, x_max, y_min, y_max, rssi_min, rssi_max, resolution } = resultat;

  // Transformation coordonnées métriques → pixels SVG
  const marge = 20;
  const plageX = x_max - x_min || 1;
  const plageY = y_max - y_min || 1;
  const echelle = Math.min((largeur - 2 * marge) / plageX, (hauteur - 2 * marge) / plageY);

  const vers = (x: number, y: number) => ({
    px: marge + (x - x_min) * echelle,
    py: hauteur - marge - (y - y_min) * echelle,
  });

  const tailleCellule = Math.max(1, echelle * (plageX / resolution));

  // Graduation de la légende
  const echelons = useMemo(() => {
    const nb = 5;
    return Array.from({ length: nb }, (_, i) => {
      const t = i / (nb - 1);
      const valeur = rssi_min + t * (rssi_max - rssi_min);
      return { t, valeur: Math.round(valeur), couleur: rssiVersCouleur(valeur, rssi_min, rssi_max) };
    });
  }, [rssi_min, rssi_max]);

  return (
    <View style={[styles.conteneur, { width: largeur, height: hauteur }]}>
      <Svg width={largeur} height={hauteur}>

        {/* Fond */}
        <Rect x={0} y={0} width={largeur} height={hauteur} fill="#0d0d1a" />

        {/* Cellules de la heatmap */}
        {resultat.cellules.map((cellule, i) => {
          const { px, py } = vers(cellule.x, cellule.y);
          const couleur = rssiVersCouleur(cellule.valeur, rssi_min, rssi_max);
          return (
            <Rect
              key={i}
              x={px - tailleCellule / 2}
              y={py - tailleCellule / 2}
              width={tailleCellule}
              height={tailleCellule}
              fill={couleur}
              opacity={opacite}
            />
          );
        })}

        {/* Contours des pièces */}
        {pieces.map(piece => {
          if (piece.points.length < 2) { return null; }
          const points = piece.points.map(p => {
            const { px, py } = vers(p.x, p.y);
            return `${px},${py}`;
          }).join(' ');
          return (
            <React.Fragment key={piece.id}>
              <SvgText />
            </React.Fragment>
          );
        })}

        {/* Points de mesure */}
        {mesures.map((m, i) => {
          const { px, py } = vers(m.x, m.y);
          return (
            <Circle key={i} cx={px} cy={py} r={4} fill="white" opacity={0.6} />
          );
        })}

      </Svg>

      {/* Légende verticale */}
      <View style={styles.legende}>
        {[...echelons].reverse().map((e, i) => (
          <View key={i} style={styles.echelon}>
            <View style={[styles.carreEchelon, { backgroundColor: e.couleur }]} />
            <Text style={styles.texteEchelon}>{e.valeur} dBm</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  conteneur: {
    backgroundColor: '#0d0d1a',
    borderRadius: 12,
    overflow: 'hidden',
  },
  legende: {
    position: 'absolute',
    right: 8,
    top: 8,
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: 8,
    padding: 6,
    gap: 4,
  },
  echelon: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
  },
  carreEchelon: {
    width: 12,
    height: 12,
    borderRadius: 2,
  },
  texteEchelon: {
    color: '#ccc',
    fontSize: 9,
  },
});
