/**
 * Composant de dessin du plan au sol.
 * Affiche les pièces (polygones), la trajectoire IMU et la position courante.
 */

import React, { useMemo } from 'react';
import { View, StyleSheet, Text } from 'react-native';
import Svg, { Polyline, Polygon, Circle, Line, Text as SvgText } from 'react-native-svg';
import type { Point, Piece } from '../types';

interface Props {
  largeur: number;
  hauteur: number;
  tracé: Point[];           // Trajectoire IMU complète
  sommets: Point[];         // Coins détectés (virages)
  positionCourante: Point;
  pieces?: Piece[];         // Pièces déjà sauvegardées
  afficherGrille?: boolean;
}

const COULEURS = {
  fond: '#1a1a2e',
  grille: '#2a2a4e',
  tracé: '#4fc3f7',
  sommet: '#ff6b6b',
  piece: '#00e5ff',
  pieceRemplissage: 'rgba(0, 229, 255, 0.1)',
  position: '#ffeb3b',
  texte: '#ffffff',
};

/**
 * Convertit les coordonnées métriques en pixels SVG.
 * Cherche les bornes dans tous les points disponibles.
 */
function utiliserTransformation(
  tousLesPoints: Point[],
  largeur: number,
  hauteur: number,
): (p: Point) => { px: number; py: number } {
  if (tousLesPoints.length === 0) {
    return () => ({ px: largeur / 2, py: hauteur / 2 });
  }

  const xs = tousLesPoints.map(p => p.x);
  const ys = tousLesPoints.map(p => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);

  const marge = 30;
  const plageX = xMax - xMin || 1;
  const plageY = yMax - yMin || 1;
  const echelle = Math.min((largeur - 2 * marge) / plageX, (hauteur - 2 * marge) / plageY);

  return (p: Point) => ({
    px: marge + (p.x - xMin) * echelle,
    py: hauteur - marge - (p.y - yMin) * echelle,
  });
}

export default function DessinateurPlan({
  largeur,
  hauteur,
  tracé,
  sommets,
  positionCourante,
  pieces = [],
  afficherGrille = true,
}: Props) {
  const tousLesPoints = useMemo(
    () => [
      ...tracé,
      ...sommets,
      positionCourante,
      ...pieces.flatMap(p => p.points),
    ],
    [tracé, sommets, positionCourante, pieces],
  );

  const vers = useMemo(
    () => utiliserTransformation(tousLesPoints, largeur, hauteur),
    [tousLesPoints, largeur, hauteur],
  );

  const polylineTrce = useMemo(
    () => tracé.map(p => vers(p)).map(({ px, py }) => `${px},${py}`).join(' '),
    [tracé, vers],
  );

  const pos = vers(positionCourante);

  return (
    <View style={[styles.conteneur, { width: largeur, height: hauteur }]}>
      <Svg width={largeur} height={hauteur}>

        {/* Fond */}
        <Polygon
          points={`0,0 ${largeur},0 ${largeur},${hauteur} 0,${hauteur}`}
          fill={COULEURS.fond}
        />

        {/* Pièces sauvegardées */}
        {pieces.map((piece) => {
          const pts = piece.points.map(p => vers(p));
          const polyPoints = pts.map(({ px, py }) => `${px},${py}`).join(' ');
          const centre = pts.reduce(
            (acc, { px, py }) => ({ px: acc.px + px / pts.length, py: acc.py + py / pts.length }),
            { px: 0, py: 0 },
          );
          return (
            <React.Fragment key={piece.id}>
              <Polygon
                points={polyPoints}
                fill={COULEURS.pieceRemplissage}
                stroke={COULEURS.piece}
                strokeWidth={1.5}
              />
              <SvgText
                x={centre.px}
                y={centre.py}
                textAnchor="middle"
                fill={COULEURS.texte}
                fontSize={10}
              >
                {piece.nom}
              </SvgText>
            </React.Fragment>
          );
        })}

        {/* Trajectoire IMU */}
        {tracé.length > 1 && (
          <Polyline
            points={polylineTrce}
            fill="none"
            stroke={COULEURS.tracé}
            strokeWidth={2}
            strokeDasharray="4,2"
          />
        )}

        {/* Sommets (coins détectés) */}
        {sommets.map((s, i) => {
          const { px, py } = vers(s);
          return (
            <Circle key={i} cx={px} cy={py} r={5} fill={COULEURS.sommet} opacity={0.8} />
          );
        })}

        {/* Position courante */}
        <Circle cx={pos.px} cy={pos.py} r={8} fill={COULEURS.position} opacity={0.9} />
        <Circle cx={pos.px} cy={pos.py} r={14} fill="none" stroke={COULEURS.position} strokeWidth={1.5} opacity={0.5} />

      </Svg>

      {/* Légende */}
      <View style={styles.legende}>
        <Text style={styles.legendeTitre}>
          {tracé.length > 1 ? `${tracé.length} pts · ${sommets.length} coins` : 'Déplacez-vous'}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  conteneur: {
    backgroundColor: COULEURS.fond,
    borderRadius: 12,
    overflow: 'hidden',
  },
  legende: {
    position: 'absolute',
    bottom: 8,
    left: 8,
    backgroundColor: 'rgba(0,0,0,0.5)',
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  legendeTitre: {
    color: '#aaa',
    fontSize: 11,
  },
});
