// Types partagés dans toute l'application KrigiFi

export interface Point {
  x: number;
  y: number;
}

export interface Batiment {
  id: number;
  nom: string;
  description?: string;
  cree_le: string;
}

export interface Piece {
  id: number;
  batiment_id: number;
  nom: string;
  points: Point[];
  cree_le: string;
}

export interface ImagePlan {
  id: number;
  batiment_id: number;
  chemin_fichier: string;
  metres_par_pixel?: number;
  largeur_px?: number;
  hauteur_px?: number;
}

export interface MesureWifi {
  id: number;
  batiment_id: number;
  x: number;
  y: number;
  rssi: number;
  ssid?: string;
  type_reseau: 'wifi' | 'mobile';
  mesure_le: string;
}

export interface CelluleHeatmap {
  x: number;
  y: number;
  valeur: number;
  variance: number;
}

export interface ResultatKrigeage {
  cellules: CelluleHeatmap[];
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
  rssi_min: number;
  rssi_max: number;
  resolution: number;
}

export type EtatCapture = 'repos' | 'en_cours' | 'pause' | 'termine';

export interface DonneesImu {
  x: number;
  y: number;
  z: number;
  timestamp: number;
}

// Paramètres de navigation
export type ParamListeRacine = {
  Accueil: undefined;
  Plan: { batimentId: number; batimentNom: string };
  Mesure: { batimentId: number; batimentNom: string };
  Carte: { batimentId: number; batimentNom: string };
};
