/**
 * Service de communication avec l'API FastAPI backend.
 * Toutes les requêtes HTTP passent par ce module.
 */

import axios from 'axios';
import {
  Batiment,
  Piece,
  Point,
  MesureWifi,
  ResultatKrigeage,
  ImagePlan,
} from '../types';

// adb reverse tcp:8000 tcp:8000 → le téléphone atteint le backend via localhost
// Pour émulateur : remplacer par http://10.0.2.2:8000
//const BASE_URL = 'http://localhost:8000';
const BASE_URL = 'http://localhost:8000';

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Bâtiments ─────────────────────────────────────────────────────────────────

export async function listerBatiments(): Promise<Batiment[]> {
  const { data } = await client.get<Batiment[]>('/batiments/');
  return data;
}

export async function creerBatiment(nom: string, description?: string): Promise<Batiment> {
  const { data } = await client.post<Batiment>('/batiments/', { nom, description });
  return data;
}

export async function obtenirBatiment(id: number): Promise<Batiment> {
  const { data } = await client.get<Batiment>(`/batiments/${id}`);
  return data;
}

export async function supprimerBatiment(id: number): Promise<void> {
  await client.delete(`/batiments/${id}`);
}

// ── Pièces ────────────────────────────────────────────────────────────────────

export async function listerPieces(batimentId: number): Promise<Piece[]> {
  const { data } = await client.get<Piece[]>(`/batiments/${batimentId}/pieces`);
  return data;
}

export async function creerPiece(batimentId: number, nom: string, points: Point[]): Promise<Piece> {
  const { data } = await client.post<Piece>(`/batiments/${batimentId}/pieces`, { nom, points });
  return data;
}

export async function mettreAJourPiece(
  batimentId: number,
  pieceId: number,
  nom: string,
  points: Point[],
): Promise<Piece> {
  const { data } = await client.put<Piece>(`/batiments/${batimentId}/pieces/${pieceId}`, {
    nom,
    points,
  });
  return data;
}

export async function supprimerPiece(batimentId: number, pieceId: number): Promise<void> {
  await client.delete(`/batiments/${batimentId}/pieces/${pieceId}`);
}

// ── Plan image ────────────────────────────────────────────────────────────────

export async function uploaderPlan(batimentId: number, uri: string): Promise<ImagePlan> {
  const formData = new FormData();
  const nomFichier = uri.split('/').pop() ?? 'plan.jpg';
  formData.append('fichier', {
    uri,
    type: 'image/jpeg',
    name: nomFichier,
  } as any);

  const { data } = await axios.post<ImagePlan>(
    `${BASE_URL}/batiments/${batimentId}/plan-image`,
    formData,
    { headers: { 'Content-Type': 'multipart/form-data' } },
  );
  return data;
}

export async function definirEchelle(
  batimentId: number,
  metresParPixel: number,
): Promise<ImagePlan> {
  const { data } = await client.patch<ImagePlan>(
    `/batiments/${batimentId}/plan-image/echelle`,
    { metres_par_pixel: metresParPixel },
  );
  return data;
}

// ── Mesures WiFi ──────────────────────────────────────────────────────────────

export async function listerMesures(batimentId: number): Promise<MesureWifi[]> {
  const { data } = await client.get<MesureWifi[]>(`/batiments/${batimentId}/mesures`);
  return data;
}

export async function ajouterMesures(
  batimentId: number,
  mesures: Array<{ x: number; y: number; rssi: number; ssid?: string; type_reseau?: string }>,
): Promise<MesureWifi[]> {
  const { data } = await client.post<MesureWifi[]>(
    `/batiments/${batimentId}/mesures/lot`,
    mesures,
  );
  return data;
}

export async function effacerMesures(batimentId: number): Promise<void> {
  await client.delete(`/batiments/${batimentId}/mesures`);
}

// ── Krigeage ──────────────────────────────────────────────────────────────────

export async function calculerHeatmap(
  batimentId: number,
  resolution: number = 50,
  variogramme: string = 'spherical',
): Promise<ResultatKrigeage> {
  const { data } = await client.post<ResultatKrigeage>(`/batiments/${batimentId}/krigeage`, {
    resolution,
    variogramme,
  });
  return data;
}
