// frontend/src/App.js


import axios from 'axios';

const API_BASE = 'http://192.168.x.x:5000/api'; // Remplacer par l'IP de votre PC

// Exemple d'appel API
const fetchData = async () => {
  try {
    const response = await axios.get(`${API_BASE}/data`);
    console.log(response.data);
  } catch (error) {
    console.error('Erreur connexion API:', error);
  }
};