import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

export const assessTransaction = async (payload) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/assess`, payload);
    return response.data;
  } catch (error) {
    console.error('API Error during transaction assessment:', error);
    throw error;
  }
};