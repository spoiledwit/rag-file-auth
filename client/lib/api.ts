import axios from "axios";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "https://admin-engine2359.jojen.co.uk/api"
});

api.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("access");
    if (token) {
      //@ts-ignore
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

export default api;
