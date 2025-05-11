// API URL базований на конфігурації середовища
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

// Типи даних для сповіщень
export interface NotificationSettings {
  id?: number;
  email: string;
  city_name: string;
  threshold_pm25: number | null;
  threshold_pm10: number | null;
  threshold_o3: number | null;
  threshold_no2: number | null;
  threshold_so2: number | null;
  threshold_co: number | null;
  threshold_aqi: number | null;
  daily_summary: boolean;
  is_active: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface NotificationHistory {
  id: number;
  email: string;
  city: string;
  notification_type: "threshold" | "daily_summary";
  pollutant?: string;
  value?: number;
  threshold?: number;
  message: string;
  sent_at: string;
}

// Інтерфейси для аутентифікації
export interface UserCredentials {
  email: string;
  password: string;
  name?: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  user: {
    id: number;
    email: string;
    name?: string;
    created_at: string;
  };
}

export interface User {
  id: number;
  email: string;
  name?: string;
  created_at: string;
}

// Інтерфейси для даних якості повітря та прогнозів
export interface AirQualityData {
  timestamp: string;
  pm2_5: number;
  pm10: number;
  o3: number;
  no2: number;
  so2: number;
  co: number;
  temperature?: number;
  humidity?: number;
  wind_speed?: number;
  wind_direction?: string;
  datetime?: string;
}

export interface PredictionResult {
  city: string;
  generated_at: string;
  predictions: Array<{
    datetime: string;
    pm2_5: number;
    pm10: number;
    o3: number;
    no2: number;
    so2: number;
    co: number;
    temperature?: number;
    humidity?: number;
    wind_speed?: number;
    wind_direction?: string;
  }>;
}

// Клас для взаємодії з API
class ApiService {
  // Загальний метод для API запитів
  private async fetchApi(endpoint: string, options: RequestInit = {}) {
    try {
      const accessToken = localStorage.getItem("access_token");
      const refreshToken = localStorage.getItem("refresh_token");
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
        ...((options.headers as Record<string, string>) || {}),
      };

      // Add Authorization header if we have an access token
      if (accessToken) {
        headers["Authorization"] = `Bearer ${accessToken}`;
      }

      console.log("Making request to:", `${API_URL}${endpoint}`);
      console.log("Headers:", headers);

      const response = await fetch(`${API_URL}${endpoint}`, {
        ...options,
        headers,
      });

      if (!response.ok) {
        const errorData = await response
          .json()
          .catch(() => ({ detail: "Помилка сервера" }));
        console.error("Error response:", {
          status: response.status,
          statusText: response.statusText,
          data: errorData,
          endpoint,
        });

        // Handle token expiration
        if (response.status === 401 && refreshToken) {
          try {
            // Try to refresh the token
            const refreshResponse = await fetch(`${API_URL}/api/auth/refresh`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${refreshToken}`,
              },
            });

            if (refreshResponse.ok) {
              const { access_token } = await refreshResponse.json();
              localStorage.setItem("access_token", access_token);

              // Retry the original request with new token
              headers["Authorization"] = `Bearer ${access_token}`;
              const retryResponse = await fetch(`${API_URL}${endpoint}`, {
                ...options,
                headers,
              });

              if (retryResponse.ok) {
                return await retryResponse.json();
              }
            }
          } catch (refreshError) {
            console.error("Token refresh failed:", refreshError);
          }

          // If refresh failed or response is still not ok, clear tokens and throw error
          localStorage.removeItem("access_token");
          localStorage.removeItem("refresh_token");
        }

        throw new Error(errorData.detail || `HTTP Error: ${response.status}`);
      }

      const data = await response.json();
      console.log("Response data:", data);
      return data;
    } catch (error) {
      console.error(`API Error (${endpoint}):`, error);
      throw error;
    }
  }

  // --- Методи аутентифікації ---

  // Реєстрація нового користувача
  async register(
    email: string,
    password: string,
    name?: string
  ): Promise<AuthResponse> {
    const response = await this.fetchApi("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password, name }),
    });
    localStorage.setItem("access_token", response.access_token);
    localStorage.setItem("refresh_token", response.refresh_token);
    return response;
  }

  // Вхід користувача
  async login(email: string, password: string): Promise<AuthResponse> {
    try {
      const response = await this.fetchApi("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });

      if (!response.access_token || !response.refresh_token) {
        throw new Error("Invalid response: missing tokens");
      }

      // Store tokens securely
      localStorage.setItem("access_token", response.access_token);
      localStorage.setItem("refresh_token", response.refresh_token);

      return response;
    } catch (error) {
      console.error("Login error:", error);
      throw error;
    }
  }

  // Отримати дані поточного користувача
  async getCurrentUser(): Promise<User> {
    try {
      const accessToken = localStorage.getItem("access_token");
      if (!accessToken) {
        throw new Error("No access token found");
      }

      const response = await this.fetchApi("/api/auth/me", {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });

      if (!response || !response.user) {
        throw new Error("Invalid response format from server");
      }

      return response.user;
    } catch (error) {
      console.error("Error getting current user:", error);
      // Clear invalid tokens
      localStorage.removeItem("access_token");
      localStorage.removeItem("refresh_token");
      throw error;
    }
  }

  // Оновити дані користувача
  async updateUserProfile(data: {
    name?: string;
    email?: string;
  }): Promise<User> {
    return this.fetchApi("/api/auth/profile", {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }

  // Зміна паролю
  async changePassword(
    currentPassword: string,
    newPassword: string
  ): Promise<{ success: boolean }> {
    return this.fetchApi("/api/auth/password", {
      method: "PUT",
      body: JSON.stringify({
        current_password: currentPassword,
        new_password: newPassword,
      }),
    });
  }

  // Отримати прогноз для міста
  async getForecast(city: string) {
    return this.fetchApi(`/api/forecast/${encodeURIComponent(city)}`);
  }

  // Отримати список міст
  async getCities() {
    return this.fetchApi("/api/cities");
  }

  // Отримати історичні дані для міста
  async getHistoricalData(city: string, days: number = 7) {
    return this.fetchApi(
      `/api/historical/${encodeURIComponent(city)}?days=${days}`
    );
  }

  // --- Методи для сповіщень ---

  // Отримати список сповіщень для email
  async getNotifications(email: string): Promise<NotificationSettings[]> {
    return this.fetchApi(
      `/api/notifications?email=${encodeURIComponent(email)}`
    );
  }

  // Отримати сповіщення для поточного користувача
  async getUserNotifications(): Promise<NotificationSettings[]> {
    return this.fetchApi("/api/notifications/user");
  }

  // Створити нове сповіщення
  async createNotification(
    data: NotificationSettings
  ): Promise<NotificationSettings> {
    return this.fetchApi("/api/notifications", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // Оновити сповіщення
  async updateNotification(
    id: number,
    data: Partial<NotificationSettings>
  ): Promise<NotificationSettings> {
    return this.fetchApi(`/api/notifications/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }

  // Видалити сповіщення
  async deleteNotification(id: number): Promise<void> {
    return this.fetchApi(`/api/notifications/${id}`, {
      method: "DELETE",
    });
  }

  // Отримати історію сповіщень для email
  async getNotificationHistory(email: string): Promise<NotificationHistory[]> {
    return this.fetchApi(
      `/api/notifications/history?email=${encodeURIComponent(email)}`
    );
  }

  // Отримати історію сповіщень для поточного користувача
  async getUserNotificationHistory(): Promise<NotificationHistory[]> {
    return this.fetchApi("/api/notifications/user/history");
  }

  // Отримати історичні дані про якість повітря
  async getHistory(city: string, days: number = 7): Promise<AirQualityData[]> {
    return this.fetchApi(
      `/api/air-quality/${encodeURIComponent(city)}/history?days=${days}`
    );
  }

  // Отримати дані якості повітря для міста
  async getAirQuality(
    city: string,
    hours: number = 24
  ): Promise<AirQualityData[]> {
    return this.fetchApi(
      `/api/air-quality/${encodeURIComponent(city)}?hours=${hours}`
    );
  }

  // Отримати прогноз якості повітря для міста
  async getPrediction(
    city: string,
    hours: number = 24
  ): Promise<PredictionResult> {
    return this.fetchApi(
      `/api/predictions/${encodeURIComponent(city)}?hours=${hours}`
    );
  }
}

export const api = new ApiService();
