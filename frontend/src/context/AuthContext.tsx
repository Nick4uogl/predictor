import {
  createContext,
  useState,
  useContext,
  ReactNode,
  useEffect,
} from "react";
import { api } from "../services/api";

// Interface for user data
export interface User {
  id: number;
  email: string;
  name?: string;
  created_at: string;
}

// Interface for auth context value
interface AuthContextValue {
  user: User | null;
  isLoading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name?: string) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
  clearError: () => void;
}

// Create context
const AuthContext = createContext<AuthContextValue | undefined>(undefined);

// Provider component
export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Check if user is logged in on initial load
  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        const accessToken = localStorage.getItem("access_token");
        if (accessToken) {
          const userData = await api.getCurrentUser();
          console.log("userData", userData);
          setUser(userData);
        }
      } catch {
        // Clear invalid tokens
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
      } finally {
        setIsLoading(false);
      }
    };

    checkAuthStatus();
  }, []);

  // Login function
  const login = async (email: string, password: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const { user } = await api.login(email, password);
      setUser(user);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Невідома помилка при вході");
      }
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Register function
  const register = async (email: string, password: string, name?: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const { user } = await api.register(email, password, name);
      setUser(user);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Невідома помилка при реєстрації");
      }
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Logout function
  const logout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    setUser(null);
  };

  // Clear error function
  const clearError = () => {
    setError(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        error,
        login,
        register,
        logout,
        isAuthenticated: !!user,
        clearError,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
