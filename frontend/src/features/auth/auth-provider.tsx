"use client";

import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { ApiError, apiRequest } from "@/lib/api/client";
import type {
  AuthTokenResponse,
  LoginRequest,
  RegisterRequest,
  RegisterResponse,
  UserRead,
} from "@/types/api";

export type AuthStatus = "loading" | "authenticated" | "unauthenticated";

interface AuthContextValue {
  status: AuthStatus;
  user: UserRead | null;
  accessToken: string | null;
  isAuthenticated: boolean;
  login: (credentials: LoginRequest) => Promise<void>;
  register: (details: RegisterRequest) => Promise<RegisterResponse>;
  logout: () => void;
  verifySession: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

function isAuthResponse(value: AuthTokenResponse): boolean {
  return Boolean(
    value &&
      typeof value.access_token === "string" &&
      value.access_token.length > 0 &&
      typeof value.token_type === "string" &&
      value.token_type.toLowerCase() === "bearer" &&
      value.user &&
      typeof value.user.id === "string" &&
      typeof value.user.email === "string" &&
      value.user.is_active === true,
  );
}

export function AuthProvider({ children }: { children: ReactNode }) {
  // The backend returns a bearer JWT without cookie or refresh support; keep it in memory only.
  const [status, setStatus] = useState<AuthStatus>("unauthenticated");
  const [user, setUser] = useState<UserRead | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);

  const login = useCallback(async (credentials: LoginRequest) => {
    setStatus("loading");
    try {
      const result = await apiRequest<AuthTokenResponse, LoginRequest>("/auth/login", {
        method: "POST",
        body: credentials,
      });
      if (!isAuthResponse(result)) throw new Error("Unexpected authentication response.");
      setAccessToken(result.access_token);
      setUser(result.user);
      setStatus("authenticated");
    } catch (error) {
      setAccessToken(null);
      setUser(null);
      setStatus("unauthenticated");
      throw error;
    }
  }, []);

  const register = useCallback((details: RegisterRequest) =>
    apiRequest<RegisterResponse, RegisterRequest>("/auth/register", {
      method: "POST",
      body: details,
    }), []);

  const logout = useCallback(() => {
    // There is no backend logout endpoint; this clears the client-held bearer token.
    setAccessToken(null);
    setUser(null);
    setStatus("unauthenticated");
  }, []);

  const verifySession = useCallback(async () => {
    if (!accessToken) return false;
    try {
      const currentUser = await apiRequest<UserRead>("/auth/me", { token: accessToken });
      setUser(currentUser);
      return true;
    } catch (error) {
      if (error instanceof ApiError && (error.status === 401 || error.status === 403)) logout();
      throw error;
    }
  }, [accessToken, logout]);

  const value = useMemo<AuthContextValue>(() => ({
    status,
    user,
    accessToken,
    isAuthenticated: status === "authenticated" && Boolean(accessToken && user),
    login,
    register,
    logout,
    verifySession,
  }), [status, user, accessToken, login, register, logout, verifySession]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within AuthProvider.");
  return context;
}
