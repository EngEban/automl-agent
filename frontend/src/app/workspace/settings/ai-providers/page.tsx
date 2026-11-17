"use client";

import { ProtectedRoute } from "@/features/auth/protected-route";
import { AIProviderSettingsPage } from "@/features/settings/ai-providers/ai-provider-settings-page";

export default function AIProviderSettingsRoute() {
  return (
    <ProtectedRoute>
      <AIProviderSettingsPage />
    </ProtectedRoute>
  );
}
