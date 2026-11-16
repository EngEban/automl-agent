"use client";

import { ProtectedRoute } from "@/features/auth/protected-route";
import { DashboardPage } from "@/features/dashboard/dashboard-page";

export default function WorkspacePage() {
  return <ProtectedRoute><DashboardPage /></ProtectedRoute>;
}
