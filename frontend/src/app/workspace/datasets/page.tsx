"use client";

import { ProtectedRoute } from "@/features/auth/protected-route";
import { DatasetsPage } from "@/features/datasets/datasets-page";

export default function WorkspaceDatasetsRoute() {
  return <ProtectedRoute><DatasetsPage /></ProtectedRoute>;
}
