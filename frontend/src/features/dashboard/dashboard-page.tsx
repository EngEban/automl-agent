"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowUpRight, Database, Plus } from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { Alert } from "@/components/ui/feedback";
import { Button } from "@/components/ui/button";
import { ApiError, ApiTimeoutError } from "@/lib/api/client";
import { useAuth } from "@/features/auth/auth-provider";
import { listDashboardSessions } from "@/features/dashboard/dashboard-api";
import {
  DashboardMetrics,
  DashboardQuickActions,
  RecentActivity,
  RecentAnalyses,
  RecentDatasets,
} from "@/features/dashboard/dashboard-sections";
import type { DashboardLoadState } from "@/features/dashboard/dashboard-sections";
import { UploadDatasetDialog } from "@/features/dashboard/upload-dataset-dialog";
import type { DashboardSession } from "@/types/api";

function dashboardErrorMessage(error: unknown): string {
  if (error instanceof ApiTimeoutError) return "The request took too long. Check your connection and try again.";
  if (error instanceof ApiError && error.status >= 500) return "The service is temporarily unavailable. Please try again shortly.";
  if (error instanceof ApiError && error.status === 422) return "The session list request was rejected. Please try again.";
  if (error instanceof TypeError) return "We could not reach the service. Check your connection and try again.";
  return "Your dataset sessions could not be loaded. Please try again.";
}

function DashboardPageContent() {
  const router = useRouter();
  const { accessToken, user, logout } = useAuth();
  const [sessions, setSessions] = useState<DashboardSession[]>([]);
  const [loadState, setLoadState] = useState<DashboardLoadState>("loading");
  const [sessionsReload, setSessionsReload] = useState(0);
  const [loadError, setLoadError] = useState("");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadNotice, setUploadNotice] = useState("");
  const sessionsRequest = useRef<{ token: string; promise: Promise<DashboardSession[]> } | null>(null);
  const sessionsLoadLock = useRef(false);
  const sessionsRevision = useRef(0);

  useEffect(() => {
    if (!accessToken) return;
    let active = true;
    const requestRevision = sessionsRevision.current;
    let request = sessionsRequest.current;
    if (!request || request.token !== accessToken) {
      request = { token: accessToken, promise: listDashboardSessions(accessToken) };
      sessionsRequest.current = request;
    }
    sessionsLoadLock.current = true;
    request.promise
      .then((result) => {
        if (!active || requestRevision !== sessionsRevision.current) return;
        setSessions(result);
        setLoadState("ready");
        setLoadError("");
      })
      .catch((error: unknown) => {
        if (!active || requestRevision !== sessionsRevision.current) return;
        if (sessionsRequest.current === request) sessionsRequest.current = null;
        if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
          logout();
          return;
        }
        setLoadError(dashboardErrorMessage(error));
        setLoadState("error");
      })
      .finally(() => {
        if (active && requestRevision === sessionsRevision.current) sessionsLoadLock.current = false;
      });
    return () => { active = false; };
  }, [accessToken, logout, sessionsReload]);

  const retrySessions = useCallback(() => {
    if (sessionsLoadLock.current) return;
    sessionsRevision.current += 1;
    sessionsRequest.current = null;
    setLoadState("loading");
    setLoadError("");
    setSessionsReload((current) => current + 1);
  }, []);

  const handleSessionCreated = useCallback((created: DashboardSession) => {
    sessionsRevision.current += 1;
    sessionsRequest.current = null;
    sessionsLoadLock.current = false;
    setSessions((current) => [created, ...current.filter((session) => session.id !== created.id)].slice(0, 50));
    setLoadState("ready");
    setLoadError("");
    router.push(`/workspace/agent?session_id=${encodeURIComponent(created.id)}`);
  }, [router]);

  const welcomeDescription = user?.email
    ? `Signed in as ${user.email}. Review your recent dataset sessions and prepare your next analysis.`
    : "Review your recent dataset sessions and prepare your next analysis.";

  return (
    <>
      <AppShell
        title="Welcome back"
        eyebrow="YOUR WORKSPACE"
        description={welcomeDescription}
        breadcrumbs={[{ label: "Workspace" }, { label: "Overview" }]}
        activeItem="Overview"
        homeHref="/workspace"
        statusLabel="Signed in"
        actions={(
          <div className="dashboard-header-actions">
            <Image
              className="dashboard-header-illustration"
              src="/images/automl-agent-illustrations/01-hero-ai-agent.png"
              width={815}
              height={306}
              alt=""
              aria-hidden="true"
              priority
            />
            <Button onClick={() => { setUploadNotice(""); setUploadOpen(true); }} leadingIcon={<Plus size={15} aria-hidden="true" />}>New Analysis</Button>
            <Link className="button button-outline" href="#recent-datasets"><Database size={14} aria-hidden="true" /> View datasets</Link>
          </div>
        )}
      >
        <div className="dashboard-stack">
          {uploadNotice && <Alert tone="success" title="Dataset ready">{uploadNotice}</Alert>}
          {loadError && (
            <div className="dashboard-error-row">
              <Alert tone="error" title="Dashboard data unavailable">{loadError}</Alert>
              <Button variant="outline" onClick={retrySessions}>Try again</Button>
            </div>
          )}

          <DashboardMetrics sessionCount={sessions.length} state={loadState} />
          <DashboardQuickActions onUpload={() => { setUploadNotice(""); setUploadOpen(true); }} />

          <div className="dashboard-two-column">
            <RecentActivity
              sessions={sessions}
              state={loadState}
              onUpload={() => { setUploadNotice(""); setUploadOpen(true); }}
            />
            <RecentAnalyses />
          </div>

          <RecentDatasets
            sessions={sessions}
            state={loadState}
            onUpload={() => { setUploadNotice(""); setUploadOpen(true); }}
          />
          <p className="dashboard-data-note"><ArrowUpRight size={13} aria-hidden="true" /> Only data available for your account is shown.</p>
        </div>
      </AppShell>

      {uploadOpen && accessToken && (
        <UploadDatasetDialog
          open
          token={accessToken}
          onClose={() => setUploadOpen(false)}
          onCreated={handleSessionCreated}
        />
      )}
    </>
  );
}

export function DashboardPage() {
  return <DashboardPageContent />;
}
