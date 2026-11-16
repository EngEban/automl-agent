"use client";

import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/feedback";
import { useAuth } from "@/features/auth/auth-provider";

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { status, user, isAuthenticated, verifySession } = useAuth();
  const pathname = usePathname();
  const router = useRouter();
  const verificationKey = useMemo(() => ({
    pathname,
    userId: user?.id,
    status,
    isAuthenticated,
    verifySession,
  }), [pathname, user?.id, status, isAuthenticated, verifySession]);
  const [verificationResult, setVerificationResult] = useState<{
    key: typeof verificationKey;
    status: "checking" | "verified" | "failed";
  } | null>(null);
  const verificationLock = useRef(false);
  const verification = verificationResult?.key === verificationKey
    ? verificationResult.status
    : "checking";

  useEffect(() => {
    if (status === "unauthenticated") {
      router.replace(`/login?next=${encodeURIComponent(pathname || "/workspace")}`);
    }
  }, [status, pathname, router]);

  useEffect(() => {
    if (status !== "authenticated") return;
    let active = true;
    verifySession()
      .then((valid) => {
        if (!active) return;
        if (valid) setVerificationResult({ key: verificationKey, status: "verified" });
        else router.replace(`/login?next=${encodeURIComponent(pathname || "/workspace")}`);
      })
      .catch(() => {
        if (active) setVerificationResult({ key: verificationKey, status: "failed" });
      });
    return () => { active = false; };
  }, [status, pathname, router, verifySession, verificationKey]);

  const retryVerification = useCallback(async () => {
    if (verificationLock.current) return;
    verificationLock.current = true;
    setVerificationResult({ key: verificationKey, status: "checking" });
    try {
      const valid = await verifySession();
      if (valid) setVerificationResult({ key: verificationKey, status: "verified" });
      else router.replace(`/login?next=${encodeURIComponent(pathname || "/workspace")}`);
    } catch {
      setVerificationResult({ key: verificationKey, status: "failed" });
    } finally {
      verificationLock.current = false;
    }
  }, [verifySession, router, pathname, verificationKey]);

  if (status !== "authenticated" || !isAuthenticated || verification === "checking") {
    return (
      <main className="protected-loading" aria-live="polite" aria-busy="true">
        <Spinner size="lg" />
        <p>{status === "unauthenticated" ? "Taking you to sign in…" : "Checking your session…"}</p>
      </main>
    );
  }

  if (verification === "failed") {
    return (
      <main className="protected-loading" aria-live="polite">
        <p>We couldn’t verify your session. Check your connection and try again.</p>
        <Button variant="outline" onClick={retryVerification}>Try again</Button>
      </main>
    );
  }

  return <>{children}</>;
}
