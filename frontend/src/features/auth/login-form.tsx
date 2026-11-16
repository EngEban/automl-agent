"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Alert } from "@/components/ui/feedback";
import { PasswordInput, TextInput } from "@/components/ui/form-controls";
import { useAuth } from "@/features/auth/auth-provider";
import { AuthError } from "@/features/auth/auth-form-parts";
import { authErrorMessage } from "@/features/auth/auth-utils";

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function LoginForm({
  destination,
  registered,
  verified,
}: {
  destination: string;
  registered: boolean;
  verified: boolean;
}) {
  const router = useRouter();
  const { status, isAuthenticated, login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fieldErrors, setFieldErrors] = useState<{ email?: string; password?: string }>({});
  const [requestError, setRequestError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const submitLock = useRef(false);

  useEffect(() => {
    if (isAuthenticated) router.replace(destination);
  }, [isAuthenticated, destination, router]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitLock.current) return;
    const nextErrors: { email?: string; password?: string } = {};
    const normalizedEmail = email.trim();
    if (!EMAIL_PATTERN.test(normalizedEmail)) nextErrors.email = "Enter a valid email address.";
    if (!password) nextErrors.password = "Enter your password.";
    setFieldErrors(nextErrors);
    setRequestError("");
    if (Object.keys(nextErrors).length) return;

    submitLock.current = true;
    setSubmitting(true);
    try {
      await login({ email: normalizedEmail, password });
    } catch (error) {
      setRequestError(authErrorMessage(error, "login"));
    } finally {
      submitLock.current = false;
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-form-content-inner">
      <div className="auth-page-heading">
        <span className="auth-form-eyebrow">YOUR WORKSPACE AWAITS</span>
        <h1>Welcome back</h1>
        <p>Sign in to continue your data science work.</p>
      </div>

      {verified && <Alert tone="success" title="Email verified">Your account is active. You can sign in now.</Alert>}
      {registered && <Alert tone="success" title="Account created">Email verification is required before sign-in. Check your inbox for the verification link.</Alert>}
      {requestError && <AuthError message={requestError} />}

      <form className="auth-form" onSubmit={handleSubmit} noValidate>
        <TextInput
          label="Email address"
          type="email"
          name="email"
          autoComplete="email"
          inputMode="email"
          placeholder="you@example.com"
          value={email}
          onChange={(event) => {
            setEmail(event.target.value);
            setFieldErrors((current) => ({ ...current, email: undefined }));
          }}
          error={fieldErrors.email}
          required
        />
        <PasswordInput
          label="Password"
          name="password"
          autoComplete="current-password"
          placeholder="Enter your password"
          value={password}
          onChange={(event) => {
            setPassword(event.target.value);
            setFieldErrors((current) => ({ ...current, password: undefined }));
          }}
          error={fieldErrors.password}
          required
        />
        <Button className="auth-submit" type="submit" loading={submitting} disabled={status === "loading" && !submitting}>
          Sign in
        </Button>
      </form>

      <p className="auth-switch-copy">New to AutoML Agent? <Link href={`/register?next=${encodeURIComponent(destination)}`}>Create an account</Link></p>
    </div>
  );
}
