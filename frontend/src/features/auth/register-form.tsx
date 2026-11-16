"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { PasswordInput, TextInput } from "@/components/ui/form-controls";
import { AuthError, PasswordStrength } from "@/features/auth/auth-form-parts";
import { useAuth } from "@/features/auth/auth-provider";
import { authErrorMessage } from "@/features/auth/auth-utils";

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function RegisterForm({ destination }: { destination: string }) {
  const router = useRouter();
  const { isAuthenticated, register } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [fieldErrors, setFieldErrors] = useState<{ email?: string; password?: string; confirmation?: string }>({});
  const [requestError, setRequestError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const submitLock = useRef(false);

  useEffect(() => {
    if (isAuthenticated) router.replace(destination);
  }, [isAuthenticated, destination, router]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitLock.current) return;
    const normalizedEmail = email.trim();
    const nextErrors: { email?: string; password?: string; confirmation?: string } = {};
    if (!EMAIL_PATTERN.test(normalizedEmail)) nextErrors.email = "Enter a valid email address.";
    if (Array.from(password).length < 8) nextErrors.password = "Use at least 8 characters.";
    if (confirmation !== password) nextErrors.confirmation = "Passwords do not match.";
    setFieldErrors(nextErrors);
    setRequestError("");
    if (Object.keys(nextErrors).length) return;

    submitLock.current = true;
    setSubmitting(true);
    try {
      const result = await register({ email: normalizedEmail, password });
      if (typeof result.message !== "string" || !result.message.trim()) {
        throw new Error("Unexpected registration response.");
      }
      router.replace(`/login?registered=1&next=${encodeURIComponent(destination)}`);
    } catch (error) {
      setRequestError(authErrorMessage(error, "register"));
    } finally {
      submitLock.current = false;
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-form-content-inner">
      <div className="auth-page-heading">
        <span className="auth-form-eyebrow">A CLEARER WAY TO BUILD</span>
        <h1>Create your account</h1>
        <p>Start exploring data and building better models.</p>
      </div>

      {requestError && <AuthError message={requestError} />}

      <form className="auth-form auth-register-form" onSubmit={handleSubmit} noValidate>
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
          name="new-password"
          autoComplete="new-password"
          placeholder="Create a password"
          value={password}
          onChange={(event) => {
            setPassword(event.target.value);
            setFieldErrors((current) => ({ ...current, password: undefined }));
          }}
          error={fieldErrors.password}
          required
        />
        <PasswordStrength password={password} />
        <PasswordInput
          label="Confirm password"
          name="confirm-password"
          autoComplete="new-password"
          placeholder="Enter your password again"
          value={confirmation}
          onChange={(event) => {
            setConfirmation(event.target.value);
            setFieldErrors((current) => ({ ...current, confirmation: undefined }));
          }}
          error={fieldErrors.confirmation}
          required
        />
        <Button className="auth-submit" type="submit" loading={submitting}>Create account</Button>
      </form>

      <p className="auth-switch-copy">Already have an account? <Link href={`/login?next=${encodeURIComponent(destination)}`}>Sign in</Link></p>
    </div>
  );
}
