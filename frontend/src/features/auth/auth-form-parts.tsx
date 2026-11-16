"use client";

import { Alert } from "@/components/ui/feedback";

export function PasswordStrength({ password }: { password: string }) {
  const score = !password
    ? 0
    : Math.min(3, (password.length >= 8 ? 1 : 0) + (password.length >= 12 ? 1 : 0) + (/[a-z]/.test(password) && /[A-Z0-9\W]/.test(password) ? 1 : 0));
  const labels = ["Add a password", "Getting started", "Good", "Strong"];

  return (
    <div className="password-strength" aria-live="polite">
      <div className="password-strength-heading">
        <span>Password strength</span>
        <span>{labels[score]}</span>
      </div>
      <div className="password-strength-bars" role="progressbar" aria-label="Password strength" aria-valuemin={0} aria-valuemax={3} aria-valuenow={score}>
        {[1, 2, 3].map((bar) => <span className={bar <= score ? `strength-fill strength-${score}` : ""} key={bar} />)}
      </div>
      <p>Use at least 8 characters. A longer password with a mix of characters is stronger.</p>
    </div>
  );
}

export function AuthError({ message }: { message: string }) {
  return <Alert tone="error" title="We couldn’t complete that request">{message}</Alert>;
}

export function AuthNotice({ title, children }: { title: string; children: string }) {
  return <Alert tone="success" title={title}>{children}</Alert>;
}
