import type { ReactNode } from "react";
import Image from "next/image";
import Link from "next/link";
import { BrandMark } from "@/components/shared/brand-mark";
import { cn } from "@/lib/utils";

export function AuthLayout({ children, preview = false }: { children: ReactNode; preview?: boolean }) {
  return (
    <div className={cn("auth-layout", preview && "auth-layout-preview")}>
      <section className="auth-form-panel" aria-label="Authentication form layout">
        <Link className="auth-brand" href="/" aria-label="AutoML Agent home"><BrandMark /></Link>
        <div className="auth-form-content">{children}</div>
        <p className="auth-footer">AutoML Agent · Intelligent Data Science Workspace</p>
      </section>
      <aside className="auth-visual-panel" aria-label="Workspace illustration">
        <div className="auth-visual-orb auth-orb-one" aria-hidden="true" />
        <div className="auth-visual-orb auth-orb-two" aria-hidden="true" />
        <div className="auth-visual-copy">
          <span className="auth-kicker"><span /> DATA SCIENCE, IN FOCUS</span>
          <h2>From raw data<br />to clear decisions.</h2>
          <p>Your intelligent workspace for data science. Explore datasets, evaluate models, and understand every run.</p>
        </div>
        <Image
          className="auth-illustration"
          src="/images/automl-agent-illustrations/06-authentication.png"
          width={848}
          height={317}
          alt="Illustration of a secure sign-in form"
          priority
        />
        <span className="auth-art-caption"><span /> An intelligent workspace for careful analysis</span>
      </aside>
    </div>
  );
}
