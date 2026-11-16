import type { Metadata } from "next";
import { AuthLayout } from "@/features/auth/auth-layout";
import { LoginForm } from "@/features/auth/login-form";
import { firstAuthSearchParam, safeAuthDestination } from "@/features/auth/auth-utils";

export const metadata: Metadata = { title: "Sign in" };

type LoginSearchParams = Record<string, string | string[] | undefined>;

export default async function LoginPage({ searchParams }: { searchParams: Promise<LoginSearchParams> }) {
  const params = await searchParams;
  return (
    <main className="auth-page">
      <AuthLayout>
        <LoginForm
          destination={safeAuthDestination(firstAuthSearchParam(params.next))}
          registered={firstAuthSearchParam(params.registered) === "1"}
          verified={firstAuthSearchParam(params.verified) === "true"}
        />
      </AuthLayout>
    </main>
  );
}
