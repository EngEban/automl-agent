import type { Metadata } from "next";
import { AuthLayout } from "@/features/auth/auth-layout";
import { RegisterForm } from "@/features/auth/register-form";
import { firstAuthSearchParam, safeAuthDestination } from "@/features/auth/auth-utils";

export const metadata: Metadata = { title: "Create account" };

type RegisterSearchParams = Record<string, string | string[] | undefined>;

export default async function RegisterPage({ searchParams }: { searchParams: Promise<RegisterSearchParams> }) {
  const params = await searchParams;
  return (
    <main className="auth-page">
      <AuthLayout>
        <RegisterForm destination={safeAuthDestination(firstAuthSearchParam(params.next))} />
      </AuthLayout>
    </main>
  );
}
