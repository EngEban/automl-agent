import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { AuthProvider } from "@/features/auth/auth-provider";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "AutoML Agent — Intelligent Data Science Workspace",
    template: "%s · AutoML Agent",
  },
  description: "A thoughtful workspace for dataset exploration, model evaluation, and reliable analysis runs.",
  applicationName: "AutoML Agent",
};

export const viewport: Viewport = {
  themeColor: "#F8FAFC",
  colorScheme: "light",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-workspace font-sans text-ink antialiased">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
