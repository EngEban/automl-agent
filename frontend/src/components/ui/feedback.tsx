import type { ReactNode } from "react";
import Image from "next/image";
import { cn } from "@/lib/utils";
export { Spinner } from "./spinner";

export function Skeleton({ className }: { className?: string }) {
  return <span className={cn("skeleton", className)} aria-hidden="true" />;
}

export function EmptyState({
  title,
  description,
  action,
  illustration = "/assets/illustrations/dataset-empty.svg",
}: {
  title: string;
  description: string;
  action?: ReactNode;
  illustration?: string;
}) {
  return (
    <div className="empty-state">
      <Image src={illustration} width={180} height={130} alt="" aria-hidden="true" />
      <h3>{title}</h3>
      <p>{description}</p>
      {action}
    </div>
  );
}

export type AlertTone = "success" | "warning" | "error" | "info";

export function Alert({ tone, title, children }: { tone: AlertTone; title: string; children: ReactNode }) {
  return (
    <div className={cn("alert", `alert-${tone}`)} role={tone === "error" ? "alert" : "status"}>
      <span className="alert-mark" aria-hidden="true">{tone === "success" ? "✓" : tone === "error" ? "!" : tone === "warning" ? "!" : "i"}</span>
      <div><strong>{title}</strong><p>{children}</p></div>
    </div>
  );
}
