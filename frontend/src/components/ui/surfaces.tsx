import type { HTMLAttributes, ReactNode } from "react";
import { Info } from "lucide-react";
import { cn } from "@/lib/utils";

export function Card({ className, ...props }: HTMLAttributes<HTMLElement>) {
  return <section className={cn("card", className)} {...props} />;
}

export function CardHeader({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("card-header", className)} {...props} />;
}

export function CardTitle({ className, ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return <h3 className={cn("card-title", className)} {...props} />;
}

export function CardDescription({ className, ...props }: HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn("card-description", className)} {...props} />;
}

export interface MetricCardProps {
  label: string;
  value: string;
  detail?: string;
  icon?: ReactNode;
}

export function MetricCard({ label, value, detail, icon }: MetricCardProps) {
  return (
    <Card className="metric-card">
      <div className="metric-topline"><span>{label}</span>{icon && <span className="metric-icon">{icon}</span>}</div>
      <strong className="metric-value">{value}</strong>
      {detail && <span className="metric-detail">{detail}</span>}
    </Card>
  );
}

export type BadgeTone = "neutral" | "blue" | "purple" | "green" | "amber" | "red";

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: BadgeTone }) {
  return <span className={cn("badge", `badge-${tone}`)}>{children}</span>;
}

export type StatusTone = "success" | "warning" | "error" | "info" | "neutral";

export function StatusIndicator({ label, tone = "neutral" }: { label: string; tone?: StatusTone }) {
  return (
    <span className={cn("status-indicator", `status-${tone}`)}>
      <span className="status-dot" aria-hidden="true" />
      {label}
    </span>
  );
}

export function Divider({ label }: { label?: string }) {
  if (!label) return <hr className="divider" />;
  return <div className="divider-with-label"><span /> <span>{label}</span> <span /></div>;
}

export function Tooltip({ label, children }: { label: string; children: ReactNode }) {
  return <span className="tooltip" role="note" tabIndex={0} data-tooltip={label} aria-label={label}>{children}<Info size={14} aria-hidden="true" /></span>;
}
