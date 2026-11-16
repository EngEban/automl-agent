import { cn } from "@/lib/utils";

export function Spinner({ size = "md", label }: { size?: "sm" | "md" | "lg"; label?: string }) {
  return <span className={cn("spinner", `spinner-${size}`)} role="status" aria-label={label ?? "Loading"} />;
}
