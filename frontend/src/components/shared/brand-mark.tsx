import { cn } from "@/lib/utils";

export function BrandMark({ compact = false }: { compact?: boolean }) {
  return (
    <span className={cn("brand-lockup", compact && "brand-compact")}>
      <span className="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 32 32" fill="none">
          <path d="M8 9.5 16 6l8 5v9.5l-8 5-8-5V9.5Z" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
          <path d="m8.5 10 7.5 4.5 8-3.5M16 14.5v10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          <circle cx="8" cy="9.5" r="2.4" fill="#3B82F6" stroke="white" strokeWidth="1.2" />
          <circle cx="24" cy="11" r="2.4" fill="#8B5CF6" stroke="white" strokeWidth="1.2" />
          <circle cx="16" cy="25" r="2.4" fill="#3B82F6" stroke="white" strokeWidth="1.2" />
        </svg>
      </span>
      {!compact && (
        <span className="brand-copy">
          <strong>AutoML Agent</strong>
          <span>Intelligent Data Science Workspace</span>
        </span>
      )}
    </span>
  );
}
