import { ProtectedRoute } from "@/features/auth/protected-route";
import { ReportsPage } from "@/features/reports/reports-page";

type SearchParams = Record<string, string | string[] | undefined>;

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function validatedQueryValue(value: string | string[] | undefined): { id: string | null; invalid: boolean } {
  const candidate = Array.isArray(value) ? value[0] : value;
  if (!candidate) return { id: null, invalid: false };
  return UUID_PATTERN.test(candidate) ? { id: candidate, invalid: false } : { id: null, invalid: true };
}

export default async function WorkspaceReportsRoute({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const params = await searchParams;
  const session = validatedQueryValue(params.session_id);
  const run = validatedQueryValue(params.run_id);

  return (
    <ProtectedRoute>
      <ReportsPage
        key={`${session.id ?? "all-sessions"}:${run.id ?? "no-run-context"}`}
        initialSessionId={session.id}
        contextRunId={run.id}
        selectionError={session.invalid || run.invalid ? "The session or run link contains an invalid identifier. Showing reports across your sessions instead." : ""}
      />
    </ProtectedRoute>
  );
}
