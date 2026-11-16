import { ProtectedRoute } from "@/features/auth/protected-route";
import { ResultsPage } from "@/features/results/results-page";

type SearchParams = Record<string, string | string[] | undefined>;

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function validatedQueryValue(value: string | string[] | undefined): { id: string | null; invalid: boolean } {
  const candidate = Array.isArray(value) ? value[0] : value;
  if (!candidate) return { id: null, invalid: false };
  return UUID_PATTERN.test(candidate)
    ? { id: candidate, invalid: false }
    : { id: null, invalid: true };
}

export default async function WorkspaceResultsRoute({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const params = await searchParams;
  const session = validatedQueryValue(params.session_id);
  const run = validatedQueryValue(params.run_id);
  const invalidSelection = session.invalid || run.invalid;

  return (
    <ProtectedRoute>
      <ResultsPage
        initialSessionId={invalidSelection ? null : session.id}
        initialRunId={invalidSelection ? null : run.id}
        selectionError={invalidSelection ? "The session or run link contains an invalid identifier. Select a session to continue." : ""}
      />
    </ProtectedRoute>
  );
}
