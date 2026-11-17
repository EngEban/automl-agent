"use client";

import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { Alert } from "@/components/ui/feedback";
import { Button } from "@/components/ui/button";
import { ApiError, ApiTimeoutError, apiRequest } from "@/lib/api/client";
import type { SessionDetails } from "@/types/api";

type TaskType = "classification" | "regression";

interface CategoryFrequency {
  category: string;
  count: number;
  percentage: number;
}

interface FeatureColumn {
  name: string;
  dtype: string;
  detected_type: string;
  missing_count: number;
  missing_percentage: number;
  unique_count: number;
  is_constant: boolean;
  is_near_constant: boolean;
  likely_id: boolean;
  duplicate_of: string | null;
  potential_issues: string[];
  numeric_statistics: Record<string, number | null> | null;
  top_categories: CategoryFrequency[];
}

interface FeatureMatrix {
  features: string[];
  pearson: (number | null)[][];
  spearman: (number | null)[][];
  strong_pairs: Array<{
    feature_a: string;
    feature_b: string;
    pearson: number | null;
    spearman: number | null;
    strength: number;
  }>;
  omitted_numeric_feature_count: number;
  status: string;
  notes?: string[];
}

interface SelectionMethod {
  name: string;
  type: string;
  description: string;
  requirements: string;
  status: string;
  message: string | null;
  selected_features: string[];
  selected_count: number;
  ranking: Array<{ feature: string; score: number }>;
  excluded_features: Array<{ feature: string; reason: string }>;
  duration_ms: number | null;
  limitations: string[];
}

interface SelectionComparison {
  status: string;
  target_column?: string;
  task_type?: string;
  message?: string;
  training_rows_used?: number;
  holdout_rows_reserved?: number;
  holdout_rows_used_for_selection?: number;
  max_features_per_ranked_method?: number;
  methods: SelectionMethod[];
  agreement: Array<{ feature: string; method_count: number; method_total: number }>;
  notes?: string[];
}

interface FeatureAnalysisResponse {
  n_rows: number;
  n_columns: number;
  profiled_rows: number;
  sampled: boolean;
  profile: {
    quality_score: number;
    missing_count: number;
    duplicate_row_count: number;
    warnings: string[];
    columns: FeatureColumn[];
  };
  relationships: FeatureMatrix;
  selection_comparison: SelectionComparison;
}

interface FeatureAnalysisRequest {
  target_column?: string | null;
  task_type?: TaskType | null;
  max_features: number;
}

function columnNames(session: SessionDetails): string[] {
  const value = session.dataset_meta?.columns;
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (typeof item === "string") return [item];
    if (typeof item === "object" && item !== null && !Array.isArray(item)) {
      const name = (item as Record<string, unknown>).name;
      return typeof name === "string" ? [name] : [];
    }
    return [];
  });
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiTimeoutError) return "Feature analysis timed out. Try again with this dataset later.";
  if (error instanceof ApiError && error.status >= 500) return "The service could not analyze this dataset. Try again shortly.";
  if (error instanceof TypeError) return "We could not reach the service. Check your connection and try again.";
  return error instanceof ApiError ? error.message : "Feature analysis could not be completed.";
}

function correlationColor(value: number | null): string | undefined {
  if (value === null) return undefined;
  const alpha = Math.min(0.7, Math.abs(value) * 0.55);
  return value >= 0
    ? `rgba(59, 130, 246, ${alpha})`
    : `rgba(139, 92, 246, ${alpha})`;
}

function FeatureOverview({ columns }: { columns: FeatureColumn[] }) {
  const [page, setPage] = useState(0);
  const pageSize = 12;
  const pageCount = Math.max(1, Math.ceil(columns.length / pageSize));
  const visible = columns.slice(page * pageSize, (page + 1) * pageSize);

  return (
    <section className="feature-explorer-section" aria-labelledby="feature-overview-title">
      <div className="feature-explorer-section-heading">
        <div><span className="section-overline">01 · FEATURE OVERVIEW</span><h4 id="feature-overview-title">Feature quality and distributions</h4></div>
        <span>{columns.length.toLocaleString()} returned</span>
      </div>
      <div className="feature-explorer-table-scroll">
        <table className="dashboard-table feature-explorer-table">
          <thead><tr><th scope="col">Feature</th><th scope="col">Type</th><th scope="col">Missing</th><th scope="col">Unique</th><th scope="col">Summary</th></tr></thead>
          <tbody>
            {visible.map((column) => (
              <tr key={column.name}>
                <th scope="row">
                  <span>{column.name}</span>
                  {column.potential_issues.length > 0 && <small>{column.potential_issues.slice(0, 2).join(" ")}</small>}
                </th>
                <td>{column.detected_type}</td>
                <td>{column.missing_count.toLocaleString()} <small>({column.missing_percentage.toFixed(1)}%)</small></td>
                <td>{column.unique_count.toLocaleString()}{column.is_constant && <small> · constant</small>}{column.is_near_constant && <small> · near constant</small>}</td>
                <td>
                  {column.numeric_statistics ? (
                    <span>Median {column.numeric_statistics.median ?? "—"} · mean {column.numeric_statistics.mean ?? "—"} · range {column.numeric_statistics.minimum ?? "—"} to {column.numeric_statistics.maximum ?? "—"}</span>
                  ) : column.top_categories.length > 0 ? (
                    <details className="feature-category-details">
                      <summary>Top {column.top_categories.length} categories</summary>
                      <ul>{column.top_categories.slice(0, 5).map((item) => <li key={item.category}>{item.category}: {item.count.toLocaleString()} ({item.percentage.toFixed(1)}%)</li>)}</ul>
                    </details>
                  ) : <span className="feature-explorer-muted">No summary available</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {pageCount > 1 && (
        <div className="feature-explorer-pagination" aria-live="polite">
          <span>Showing {page * pageSize + 1}–{Math.min((page + 1) * pageSize, columns.length)} of {columns.length}</span>
          <div>
            <Button type="button" variant="outline" disabled={page === 0} onClick={() => setPage((current) => Math.max(0, current - 1))}>Previous</Button>
            <Button type="button" variant="outline" disabled={page + 1 >= pageCount} onClick={() => setPage((current) => Math.min(pageCount - 1, current + 1))}>Next</Button>
          </div>
        </div>
      )}
    </section>
  );
}

function RelationshipSection({ matrix }: { matrix: FeatureMatrix }) {
  const [selectedPairKey, setSelectedPairKey] = useState("");
  const features = matrix.features.slice(0, 12);
  const selectedPair = matrix.strong_pairs.find((pair) => `${pair.feature_a}|${pair.feature_b}` === selectedPairKey)
    ?? matrix.strong_pairs[0];

  return (
    <section className="feature-explorer-section" aria-labelledby="feature-relationships-title">
      <div className="feature-explorer-section-heading">
        <div><span className="section-overline">02 · FEATURE RELATIONSHIPS</span><h4 id="feature-relationships-title">Numeric correlation</h4></div>
        <span>Pearson + Spearman</span>
      </div>
      {matrix.status !== "available" ? (
        <p className="feature-explorer-muted">At least two numeric features with sufficient observations are needed to calculate correlations.</p>
      ) : (
        <>
          <p className="feature-explorer-note">Pearson captures linear relationships; Spearman captures monotonic rank relationships. Values near −1 or +1 indicate stronger association, not causation.</p>
          <div className="feature-correlation-scroll">
            <table className="feature-correlation-table">
              <caption className="sr-only">Pearson correlation matrix for the first twelve numeric features</caption>
              <thead><tr><th scope="col">Feature</th>{features.map((feature) => <th scope="col" key={feature} title={feature}>{feature}</th>)}</tr></thead>
              <tbody>{features.map((feature, row) => (
                <tr key={feature}><th scope="row" title={feature}>{feature}</th>{features.map((other, column) => {
                  const value = matrix.pearson[row]?.[column] ?? null;
                  return <td key={other} style={{ backgroundColor: correlationColor(value) }} title={`${feature} and ${other}: ${value === null ? "insufficient data" : value.toFixed(3)}`}>{value === null ? "—" : value.toFixed(2)}</td>;
                })}</tr>
              ))}</tbody>
            </table>
          </div>
          {matrix.features.length > features.length && <p className="feature-explorer-muted">The matrix contains {matrix.features.length} numeric features; the table shows the first {features.length} for readability.</p>}
          {matrix.omitted_numeric_feature_count > 0 && <p className="feature-explorer-muted">{matrix.omitted_numeric_feature_count} more numeric features were omitted by the API response limit.</p>}
          {matrix.strong_pairs.length > 0 ? (
            <div className="feature-pair-detail">
              <label htmlFor="feature-pair-select">Inspect a strongly associated pair</label>
              <select id="feature-pair-select" value={selectedPairKey || `${selectedPair?.feature_a}|${selectedPair?.feature_b}`} onChange={(event) => setSelectedPairKey(event.target.value)}>
                {matrix.strong_pairs.slice(0, 50).map((pair) => <option key={`${pair.feature_a}|${pair.feature_b}`} value={`${pair.feature_a}|${pair.feature_b}`}>{pair.feature_a} ↔ {pair.feature_b}</option>)}
              </select>
              {selectedPair && <dl><div><dt>Pearson</dt><dd>{selectedPair.pearson?.toFixed(3) ?? "Insufficient data"}</dd></div><div><dt>Spearman</dt><dd>{selectedPair.spearman?.toFixed(3) ?? "Insufficient data"}</dd></div></dl>}
            </div>
          ) : <p className="feature-explorer-muted">No numeric pairs crossed the strong-association threshold (|r| ≥ 0.70).</p>}
          <p className="feature-explorer-note">Categorical association statistics are not implemented. Category frequencies are available in the feature overview.</p>
        </>
      )}
    </section>
  );
}

function SelectionSection({ comparison }: { comparison: SelectionComparison }) {
  return (
    <section className="feature-explorer-section" aria-labelledby="feature-selection-title">
      <div className="feature-explorer-section-heading">
        <div><span className="section-overline">03 · FEATURE SELECTION</span><h4 id="feature-selection-title">Training-only method preview</h4></div>
        {comparison.training_rows_used !== undefined && <span>{comparison.training_rows_used.toLocaleString()} training rows</span>}
      </div>
      {comparison.status !== "success" ? (
        <p className="feature-explorer-muted">{comparison.message ?? "Choose a valid target and task type to compare selectors."}</p>
      ) : (
        <>
          <p className="feature-explorer-note">The holdout was reserved before selection and did not contribute to selector fitting. This preview does not modify the dataset or a training run.</p>
          <div className="feature-selection-grid">
            {comparison.methods.map((method) => (
              <article className="feature-selection-method" key={method.name}>
                <div className="feature-selection-method-heading"><div><span>{method.type}</span><h5>{method.name}</h5></div><strong>{method.status === "success" ? `${method.selected_count} selected` : method.status}</strong></div>
                <p>{method.description}</p>
                {method.message && <p className="feature-explorer-muted">{method.message}</p>}
                {method.duration_ms !== null && <small>Completed in {method.duration_ms.toFixed(1)} ms</small>}
                {method.status === "success" && (
                  <>
                    <div className="feature-selection-chips">{method.selected_features.length > 0 ? method.selected_features.map((feature) => <span key={feature}>{feature}</span>) : <span>No features selected</span>}</div>
                    {method.ranking.length > 0 && <details><summary>View rankings and scores</summary><ol>{method.ranking.slice(0, 10).map((item) => <li key={item.feature}><span>{item.feature}</span><code>{item.score.toLocaleString(undefined, { maximumFractionDigits: 5 })}</code></li>)}</ol></details>}
                    {method.excluded_features.length > 0 && <details><summary>Excluded features ({method.excluded_features.length})</summary><ul>{method.excluded_features.slice(0, 12).map((item) => <li key={`${item.feature}:${item.reason}`}><strong>{item.feature}</strong>: {item.reason}</li>)}</ul></details>}
                    {method.limitations.map((limitation) => <small className="feature-explorer-note" key={limitation}>{limitation}</small>)}
                  </>
                )}
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function MethodComparison({ comparison }: { comparison: SelectionComparison }) {
  const successful = comparison.methods.filter((method) => method.status === "success");
  if (comparison.status !== "success") return null;

  return (
    <section className="feature-explorer-section" aria-labelledby="feature-method-comparison-title">
      <div className="feature-explorer-section-heading"><div><span className="section-overline">04 · METHOD COMPARISON</span><h4 id="feature-method-comparison-title">Selection agreement</h4></div><span>{successful.length} methods completed</span></div>
      <div className="feature-explorer-table-scroll">
        <table className="dashboard-table feature-method-comparison-table">
          <thead><tr><th scope="col">Method</th><th scope="col">Status</th><th scope="col">Selected</th><th scope="col">Features</th><th scope="col">Time</th></tr></thead>
          <tbody>{comparison.methods.map((method) => <tr key={method.name}><th scope="row">{method.name}</th><td>{method.status}</td><td>{method.selected_count}</td><td>{method.selected_features.join(", ") || "—"}</td><td>{method.duration_ms === null ? "—" : `${method.duration_ms.toFixed(1)} ms`}</td></tr>)}</tbody>
        </table>
      </div>
      <div className="feature-agreement-list">
        <h5>Features selected by multiple methods</h5>
        {comparison.agreement.length > 0 ? <ul>{comparison.agreement.map((item) => <li key={item.feature}><strong>{item.feature}</strong><span>{item.method_count} of {item.method_total} successful methods</span></li>)}</ul> : <p className="feature-explorer-muted">No feature was selected by more than one completed method.</p>}
      </div>
      <p className="feature-explorer-note">Each method has its own score definition. Rankings and scores are not directly comparable across methods.</p>
    </section>
  );
}

export function FeatureExplorationPanel({
  session,
  token,
  onUnauthorized,
}: {
  session: SessionDetails;
  token: string;
  onUnauthorized: () => void;
}) {
  const columns = useMemo(() => columnNames(session), [session]);
  const initialTarget = columns.includes(session.target_column ?? "") ? session.target_column ?? "" : "";
  const [targetColumn, setTargetColumn] = useState(initialTarget);
  const [taskType, setTaskType] = useState<TaskType | "">(
    session.task_type === "regression" || session.task_type === "classification" ? session.task_type : "",
  );
  const [maxFeatures, setMaxFeatures] = useState(10);
  const [result, setResult] = useState<FeatureAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function runAnalysis(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    const request: FeatureAnalysisRequest = {
      target_column: targetColumn || null,
      task_type: taskType || null,
      max_features: maxFeatures,
    };
    try {
      const analysis = await apiRequest<FeatureAnalysisResponse, FeatureAnalysisRequest>(
        `/sessions/${encodeURIComponent(session.id)}/feature-analysis`,
        { method: "POST", token, body: request, timeoutMs: 60_000 },
      );
      setResult(analysis);
    } catch (requestError) {
      if (requestError instanceof ApiError && (requestError.status === 401 || requestError.status === 403)) {
        onUnauthorized();
        return;
      }
      setError(errorMessage(requestError));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="feature-exploration-panel" aria-labelledby="feature-exploration-title">
      <div className="feature-exploration-header">
        <div><span className="section-overline">DATA UNDERSTANDING</span><h3 id="feature-exploration-title">Feature exploration</h3><p>Review feature quality, numeric relationships, and training-only selection previews.</p></div>
      </div>
      <form className="feature-exploration-controls" onSubmit={runAnalysis}>
        <label>Target column
          <select value={targetColumn} onChange={(event) => setTargetColumn(event.target.value)}>
            <option value="">Exploration only</option>
            {columns.map((column) => <option value={column} key={column}>{column}</option>)}
          </select>
        </label>
        <label>Task type
          <select value={taskType} onChange={(event) => setTaskType(event.target.value as TaskType)}>
            <option value="">Choose task type</option>
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
          </select>
        </label>
        <label>Top features per ranked method
          <select value={maxFeatures} onChange={(event) => setMaxFeatures(Number(event.target.value))}>
            {[5, 10, 15, 20].map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
        <Button type="submit" loading={loading} disabled={loading || columns.length === 0}>Analyze features</Button>
      </form>
      {error && <Alert tone="error" title="Feature analysis failed">{error}</Alert>}
      {result && (
        <div className="feature-exploration-results" aria-live="polite">
          <div className="feature-exploration-summary">
            <div><span>Rows</span><strong>{result.n_rows.toLocaleString()}</strong></div>
            <div><span>Features</span><strong>{result.n_columns.toLocaleString()}</strong></div>
            <div><span>Missing values</span><strong>{result.profile.missing_count.toLocaleString()}</strong></div>
            <div><span>Duplicate rows</span><strong>{result.profile.duplicate_row_count.toLocaleString()}</strong></div>
            <div><span>Quality score</span><strong>{result.profile.quality_score.toFixed(1)}%</strong></div>
          </div>
          {result.sampled && <p className="feature-explorer-note">Profile and relationship statistics use {result.profiled_rows.toLocaleString()} deterministically sampled rows.</p>}
          {result.profile.warnings.slice(0, 5).map((warning) => <p className="feature-explorer-warning" key={warning}>{warning}</p>)}
          <FeatureOverview columns={result.profile.columns} />
          <RelationshipSection matrix={result.relationships} />
          <SelectionSection comparison={result.selection_comparison} />
          <MethodComparison comparison={result.selection_comparison} />
        </div>
      )}
    </section>
  );
}
