"use client";

import { useState } from "react";
import Image from "next/image";
import {
  ArrowUpRight, Check, CircleHelp, Cpu, Database, Download, FileText,
  FlaskConical, Layers3, Sparkles,
} from "lucide-react";
import { AppShell } from "@/components/layout/app-shell";
import { AuthLayout } from "@/features/auth/auth-layout";
import { Button } from "@/components/ui/button";
import { CheckboxField, PasswordInput, SelectField, TextInput, Toggle } from "@/components/ui/form-controls";
import { Alert, EmptyState, Skeleton, Spinner } from "@/components/ui/feedback";
import { Modal } from "@/components/ui/modal";
import { Badge, Card, CardDescription, CardHeader, CardTitle, Divider, MetricCard, StatusIndicator, Tooltip } from "@/components/ui/surfaces";
import { Tabs } from "@/components/ui/tabs";

const tabItems = [
  { id: "overview", label: "Overview", content: <p>This panel is a reusable tab foundation. It contains no connected workspace data.</p> },
  { id: "schema", label: "Schema", content: <p>Dataset and model details will be rendered here when a real session is connected.</p> },
  { id: "activity", label: "Activity", content: <p>Run events will be supplied by the backend in a later integration task.</p> },
];

export function DesignSystemPreview() {
  const [compact, setCompact] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);

  return (
    <AppShell actions={<span className="preview-pill"><span /> Development preview</span>}>
      <div className="preview-stack">
        <section className="preview-intro-grid" aria-labelledby="preview-intro-title">
          <div className="preview-intro-copy">
            <span className="section-overline"><Sparkles size={14} aria-hidden="true" /> SYSTEM OVERVIEW</span>
            <h2 id="preview-intro-title">A clearer space for<br /><span>data science work.</span></h2>
            <p>Reusable interface foundations for exploring datasets, evaluating models, and following every analysis run.</p>
            <div className="intro-actions">
              <Button leadingIcon={<Layers3 size={16} />} onClick={() => document.getElementById("components")?.scrollIntoView({ behavior: "smooth" })}>Explore components</Button>
              <Button variant="outline" leadingIcon={<CircleHelp size={16} />} onClick={() => setDialogOpen(true)}>About this preview</Button>
            </div>
            <div className="intro-note"><span><Check size={13} /></span> Built for real analysis workflows</div>
          </div>
          <div className="intro-illustration-card">
            <div className="illustration-caption"><span className="caption-dot" /> DATA → INSIGHT → MODEL</div>
            <Image src="/assets/illustrations/ai-network.svg" width={460} height={360} alt="Connected dataset, analysis, and model nodes" priority />
            <div className="illustration-footnote"><span>Composable workflow</span><ArrowUpRight size={15} aria-hidden="true" /></div>
          </div>
        </section>

        <section className="preview-section" aria-labelledby="tokens-title">
          <div className="section-heading"><div><p className="section-overline">01 · FOUNDATIONS</p><h2 id="tokens-title">Color & typography</h2></div><p>Quiet neutrals create focus; blue and violet mark key actions.</p></div>
          <Card className="token-card">
            <div className="token-swatches">
              <TokenSwatch name="Workspace" hex="#F8FAFC" className="swatch-workspace" />
              <TokenSwatch name="Surface" hex="#FFFFFF" className="swatch-surface" />
              <TokenSwatch name="Ink" hex="#0F172A" className="swatch-ink" />
              <TokenSwatch name="Primary blue" hex="#3B82F6" className="swatch-blue" />
              <TokenSwatch name="Secondary purple" hex="#8B5CF6" className="swatch-purple" />
            </div>
            <Divider />
            <div className="type-scale">
              <div><span>Page title</span><strong className="type-page">Intelligent data science</strong></div>
              <div><span>Section heading</span><strong className="type-section">Explore a dataset</strong></div>
              <div><span>Body copy</span><p>Turn raw data into measured, explainable outcomes.</p></div>
              <div><span>Caption / label</span><small className="type-caption">LAST UPDATED · A MOMENT AGO</small></div>
            </div>
          </Card>
        </section>

        <section className="preview-section" id="components" aria-labelledby="controls-title">
          <div className="section-heading"><div><p className="section-overline">02 · INTERACTION</p><h2 id="controls-title">Buttons & form controls</h2></div><p>Consistent states, labels, and visible keyboard focus.</p></div>
          <div className="preview-grid two-columns">
            <Card className="card-hover">
              <CardHeader><div><CardTitle>Buttons</CardTitle><CardDescription>Five action levels plus disabled and loading states.</CardDescription></div></CardHeader>
              <div className="button-showcase">
                <Button>Primary action</Button>
                <Button variant="secondary">Secondary</Button>
                <Button variant="outline">Outline</Button>
                <Button variant="ghost">Ghost</Button>
                <Button variant="destructive">Destructive</Button>
                <Button disabled>Disabled</Button>
                <Button loading>Saving</Button>
                <Button variant="outline" leadingIcon={<Download size={16} />}>Export</Button>
              </div>
              <Divider label="Keyboard focus is always visible" />
              <p className="helper-copy">Tab to any control to see the shared focus ring.</p>
            </Card>
            <Card>
              <CardHeader><div><CardTitle>Form elements</CardTitle><CardDescription>Labels stay connected to inputs and messages.</CardDescription></div></CardHeader>
              <div className="form-showcase">
                <TextInput label="Workspace name" placeholder="e.g. Research team" hint="A clear name helps identify this workspace." />
                <PasswordInput label="Password" placeholder="Enter a password" autoComplete="new-password" />
                <SelectField label="Analysis mode" options={[{ value: "simple", label: "Simple mode" }, { value: "advanced", label: "Advanced mode" }]} />
                <CheckboxField label="Keep this preference for the session" />
                <Toggle label="Compact density" description="A reusable preference toggle." checked={compact} onCheckedChange={setCompact} />
                <TextInput label="Email address" placeholder="name@example.com" defaultValue="not-an-email" error="Enter a valid email address." />
              </div>
            </Card>
          </div>
        </section>

        <section className="preview-section" aria-labelledby="surfaces-title">
          <div className="section-heading"><div><p className="section-overline">03 · INFORMATION</p><h2 id="surfaces-title">Surfaces & status</h2></div><p>Structured summaries without fabricated workspace data.</p></div>
          <div className="metric-grid">
            <MetricCard label="Analyses" value="—" detail="Live data appears after integration" icon={<FlaskConical size={17} />} />
            <MetricCard label="Datasets" value="—" detail="No dataset connected" icon={<Database size={17} />} />
            <MetricCard label="Models" value="—" detail="No model selected" icon={<Cpu size={17} />} />
          </div>
          <div className="preview-grid two-columns surface-grid">
            <Card>
              <CardHeader><div><CardTitle>Badges & statuses</CardTitle><CardDescription>Semantic labels pair color with readable text.</CardDescription></div></CardHeader>
              <div className="status-showcase">
                <Badge>Draft</Badge><Badge tone="blue">Queued</Badge><Badge tone="purple">Advanced</Badge><Badge tone="green">Completed</Badge><Badge tone="amber">Waiting</Badge><Badge tone="red">Failed</Badge>
              </div>
              <Divider />
              <div className="status-showcase">
                <StatusIndicator label="Success" tone="success" /><StatusIndicator label="Warning" tone="warning" /><StatusIndicator label="Error" tone="error" /><StatusIndicator label="Information" tone="info" />
              </div>
              <div className="tooltip-line"><span>Contextual help</span><Tooltip label="Explains a field or action in context."><span className="tooltip-help">?</span></Tooltip></div>
            </Card>
            <Card>
              <CardHeader><div><CardTitle>Tabs & divider</CardTitle><CardDescription>Keyboard-friendly tab selection with concise panels.</CardDescription></div></CardHeader>
              <Tabs items={tabItems} label="Preview component tabs" />
              <Divider label="Section divider" />
              <div className="inline-icons"><FileText size={17} aria-hidden="true" /><span>Reusable surfaces for future product pages</span></div>
            </Card>
          </div>
        </section>

        <section className="preview-section" aria-labelledby="feedback-title">
          <div className="section-heading"><div><p className="section-overline">04 · FEEDBACK</p><h2 id="feedback-title">Loading, empty & alerts</h2></div><p>Motion stays subtle and respects reduced-motion settings.</p></div>
          <div className="preview-grid two-columns">
            <Card>
              <CardHeader><div><CardTitle>Loading skeleton</CardTitle><CardDescription>Placeholders suggest structure without implying progress.</CardDescription></div></CardHeader>
              <div className="skeleton-demo"><Skeleton className="skeleton-avatar" /><div><Skeleton className="skeleton-line skeleton-wide" /><Skeleton className="skeleton-line skeleton-short" /></div><Spinner size="sm" label="Example loading indicator" /></div>
              <div className="skeleton-table"><Skeleton className="skeleton-line skeleton-full" /><Skeleton className="skeleton-line skeleton-full" /><Skeleton className="skeleton-line skeleton-full" /></div>
              <div className="chart-container" role="note">Chart surface foundation · no data loaded</div>
            </Card>
            <Card>
              <CardHeader><div><CardTitle>Empty state</CardTitle><CardDescription>Clear next steps with a small original SVG.</CardDescription></div></CardHeader>
              <EmptyState title="No dataset connected" description="A future session will show its real dataset profile here." action={<Button variant="outline" disabled>Connect a dataset</Button>} />
            </Card>
          </div>
          <div className="alert-stack">
            <Alert tone="success" title="Success">The interface can acknowledge a completed action.</Alert>
            <Alert tone="warning" title="Needs attention">A clear message explains what needs review.</Alert>
            <Alert tone="error" title="Something went wrong">An error is described with text, not color alone.</Alert>
            <Alert tone="info" title="Information">Helpful context appears close to the relevant content.</Alert>
          </div>
        </section>

        <section className="preview-section" aria-labelledby="shell-title">
          <div className="section-heading"><div><p className="section-overline">05 · LAYOUT</p><h2 id="shell-title">Workspace & visual foundations</h2></div><p>Shared shell, future auth composition, and original SVG assets.</p></div>
          <div className="preview-grid two-columns">
            <Card className="workflow-card">
              <CardHeader><div><CardTitle>Analysis flow illustration</CardTitle><CardDescription>Original SVG prepared for future data and training flows.</CardDescription></div></CardHeader>
              <Image src="/assets/illustrations/workflow.svg" width={440} height={230} alt="Three connected stages representing data preparation, model evaluation, and a reusable result" />
              <div className="workflow-caption"><span><Database size={15} /> Prepare</span><span><Cpu size={15} /> Evaluate</span><span><FileText size={15} /> Reuse</span></div>
            </Card>
            <Card className="auth-foundation-card">
              <CardHeader><div><CardTitle>Authentication layout foundation</CardTitle><CardDescription>Responsive split-screen composition reserved for Task 16.</CardDescription></div></CardHeader>
              <AuthLayout preview>
                <div className="auth-preview-form">
                  <span className="section-overline">WELCOME</span>
                  <h3>Sign in to your workspace</h3>
                  <p>The form panel is ready for the future auth flow.</p>
                  <TextInput label="Email address" placeholder="you@example.com" />
                  <Button disabled>Continue</Button>
                </div>
              </AuthLayout>
            </Card>
          </div>
        </section>

        <footer className="preview-footer"><span>AutoML Agent</span><span>Intelligent Data Science Workspace</span><span>Foundation · 01</span></footer>
      </div>

      <Modal open={dialogOpen} onClose={() => setDialogOpen(false)} title="About this preview">
        <p>This development-only page demonstrates shared visual foundations. It contains no live metrics, user data, or backend integration.</p>
        <div className="modal-actions"><Button variant="outline" onClick={() => setDialogOpen(false)}>Close preview note</Button></div>
      </Modal>
    </AppShell>
  );
}

function TokenSwatch({ name, hex, className }: { name: string; hex: string; className: string }) {
  return <div className="token-swatch"><span className={className} /><div><strong>{name}</strong><small>{hex}</small></div></div>;
}
