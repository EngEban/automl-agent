"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import {
  Activity, Boxes, ChartNoAxesCombined, ChevronRight, Command,
  FileBarChart, FolderKanban, LayoutDashboard, Menu, PanelLeftClose,
  Settings2, Sparkles, X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { BrandMark } from "@/components/shared/brand-mark";
import { useAuth } from "@/features/auth/auth-provider";

interface NavigationItem {
  label: string;
  icon: LucideIcon;
  href?: string;
  active?: boolean;
}

const navigationGroups: Array<{ label: string; items: NavigationItem[] }> = [
  { label: "Workspace", items: [
    { label: "Overview", icon: LayoutDashboard, href: "/workspace" },
    { label: "AI Workspace", icon: Sparkles, href: "/workspace/agent" },
    { label: "Datasets", icon: FolderKanban, href: "/workspace/datasets" },
  ] },
  { label: "Intelligence", items: [
    { label: "Results", icon: ChartNoAxesCombined, href: "/workspace/results" },
    { label: "Runs", icon: Activity },
    { label: "Models", icon: Boxes },
    { label: "Reports", icon: FileBarChart, href: "/workspace/reports" },
  ] },
];

export function SidebarNavItem({ item, collapsed = false, onNavigate }: { item: NavigationItem; collapsed?: boolean; onNavigate?: () => void }) {
  const Icon = item.icon;
  const className = cn("sidebar-item", item.active && "sidebar-item-active", collapsed && "sidebar-item-collapsed");
  const content = <><Icon size={18} strokeWidth={1.8} aria-hidden="true" /><span>{item.label}</span></>;

  if (!item.href) {
    return <button className={cn(className, "sidebar-item-placeholder")} type="button" disabled aria-label={item.label} title={`${item.label} will be available in a later task`}>
      {content}
      <span className="sidebar-soon">Soon</span>
    </button>;
  }

  return <Link className={className} href={item.href} onClick={onNavigate} aria-label={collapsed ? item.label : undefined} aria-current={item.active ? "page" : undefined} title={collapsed ? item.label : undefined}>
    {content}
  </Link>;
}

function WorkspaceSidebar({ collapsed, onToggle, onNavigate, activeItem, homeHref }: { collapsed: boolean; onToggle: () => void; onNavigate: () => void; activeItem?: string; homeHref: string }) {
  return (
    <aside className={cn("app-sidebar", collapsed && "is-collapsed")} aria-label="Application navigation">
      <div className="sidebar-brand-row">
        <Link className="sidebar-brand" href={homeHref} onClick={onNavigate} aria-label="AutoML Agent home">
          <BrandMark compact={collapsed} />
          {collapsed && <span className="sr-only">AutoML Agent</span>}
        </Link>
        <button className="icon-button sidebar-collapse" type="button" aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"} aria-expanded={!collapsed} onClick={onToggle}>
          {collapsed ? <ChevronRight size={17} /> : <PanelLeftClose size={17} />}
        </button>
      </div>
      <div className="workspace-switcher" title="AutoML Agent workspace">
        <span className="workspace-avatar"><Command size={16} aria-hidden="true" /></span>
        <span className="workspace-copy"><strong>AutoML Workspace</strong><small>Personal workspace</small></span>
        <ChevronRight className="workspace-chevron" size={15} aria-hidden="true" />
      </div>
      <nav className="sidebar-navigation" aria-label="Primary navigation">
        {navigationGroups.map((group) => (
          <div className="sidebar-group" key={group.label}>
            <p className="sidebar-group-label">{group.label}</p>
            {group.items.map((item) => <SidebarNavItem key={item.label} item={{ ...item, active: item.label === activeItem }} collapsed={collapsed} onNavigate={onNavigate} />)}
          </div>
        ))}
      </nav>
      <div className="sidebar-footer">
        <SidebarNavItem
          item={{ label: "Settings", icon: Settings2, href: "/workspace/settings/ai-providers", active: activeItem === "Settings" }}
          collapsed={collapsed}
          onNavigate={onNavigate}
        />
        <div className="sidebar-build-note"><span className="sidebar-build-dot" /> <span>AutoML Agent</span></div>
      </div>
    </aside>
  );
}

export interface BreadcrumbItem {
  label: string;
  href?: string;
}

export function Breadcrumbs({ items }: { items: BreadcrumbItem[] }) {
  return (
    <nav className="breadcrumbs" aria-label="Breadcrumb">
      <ol>
        {items.map((item, index) => (
          <li key={`${item.label}-${index}`}>
            {index > 0 && <ChevronRight size={13} aria-hidden="true" />}
            {item.href && index < items.length - 1
              ? <Link href={item.href}>{item.label}</Link>
              : <span aria-current={index === items.length - 1 ? "page" : undefined}>{item.label}</span>}
          </li>
        ))}
      </ol>
    </nav>
  );
}

function UserMenu() {
  const router = useRouter();
  const { isAuthenticated, user, logout } = useAuth();
  const accountLabel = user?.email ?? "Guest";
  const avatarLabel = user?.email.slice(0, 1).toUpperCase() ?? "A";

  function handleAccountAction() {
    if (isAuthenticated) {
      logout();
      router.replace("/login");
      return;
    }
    router.push("/login");
  }

  return (
    <details className="user-menu">
      <summary className="user-menu-trigger" aria-label="Open account menu">
        <span className="user-avatar" aria-hidden="true">{avatarLabel}</span>
        <span className="user-menu-copy"><strong>{isAuthenticated ? "Account" : "Guest"}</strong><small>{accountLabel}</small></span>
        <ChevronRight className="user-menu-chevron" size={15} aria-hidden="true" />
      </summary>
      <div className="user-menu-panel">
        <p>{isAuthenticated ? accountLabel : "Sign in to access your workspace."}</p>
        <button type="button" onClick={handleAccountAction}>{isAuthenticated ? "Sign out" : "Sign in"}</button>
      </div>
    </details>
  );
}

export function Header({
  breadcrumbs,
  mobileOpen,
  onToggleMobileNavigation,
  statusLabel = "Preview environment",
}: {
  breadcrumbs: BreadcrumbItem[];
  mobileOpen: boolean;
  onToggleMobileNavigation: () => void;
  statusLabel?: string;
}) {
  return (
    <header className="app-header">
      <button className="icon-button mobile-menu-button" type="button" aria-label={mobileOpen ? "Close navigation" : "Open navigation"} aria-expanded={mobileOpen} onClick={onToggleMobileNavigation}>
        {mobileOpen ? <X size={19} /> : <Menu size={19} />}
      </button>
      <div className="header-brand-mobile"><BrandMark compact /></div>
      <Breadcrumbs items={breadcrumbs} />
      <div className="header-tools"><span className="header-status"><span /> {statusLabel}</span><UserMenu /></div>
    </header>
  );
}

export function AppShell({
  children,
  title = "Design system",
  eyebrow = "FOUNDATION PREVIEW",
  description = "A shared visual language for the AutoML Agent workspace.",
  breadcrumbs = [{ label: "Workspace", href: "/design-preview" }, { label: "Design system" }],
  activeItem,
  homeHref = "/",
  statusLabel = "Preview environment",
  actions,
}: {
  children: ReactNode;
  title?: string;
  eyebrow?: string;
  description?: string;
  breadcrumbs?: BreadcrumbItem[];
  activeItem?: string;
  homeHref?: string;
  statusLabel?: string;
  actions?: ReactNode;
}) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, []);

  return (
    <div className={cn("app-frame", collapsed && "sidebar-collapsed", mobileOpen && "mobile-nav-open")}>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <button className="mobile-backdrop" type="button" onClick={() => setMobileOpen(false)} aria-label="Close navigation" tabIndex={mobileOpen ? 0 : -1} />
      <WorkspaceSidebar collapsed={collapsed} onToggle={() => setCollapsed((value) => !value)} onNavigate={() => setMobileOpen(false)} activeItem={activeItem} homeHref={homeHref} />
      <div className="app-main-column">
        <Header breadcrumbs={breadcrumbs} mobileOpen={mobileOpen} onToggleMobileNavigation={() => setMobileOpen((value) => !value)} statusLabel={statusLabel} />
        <main id="main-content" className="app-page-content" tabIndex={-1}>
          <PageHeader title={title} eyebrow={eyebrow} description={description} actions={actions} />
          <div className="app-content-inner">{children}</div>
        </main>
      </div>
    </div>
  );
}

export function PageHeader({
  title,
  eyebrow,
  description,
  actions,
}: {
  title: string;
  eyebrow?: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        {eyebrow && <p className="page-eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        {description && <p className="page-description">{description}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </div>
  );
}
